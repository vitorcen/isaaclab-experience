"""LoRA injection + npz I/O for PI05Policy.

The npz format uses three short prefixes (`vision`, `vlm`, `expert`) and
two top-level Linear layers (`action_in_proj`, `action_out_proj`).
PyTorch PI05Policy uses fully qualified paths under
`model.paligemma_with_expert.*`. The mapping is one-to-one — no shape
or transpose ambiguity, both encode (in_dim, out_dim, rank=16) the same
way (LoRA A: (r, in), B: (out, r); merged: W += scale * B @ A).
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
from torch import nn


# npz short prefix → PyTorch path (for the per-layer self_attn case)
_LAYER_PREFIX_MAP = {
    "vision.encoder.layers": "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers",
    "vlm.layers": "model.paligemma_with_expert.paligemma.model.language_model.layers",
    "expert.layers": "model.paligemma_with_expert.gemma_expert.model.layers",
}

# npz top-level → PyTorch Linear
_TOPLEVEL_MAP = {
    "action_in_proj": "model.action_in_proj",
    "action_out_proj": "model.action_out_proj",
}

_LAYER_PATTERN = re.compile(
    r"^(?P<prefix>vision\.encoder\.layers|vlm\.layers|expert\.layers)"
    r"\.(?P<idx>\d+)\.self_attn\.(?P<proj>q_proj|v_proj)\.lora_(?P<ab>A|B)$"
)
_TOPLEVEL_PATTERN = re.compile(
    r"^(?P<proj>action_in_proj|action_out_proj)\.lora_(?P<ab>A|B)$"
)


class LoRALinear(nn.Module):
    """Wrap an nn.Linear so its forward is `base(x) + scale * (x @ A^T) @ B^T`.

    npz convention:
        A: (rank, in_features)
        B: (out_features, rank)
    so `down(x) = x @ A.T` lands in rank-dim, then `up = down @ B.T`.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank), requires_grad=False)
        # freeze base too (we're inference-only here)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lerobot internals (pi_gemma) sometimes deliver fp32 activations
        # even when weights are bf16 — autocast inside the VLM forward
        # downgrades to compute dtype but returns fp32 on the way out.
        # Be dtype-tolerant rather than blow up at the next Linear.
        wdtype = self.base.weight.dtype
        if x.dtype != wdtype:
            x = x.to(wdtype)
        out = self.base(x)
        if self.scale != 0.0:
            down = torch.nn.functional.linear(x, self.lora_A)
            up = torch.nn.functional.linear(down, self.lora_B)
            out = out + self.scale * up
        return out

    # --- nn.Linear compatibility shim ----------------------------------------
    # Upstream lerobot code occasionally pokes at `self_attn.q_proj.weight`
    # or `.in_features` directly to introspect the model. Forward those
    # to the base Linear so wrapping is transparent for non-forward access.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _replace_module(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _resolve_pt_name(npz_key: str) -> tuple[str, str] | None:
    """Return (pytorch_linear_path, 'A'|'B') for an npz key, or None to skip."""
    m = _LAYER_PATTERN.match(npz_key)
    if m:
        prefix = _LAYER_PREFIX_MAP[m.group("prefix")]
        pt_path = f"{prefix}.{m.group('idx')}.self_attn.{m.group('proj')}"
        return pt_path, m.group("ab")
    m = _TOPLEVEL_PATTERN.match(npz_key)
    if m:
        pt_path = _TOPLEVEL_MAP[m.group("proj")]
        return pt_path, m.group("ab")
    return None


def wrap_pi05_with_lora(
    policy: nn.Module, *, rank: int = 16, alpha: float = 16.0
) -> dict[str, LoRALinear]:
    """In-place wrap every target Linear in `policy` with LoRALinear.

    Returns a dict {pt_path: wrapper} for the caller's bookkeeping.
    Targets:
      - q_proj, v_proj inside every transformer block of
        vision_tower / language_model / gemma_expert
      - action_in_proj, action_out_proj (top-level Linears on `model`)
    """
    targets = {"q_proj", "v_proj", "action_in_proj", "action_out_proj"}
    to_wrap: list[tuple[str, nn.Linear]] = []
    already_wrapped: dict[str, LoRALinear] = {}
    for name, module in policy.named_modules():
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        if isinstance(module, LoRALinear):
            # Idempotent: a second call should return the existing wraps so
            # callers (e.g. load_lora_npz on a continuation run) can
            # still inject weights into them.
            already_wrapped[name] = module
        elif isinstance(module, nn.Linear):
            to_wrap.append((name, module))

    wrapped: dict[str, LoRALinear] = dict(already_wrapped)
    for name, lin in to_wrap:
        wrap = LoRALinear(lin, rank=rank, alpha=alpha).to(
            device=lin.weight.device, dtype=lin.weight.dtype
        )
        _replace_module(policy, name, wrap)
        wrapped[name] = wrap
    return wrapped


def load_lora_npz(
    policy: nn.Module, npz_path: str, *, rank: int = 16, alpha: float = 16.0
) -> dict[str, list[str]]:
    """Load `final_lora.npz` into a PI05Policy.

    Wraps target layers if not already wrapped, then injects A/B.
    Returns {"loaded": [...], "skipped": [...], "missing": [...]} for diagnosis.
    """
    z = np.load(npz_path)
    wrapped = wrap_pi05_with_lora(policy, rank=rank, alpha=alpha)

    name_to_wrap = wrapped  # alias
    keys_by_pt: dict[str, dict[str, np.ndarray]] = {}
    skipped: list[str] = []
    for npz_key in z.files:
        resolved = _resolve_pt_name(npz_key)
        if resolved is None:
            skipped.append(npz_key)
            continue
        pt_path, ab = resolved
        keys_by_pt.setdefault(pt_path, {})[ab] = z[npz_key]

    loaded: list[str] = []
    missing: list[str] = []
    for pt_path, wrap in name_to_wrap.items():
        ab = keys_by_pt.get(pt_path)
        if ab is None:
            missing.append(pt_path)
            continue
        a = ab.get("A")
        b = ab.get("B")
        if a is None or b is None:
            missing.append(pt_path)
            continue
        # Shape sanity
        if a.shape != tuple(wrap.lora_A.shape) or b.shape != tuple(wrap.lora_B.shape):
            raise ValueError(
                f"Shape mismatch at {pt_path}: "
                f"npz A {a.shape} vs param {tuple(wrap.lora_A.shape)}; "
                f"npz B {b.shape} vs param {tuple(wrap.lora_B.shape)}"
            )
        with torch.no_grad():
            wrap.lora_A.copy_(torch.from_numpy(a).to(wrap.lora_A))
            wrap.lora_B.copy_(torch.from_numpy(b).to(wrap.lora_B))
        loaded.append(pt_path)

    return {"loaded": loaded, "skipped": skipped, "missing": missing}


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-id", default="lerobot/pi05_base")
    parser.add_argument(
        "--lora-npz", required=True, help="Path to trained final_lora.npz"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()

    # Optional: extend sys.path with a local lerobot fork (e.g. our DP-patched one)
    lerobot_src = os.environ.get("LEROBOT_SRC", "")
    if lerobot_src and os.path.isdir(lerobot_src):
        sys.path.insert(0, lerobot_src)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"loading PI05Policy from {args.policy_id} ...", flush=True)
    pol = PI05Policy.from_pretrained(args.policy_id).cuda().eval()

    print("wrapping + loading LoRA ...", flush=True)
    report = load_lora_npz(pol, args.lora_npz, rank=args.rank, alpha=args.alpha)
    print(
        f"loaded={len(report['loaded'])} "
        f"missing={len(report['missing'])} "
        f"skipped={len(report['skipped'])}"
    )
    if report["missing"]:
        print("missing pt_path examples:", report["missing"][:5])
    if report["skipped"]:
        print("skipped npz keys examples:", report["skipped"][:5])
