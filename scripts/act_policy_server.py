"""ACT policy server (runs in the lerobot env) for the closed-loop MimicKit eval.

The student (ACT, lerobot/torch) and the teacher env (MimicKit, isaaclab/IsaacSim) live
in different conda envs and cannot share a process. So ACT runs here behind a tiny TCP
socket: the MimicKit client (rollout_act_eval.py, isaaclab env) sends an observation each
control step and gets back a 29-DoF joint target.

Protocol (length-prefixed pickle, one round-trip per step):
  request : {"cmd": "reset"}                       -> {"ok": True}
            {"cmd": "act",
             "state": float32[95],
             "image": uint8[H,W,3],                 (env camera, HWC)
             "task":  str}                          -> {"action": float32[ndof]}
ACT's select_action manages the 16-step chunk queue internally; "reset" clears it at
episode boundaries.

  conda run -n lerobot python scripts/act_policy_server.py \
    --ckpt outputs/act_g1_lafan_sanity/checkpoints/last/pretrained_model --port 5599
"""

import argparse
import pickle
import socket
import struct

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy

IMG_KEY = "observation.images.front"


def nd_pack(a):
    """portable ndarray wire form — pickling ndarrays directly breaks across the
    numpy 1.x (isaaclab) / 2.x (lerobot) split (numpy.core vs numpy._core)."""
    a = np.ascontiguousarray(a)
    return {"__nd__": True, "b": a.tobytes(), "shape": list(a.shape), "dtype": str(a.dtype)}


def nd_unpack(d):
    return np.frombuffer(d["b"], dtype=d["dtype"]).reshape(d["shape"])


def _recv(conn):
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    (n,) = struct.unpack(">I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _send(conn, obj):
    buf = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(buf)) + buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="…/pretrained_model dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=None,
                    help="override execution horizon: how many actions to run open-loop "
                         "before replanning. The model still predicts chunk_size actions; "
                         "this only changes how many are used. =1 ⇒ replan every step "
                         "(tightest closed-loop, best for dynamic balance).")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.ckpt).to(args.device).eval()
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
        print(f"[act-server] n_action_steps overridden -> {args.n_action_steps}", flush=True)
    policy.reset()
    print(f"[act-server] loaded {args.ckpt}; listening on {args.host}:{args.port}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[act-server] client connected", flush=True)
        try:
            while True:
                req = _recv(conn)
                if req is None:
                    break
                if req.get("cmd") == "reset":
                    policy.reset()
                    _send(conn, {"ok": True})
                    continue
                # build batch in the same format the dataset yielded at train time:
                # state float32[95], image CHW float32 [0,1]
                state = torch.from_numpy(nd_unpack(req["state"]).astype(np.float32))[None].to(args.device)
                img = nd_unpack(req["image"]).astype(np.uint8)              # HWC uint8
                img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)[None].to(args.device)
                batch = {"observation.state": state, IMG_KEY: img, "task": [req["task"]]}
                with torch.no_grad():
                    action = policy.select_action(batch)
                _send(conn, {"action": nd_pack(action.squeeze(0).cpu().numpy().astype(np.float32))})
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("[act-server] client disconnected", flush=True)


if __name__ == "__main__":
    main()
