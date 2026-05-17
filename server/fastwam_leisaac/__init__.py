"""FastWAM (Wan2.2-5B DiT + 1B action expert) inference server for LeIsaac SO-101.

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient`` so the same Isaac Sim
eval entry (``policy_inference.py --policy_type=pi05 --policy_port=<port>``) can
hit either server.

Modules:
    server  — ZMQ + msgpack inference server, bf16 by default (~20 GB) or 4-bit (~8 GB)
    smoke   — offline smoke test: load model, run one synthetic 2-cam frame, time it
"""

__all__ = ["server", "smoke"]
