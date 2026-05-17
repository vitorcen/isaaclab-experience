"""OpenVLA-7B 4-bit quantized inference server for LeIsaac SO-101.

Wire-compatible with LeIsaac ``Pi05ServicePolicyClient`` so the same Isaac Sim
eval entry (`policy_inference.py --policy_type=pi05 --policy_port=5557`) can
hit either server.

Modules:
    server  — ZMQ + msgpack inference server, NF4 quantized (~4.4 GB GPU)
    smoke   — offline smoke test: load model, run a few prompts on one image
"""

__all__ = ["server", "smoke"]
