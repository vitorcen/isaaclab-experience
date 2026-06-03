"""Exit 0 iff every *.safetensors in the given checkpoint dir can be fully read.

Used by the self-healing resume loop to detect reboot/crash-truncated checkpoints
(a partial save throws SafetensorError on key enumeration).
"""
import sys, glob
from safetensors import safe_open
for f in glob.glob(sys.argv[1] + "/*.safetensors"):
    with safe_open(f, framework="pt") as h:
        list(h.keys())
