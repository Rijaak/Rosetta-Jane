"""Run the phone's p2p service the way a phone does: with PyTorch and its relatives impossible to import.

Usage: python no_torch.py serve --model ... (the arguments of `python -m rosetta_jane.mobile`)
"""
import importlib.abc
import sys

BLOCKED = ("torch", "transformers", "safetensors", "peft", "tokenizers", "bitsandbytes")


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} does not exist on this device")
        return None


sys.meta_path.insert(0, Blocker())
from rosetta_jane.mobile.__main__ import main  # noqa: E402

try:
    code = main(sys.argv[1:])
finally:
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    if loaded:
        print(f"FORBIDDEN IMPORTS: {loaded}", flush=True)
        code = 97
sys.exit(code)
