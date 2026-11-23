"""Run the phone app's Python the way the phone does: where PyTorch and its relatives cannot be imported.

Reads one call per line on standard input, {"fn": name, "args": [...]}, and answers each with a line
"RESULT <json>". This stands in for the app's buttons in tests/test_app.py.
"""
import importlib.abc
import json
import os
import sys

BLOCKED = ("torch", "transformers", "safetensors", "peft", "tokenizers", "bitsandbytes", "cryptography")


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} does not exist on this device")
        return None


sys.meta_path.insert(0, Blocker())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "android", "app", "src", "main", "python"))
import rj_app  # noqa: E402

for line in sys.stdin:
    call = json.loads(line)
    if call["fn"] == "quit":
        break
    try:
        out = getattr(rj_app, call["fn"])(*call["args"])
    except Exception as e:                           # the test wants to see it, not a dead driver
        out = f"RAISED {type(e).__name__}: {e}"
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    print("RESULT " + json.dumps({"out": out, "forbidden": loaded}), flush=True)
rj_app.stop()
if rj_app._own["node"] is not None:
    rj_app._own["node"].stop()
