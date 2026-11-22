"""Planning which machine holds which layers, and cutting a model folder into slices.

Every machine downloads the same base model. The host writes a plan that gives
each machine a run of consecutive decoder layers. Each machine then cuts its own
slice out of its local copy, so no model weights travel over the network.

Layout of a run with two guests (two break points):

    host:   embeddings + layers [0, a)                    ... final norm + output head
    guest1:              layers [a, b)
    guest2:              layers [b, N)

The host keeps both ends of the model, so the training text and the answers
never leave it. Guests only receive and return hidden-state tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from safetensors import safe_open

PLAN_VERSION = 1
_LAYER_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.)(?P<index>\d+)\.(?P<rest>.+)$")
_SHARD_BYTES = 2 << 30


# Model families whose split runs were checked against single-machine training on
# tiny random models (see tests). A family outside this list may treat layers
# differently by position in a way a cut would silently break. Gemma 2 is left out
# because its split run drifted slightly from the reference.
VERIFIED_MODEL_TYPES = frozenset({"llama", "qwen2", "qwen3", "mistral", "mixtral", "phi3", "gemma",
                                  "gemma3_text", "starcoder2"})


class SliceError(Exception):
    pass


def check_model_type(model_dir: str, allow_untested: bool = False) -> str:
    with open(os.path.join(model_dir, "config.json")) as f:
        model_type = str(json.load(f).get("model_type", "unknown"))
    if model_type not in VERIFIED_MODEL_TYPES and not allow_untested:
        raise SliceError(
            f"model family {model_type!r} has not been verified to split correctly. Verified: "
            f"{', '.join(sorted(VERIFIED_MODEL_TYPES))}. Pass --allow-untested-model to try anyway, "
            "and compare a short run against single-machine training before trusting it.")
    return model_type


def _weight_files(model_dir: str) -> List[str]:
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        raise SliceError(
            f"no .safetensors weight files in {model_dir}. Only safetensors models are "
            "supported; pickle-based .bin weights are refused on purpose.")
    return [os.path.join(model_dir, f) for f in files]


def _scan(model_dir: str) -> Dict[str, Tuple[str, List[int], str]]:
    """Map every tensor name to (file, shape, dtype) without loading any weights."""
    found: Dict[str, Tuple[str, List[int], str]] = {}
    for path in _weight_files(model_dir):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                found[key] = (path, list(sl.get_shape()), str(sl.get_dtype()))
    return found


def detect_layout(keys) -> Tuple[str, int]:
    """Return (layer key prefix, number of layers), e.g. ("model.layers.", 32)."""
    per_prefix: Dict[str, set] = {}
    for key in keys:
        m = _LAYER_RE.match(key)
        if m:
            per_prefix.setdefault(m.group("prefix"), set()).add(int(m.group("index")))
    if not per_prefix:
        raise SliceError("could not find a stack of decoder layers (no '...layers.N.' weights)")
    prefix, indices = max(per_prefix.items(), key=lambda kv: len(kv[1]))
    count = max(indices) + 1
    if indices != set(range(count)):
        raise SliceError("decoder layer numbering has gaps; unsupported model layout")
    return prefix, count


def model_fingerprint(model_dir: str) -> str:
    """A short ID of the base model: same weights layout and config give the same ID.

    It hashes tensor names, shapes and dtypes plus the config, not the weight
    values, so it is fast. It catches a different model or size, not a fine-tune
    of the same shape.
    """
    h = hashlib.sha256()
    for key, (_, shape, dtype) in sorted(_scan(model_dir).items()):
        h.update(f"{key}|{shape}|{dtype}\n".encode())
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        cfg = json.load(f)
    for volatile in ("_name_or_path", "transformers_version"):
        cfg.pop(volatile, None)
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()[:16]


def split_layers(num_layers: int, weights: List[float]) -> List[Tuple[int, int]]:
    """Split num_layers into consecutive ranges sized in proportion to weights."""
    n = len(weights)
    if n < 1 or any(w <= 0 for w in weights):
        raise SliceError("every machine needs a positive memory figure")
    if num_layers < n:
        raise SliceError(f"{n} machines but only {num_layers} layers; each machine needs at least one")
    total = sum(weights)
    counts = [max(1, int(round(num_layers * w / total))) for w in weights]
    while sum(counts) > num_layers:   # trim the largest that can spare a layer
        i = max((i for i in range(n) if counts[i] > 1), key=lambda i: counts[i] - num_layers * weights[i] / total)
        counts[i] -= 1
    while sum(counts) < num_layers:   # top up whoever is furthest under its share
        i = min(range(n), key=lambda i: counts[i] - num_layers * weights[i] / total)
        counts[i] += 1
    ranges, start = [], 0
    for c in counts:
        ranges.append((start, start + c))
        start += c
    return ranges


def make_plan(model_dir: str, host_mem_gb: float, guests: List[dict], host_reserve_gb: float = 2.0,
              allow_untested: bool = False) -> dict:
    """guests: [{"name": str, "addr": "ip:port", "mem_gb": float}, ...] in chain order."""
    model_type = check_model_type(model_dir, allow_untested)
    prefix, num_layers = detect_layout(_scan(model_dir).keys())
    names = ["host"] + [g["name"] for g in guests]
    if len(set(names)) != len(names):
        raise SliceError("machine names must be unique, and 'host' is reserved")
    usable_host = host_mem_gb - host_reserve_gb
    if usable_host <= 0:
        raise SliceError("host memory must exceed the reserve kept for embeddings and the output layer")
    ranges = split_layers(num_layers, [usable_host] + [float(g["mem_gb"]) for g in guests])
    stages = [{"name": "host", "layers": list(ranges[0]), "mem_gb": host_mem_gb}]
    for g, r in zip(guests, ranges[1:]):
        stages.append({"name": g["name"], "addr": g["addr"], "layers": list(r), "mem_gb": float(g["mem_gb"])})
    return {
        "plan_version": PLAN_VERSION,
        "fingerprint": model_fingerprint(model_dir),
        "model_type": model_type,
        "allow_untested": bool(allow_untested),
        "num_layers": num_layers,
        "layer_prefix": prefix,
        "stages": stages,
    }


def validate_plan(plan: dict) -> None:
    if plan.get("plan_version") != PLAN_VERSION:
        raise SliceError("plan file is from a different trainer version")
    stages = plan.get("stages") or []
    if not stages or stages[0].get("name") != "host":
        raise SliceError("the first stage in a plan must be 'host'")
    expected = 0
    for s in stages:
        a, b = s["layers"]
        if a != expected or b <= a:
            raise SliceError("plan layer ranges must be consecutive and non-empty")
        expected = b
    if expected != plan["num_layers"]:
        raise SliceError("plan does not cover every layer")


def stage_of(plan: dict, name: str) -> dict:
    for s in plan["stages"]:
        if s["name"] == name:
            return s
    raise SliceError(f"no machine named {name!r} in the plan")


def make_slice(model_dir: str, out_dir: str, plan: dict, name: str, force: bool = False) -> str:
    """Write this machine's part of the model to out_dir as a loadable model folder."""
    validate_plan(plan)
    stage = stage_of(plan, name)
    start, end = stage["layers"]
    is_host = name == "host"
    check_model_type(model_dir, bool(plan.get("allow_untested")))
    fingerprint = model_fingerprint(model_dir)
    if fingerprint != plan["fingerprint"]:
        raise SliceError(
            f"this copy of the model ({fingerprint}) is not the one the plan was made for "
            f"({plan['fingerprint']}). Every machine must use the same base model.")
    marker = {"fingerprint": fingerprint, "name": name, "layers": [start, end], "slice_version": 1}
    marker_path = os.path.join(out_dir, "rosetta_jane_slice.json")
    if not force and os.path.exists(marker_path):
        with open(marker_path) as f:
            if json.load(f) == marker:
                return out_dir
    tmp = out_dir.rstrip("/") + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    prefix = plan["layer_prefix"]
    scan = _scan(model_dir)
    wanted: List[Tuple[str, str, str]] = []  # (source key, new key, file)
    for key, (path, _, _) in scan.items():
        m = _LAYER_RE.match(key)
        if m and m.group("prefix") == prefix:
            i = int(m.group("index"))
            if start <= i < end:
                wanted.append((key, f"{prefix}{i - start}.{m.group('rest')}", path))
        elif is_host:
            wanted.append((key, key, path))  # embeddings, final norm, output head, anything else
    wanted.sort(key=lambda w: (w[2], w[0]))

    if not wanted:
        raise SliceError(f"plan gives {name!r} no weights; check the layer ranges")
    current: Dict[str, object] = {}
    current_bytes = 0
    count = 0                      # shards written so far
    weight_map: Dict[str, int] = {}
    total = 0
    open_path, handle = None, None

    def flush() -> None:
        nonlocal current, current_bytes, count
        if current:
            from safetensors.torch import save_file
            save_file(current, os.path.join(tmp, f"shard-{count:05d}.safetensors"), metadata={"format": "pt"})
            count += 1
            current = {}
            current_bytes = 0

    try:
        for src_key, new_key, path in wanted:
            if path != open_path:
                if handle is not None:
                    handle.__exit__(None, None, None)
                handle = safe_open(path, framework="pt").__enter__()
                open_path = path
            t = handle.get_tensor(src_key)
            size = t.numel() * t.element_size()
            if current_bytes and current_bytes + size > _SHARD_BYTES:
                flush()
            current[new_key] = t
            weight_map[new_key] = count
            current_bytes += size
            total += size
        flush()
    finally:
        if handle is not None:
            handle.__exit__(None, None, None)

    if count == 1:
        os.replace(os.path.join(tmp, "shard-00000.safetensors"), os.path.join(tmp, "model.safetensors"))
    else:
        names = {}
        for i in range(count):
            new = f"model-{i + 1:05d}-of-{count:05d}.safetensors"
            os.replace(os.path.join(tmp, f"shard-{i:05d}.safetensors"), os.path.join(tmp, new))
            names[i] = new
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total},
                       "weight_map": {k: names[i] for k, i in sorted(weight_map.items())}}, f)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    total_layers = plan["num_layers"]
    for key, value in list(cfg.items()):
        # Some families list a setting per layer (which kind of attention each layer
        # uses, for example). Each machine must keep the entries for its own layers.
        if isinstance(value, list) and len(value) == total_layers and key != "architectures":
            cfg[key] = value[start:end]
    cfg["num_hidden_layers"] = end - start
    if not is_host:
        # A guest never looks up tokens, so give it a token table of almost no size
        # instead of a multi-gigabyte one it would never use.
        cfg["vocab_size"] = 8
        cfg["pad_token_id"] = None
        cfg["bos_token_id"] = 0
        cfg["eos_token_id"] = 1
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(tmp, "rosetta_jane_slice.json"), "w") as f:
        json.dump(marker, f)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return out_dir


def load_plan(path: str) -> dict:
    with open(path) as f:
        plan = json.load(f)
    validate_plan(plan)
    return plan


def parse_addr(addr: str) -> Tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host or not port.isdigit():
        raise SliceError(f"address must look like 192.168.1.20:7000, got {addr!r}")
    return host, int(port)
