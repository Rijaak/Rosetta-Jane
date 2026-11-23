"""LoRA as plain tensors that can be handed to a set of layers for one calculation.

In ordinary LoRA training the small trained matrices live inside the model. Here
they live with the person training, as a dictionary of tensors, and are sent to
whichever device is running a given layer for the length of one calculation.
That makes each device's work a pure function of what it was sent, which is what
lets two devices be compared with each other, or checked against a recomputation.

Key naming: "<layer number>.<module path>.A" and ".B", for example
"12.self_attn.q_proj.A". Layer numbers are positions in the whole model.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_KEY_RE = re.compile(r"^(\d+)\.([A-Za-z0-9_.]+)\.(A|B)$")
MAX_RANK = 256


def check_spec(rank, alpha, targets) -> Tuple[int, float, List[str]]:
    if not (isinstance(rank, int) and 1 <= rank <= MAX_RANK):
        raise ValueError(f"LoRA rank must be a whole number from 1 to {MAX_RANK}")
    if not (isinstance(alpha, (int, float)) and 0 < alpha <= 4096):
        raise ValueError("bad LoRA alpha")
    if not (isinstance(targets, (list, tuple)) and 1 <= len(targets) <= 16
            and all(isinstance(t, str) and t.replace("_", "").isalnum() and len(t) <= 40 for t in targets)):
        raise ValueError("bad LoRA target names")
    return rank, float(alpha), list(targets)


def find_targets(layer: nn.Module, targets: Sequence[str]) -> Dict[str, nn.Module]:
    """The modules inside one layer that get an adapter, by path within the layer."""
    found = {}
    for name, module in layer.named_modules():
        if name and name.split(".")[-1] in targets and hasattr(module, "in_features") \
                and hasattr(module, "out_features"):
            found[name] = module
    return found


def shapes_of_layers(layers: Sequence[nn.Module], first: int, targets: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """{"<layer>.<path>": (in, out)} for layers loaded in memory, numbered from `first`."""
    out = {}
    for offset, layer in enumerate(layers):
        for path, module in find_targets(layer, targets).items():
            out[f"{first + offset}.{path}"] = (int(module.in_features), int(module.out_features))
    return out


def shapes_from_files(model_dir: str, targets: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """The same, for every layer of a model, read from the weight files without loading them."""
    from .lan.slicing import _scan, detect_layout
    scan = _scan(model_dir)
    prefix, _ = detect_layout(scan.keys())
    out = {}
    for key, (_, shape, _) in scan.items():
        if not key.startswith(prefix) or not key.endswith(".weight") or len(shape) != 2:
            continue
        index, _, path = key[len(prefix):-len(".weight")].partition(".")
        if index.isdigit() and path.split(".")[-1] in targets:
            out[f"{int(index)}.{path}"] = (int(shape[1]), int(shape[0]))
    if not out:
        raise ValueError(f"no modules named {list(targets)} found in this model")
    return out


def init_adapters(shapes: Dict[str, Tuple[int, int]], rank: int, seed: int) -> Dict[str, torch.Tensor]:
    """Fresh adapters: A random, B zero, so training starts from the unchanged model."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name in sorted(shapes):
        fan_in, fan_out = shapes[name]
        bound = math.sqrt(6.0 / ((1 + 5.0) * fan_in))        # the usual LoRA start (kaiming uniform, a=sqrt(5))
        out[f"{name}.A"] = (torch.rand(rank, fan_in, generator=gen) * 2 - 1) * bound
        out[f"{name}.B"] = torch.zeros(fan_out, rank)
    return out


def keys_for(shapes: Dict[str, Tuple[int, int]], start: int, end: int) -> List[str]:
    return sorted(f"{name}.{part}" for name in shapes if start <= int(name.split(".", 1)[0]) < end
                  for part in ("A", "B"))


def check_adapters(adapters: Dict[str, torch.Tensor], shapes: Dict[str, Tuple[int, int]], rank: int) -> None:
    """Refuse adapters that do not exactly fit the layers they are meant for."""
    expected = set(f"{name}.{part}" for name in shapes for part in ("A", "B"))
    if set(adapters) != expected:
        raise ValueError("adapters do not match these layers")
    for key, value in adapters.items():
        m = _KEY_RE.match(key)
        fan_in, fan_out = shapes[key[:-2]]
        want = (rank, fan_in) if m.group(3) == "A" else (fan_out, rank)
        if tuple(value.shape) != want or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"adapter {key} has the wrong shape or contents")


class Applied:
    """Apply adapters to a run of layers for the length of a `with` block."""

    def __init__(self, layers: Sequence[nn.Module], first: int, targets: Sequence[str],
                 adapters: Dict[str, torch.Tensor], scale: float, place: bool = False):
        """With place=True each adapter is moved to the device its module is on, once, here.
        That is for answering, where a model may be spread over several graphics cards."""
        self._pairs = []
        for offset, layer in enumerate(layers):
            for path, module in find_targets(layer, targets).items():
                name = f"{first + offset}.{path}"
                a, b = adapters[f"{name}.A"], adapters[f"{name}.B"]
                if place:
                    where = next((p.device for p in module.parameters()), a.device)
                    a, b = a.to(where), b.to(where)
                self._pairs.append((module, a, b))
        self._scale = scale
        self._handles = []

    def __enter__(self):
        for module, a, b in self._pairs:
            def hook(mod, inputs, output, a=a, b=b):
                x = inputs[0]
                delta = F.linear(F.linear(x.to(a.dtype), a), b) * self._scale
                return output + delta.to(output.dtype)
            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def flat(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """All values as one vector in key order, for comparing two sets."""
    if not tensors:
        return torch.zeros(0)
    return torch.cat([tensors[k].detach().float().reshape(-1).cpu() for k in sorted(tensors)])


def to_peft(adapters: Dict[str, torch.Tensor], layer_prefix: str) -> Dict[str, torch.Tensor]:
    """Rename to the standard adapter file layout so any LoRA tool can load the result."""
    out = {}
    for key, value in adapters.items():
        m = _KEY_RE.match(key)
        out[f"base_model.model.{layer_prefix}{m.group(1)}.{m.group(2)}.lora_{m.group(3)}.weight"] = \
            value.detach().cpu().contiguous()
    return out


def from_peft(state: Dict[str, torch.Tensor], layer_prefix: str) -> Dict[str, torch.Tensor]:
    pattern = re.compile(r"^base_model\.model\." + re.escape(layer_prefix) + r"(\d+)\.(.+)\.lora_(A|B)(?:\.[^.]+)?\.weight$")
    out = {}
    for key, value in state.items():
        m = pattern.match(key)
        if m:
            out[f"{m.group(1)}.{m.group(2)}.{m.group(3)}"] = value.detach().clone()
    return out
