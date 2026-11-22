"""Reading and writing safetensors files with nothing but NumPy.

The format is small: 8 bytes giving the length of a JSON header, the header
(tensor name -> dtype, shape, byte range), then the raw numbers. Model files are
read through a memory map, so only the tensors actually asked for are loaded.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Tuple

import numpy as np

MAX_HEADER = 100 << 20
_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
           "U8": "u1", "BOOL": "?"}
_WIDTH = {**{k: np.dtype(v).itemsize for k, v in _DTYPES.items()}, "BF16": 2}
_NAMES = {"float32": "F32", "float16": "F16", "float64": "F64", "int64": "I64", "int32": "I32"}


class FormatError(ValueError):
    pass


def _header(raw: bytes, total: int) -> Tuple[Dict[str, dict], int]:
    if len(raw) < 8:
        raise FormatError("not a safetensors file")
    (n,) = struct.unpack("<Q", raw[:8])
    if n > MAX_HEADER or 8 + n > total:
        raise FormatError("safetensors header is too large or cut off")
    return n, 8 + n


def _check(name: str, info: dict, data_bytes: int) -> Tuple[str, List[int], int, int]:
    if not isinstance(info, dict):
        raise FormatError(f"bad entry for {name}")
    dtype, shape, span = info.get("dtype"), info.get("shape"), info.get("data_offsets")
    if dtype not in _WIDTH or not isinstance(shape, list) or not isinstance(span, list) or len(span) != 2:
        raise FormatError(f"bad entry for {name}")
    if not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape + span):
        raise FormatError(f"bad entry for {name}")
    count = 1
    for d in shape:
        count *= d
    a, b = span
    if not 0 <= a <= b <= data_bytes or b - a != count * _WIDTH[dtype]:
        raise FormatError(f"{name}: data range does not match its shape")
    return dtype, shape, a, b


def _decode(buf, dtype: str, shape: List[int]) -> np.ndarray:
    if dtype == "BF16":                          # the top half of a float32
        return (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).view("<f4").reshape(shape)
    return np.frombuffer(buf, dtype=_DTYPES[dtype]).reshape(shape)


class File:
    """One safetensors file, opened without loading its weights."""

    def __init__(self, path: str):
        self.path = path
        total = os.path.getsize(path)
        with open(path, "rb") as f:
            n, self._start = _header(f.read(8), total)
            try:
                meta = json.loads(f.read(n))
            except ValueError as e:
                raise FormatError("safetensors header is not JSON") from e
        if not isinstance(meta, dict):
            raise FormatError("safetensors header is not an object")
        meta.pop("__metadata__", None)
        self.entries = {name: _check(name, info, total - self._start) for name, info in meta.items()}
        self._map = None

    def keys(self) -> List[str]:
        return list(self.entries)

    def shape(self, name: str) -> List[int]:
        return self.entries[name][1]

    def dtype(self, name: str) -> str:
        return self.entries[name][0]

    def get(self, name: str, as_float32: bool = True) -> np.ndarray:
        dtype, shape, a, b = self.entries[name]
        if self._map is None:
            self._map = np.memmap(self.path, dtype="u1", mode="r", offset=self._start)
        out = _decode(self._map[a:b], dtype, shape)
        if as_float32 and out.dtype.kind == "f":
            return np.ascontiguousarray(out, dtype=np.float32)
        return np.array(out)

    def rows(self, name: str, indexes) -> np.ndarray:
        """Chosen rows of a 2-D table as float32, without loading the rest (a token table can be huge)."""
        dtype, shape, a, b = self.entries[name]
        if len(shape) != 2:
            raise FormatError(f"{name} is not a table")
        if self._map is None:
            self._map = np.memmap(self.path, dtype="u1", mode="r", offset=self._start)
        width = (b - a) // shape[0]
        out = np.empty((len(indexes), shape[1]), dtype=np.float32)
        for i, row in enumerate(indexes):
            if not 0 <= row < shape[0]:
                raise FormatError(f"token {row} is outside the table")
            out[i] = _decode(self._map[a + row * width:a + (row + 1) * width], dtype, [shape[1]])
        return out

    def close(self) -> None:
        self._map = None


def weight_files(model_dir: str) -> List[str]:
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        raise FormatError(f"no .safetensors weight files in {model_dir}")
    return [os.path.join(model_dir, f) for f in files]


def loads(blob: bytes) -> Dict[str, np.ndarray]:
    """Decode a safetensors blob received from a peer. Every size is checked before use."""
    n, start = _header(blob, len(blob))
    try:
        meta = json.loads(blob[8:start])
    except ValueError as e:
        raise FormatError("safetensors header is not JSON") from e
    if not isinstance(meta, dict):
        raise FormatError("safetensors header is not an object")
    meta.pop("__metadata__", None)
    out = {}
    for name, info in meta.items():
        dtype, shape, a, b = _check(name, info, len(blob) - start)
        arr = _decode(memoryview(blob)[start + a:start + b], dtype, shape)
        out[name] = np.array(arr, dtype=np.float32 if arr.dtype.kind == "f" else arr.dtype)
    return out


def dumps(tensors: Dict[str, np.ndarray]) -> bytes:
    """Encode arrays the way the safetensors library does: sorted by dtype then name, header padded to 8."""
    order = sorted(tensors, key=lambda k: (-tensors[k].dtype.alignment, k))
    meta, chunks, at = {}, [], 0
    for name in order:
        arr = tensors[name]
        code = _NAMES.get(arr.dtype.name)
        if code is None:
            raise FormatError(f"cannot store dtype {arr.dtype}")
        raw = np.ascontiguousarray(arr, dtype=_DTYPES[code]).tobytes()
        meta[name] = {"dtype": code, "shape": list(arr.shape), "data_offsets": [at, at + len(raw)]}
        chunks.append(raw)
        at += len(raw)
    head = json.dumps(meta, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    return struct.pack("<Q", len(head)) + head + b"".join(chunks)
