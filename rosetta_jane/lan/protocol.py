"""Wire protocol: framed, authenticated messages carrying JSON plus one optional tensor.

Design rules:
  * No pickle ever crosses the network. Headers are JSON, tensors are raw bytes
    with an explicit dtype and shape, adapters are safetensors.
  * Both ends prove they know a shared secret, and every frame carries an
    HMAC-SHA256 tag over a per-connection sequence number, the header and the
    payload. That stops strangers on the network from joining or tampering.
  * Connections are encrypted by default with TLS keyed from the same secret
    (TLS-PSK, needs Python 3.13 or newer). Anyone without the secret can
    neither read the traffic nor join.
  * The secret can be a random file or a group code that people type in, the
    way a co-op game lobby works. Everyone who enters the same code can connect
    to each other. A short or guessable code is a weak lock.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import ssl
import struct
from typing import Optional, Tuple

import numpy as np

try:                                              # a phone has no PyTorch; the wire format does not need it
    import torch
except ImportError:                               # pragma: no cover - exercised in a torch-free subprocess
    torch = None

MAGIC = b"LNLR"
WIRE_VERSION = 1
MAX_HEADER = 1 << 20          # 1 MiB of JSON is far more than any message needs
MAX_PAYLOAD = 1 << 32         # 4 GiB hard ceiling on one tensor
_FRAME = struct.Struct("!4sBIQ")  # magic, version, header length, payload length
_TAG_LEN = 32
_NONCE_LEN = 32

# dtype name -> (torch dtype, same-width integer dtype used to move raw bits)
_DTYPES = {} if torch is None else {
    "float32": (torch.float32, torch.int32),
    "float16": (torch.float16, torch.int16),
    "bfloat16": (torch.bfloat16, torch.int16),
    "int64": (torch.int64, torch.int64),
    "int32": (torch.int32, torch.int32),
}
_NAME_OF = {v[0]: k for k, v in _DTYPES.items()}
# The same wire names for plain arrays. Little-endian, as PyTorch writes them on every
# platform this runs on. bfloat16 has no array type; it arrives widened to float32.
_NP = {"float32": "<f4", "float16": "<f2", "int64": "<i8", "int32": "<i4"}
_WIDTH = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4}


def bfloat16_to_float32(raw: bytes) -> np.ndarray:
    """bfloat16 is the top half of a float32: put the bits back and fill the rest with zeros."""
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")


class ProtocolError(Exception):
    """The peer sent something malformed, unauthenticated or out of order."""


class AuthError(ProtocolError):
    """The other side has a different secret or group code. Retrying will not help."""


class MismatchError(ProtocolError):
    """The other side is set up for a different model, plan or machine name. Retrying will not help."""


class RemoteError(Exception):
    """The peer reported that it failed to carry out a request."""


def tensor_to_bytes(t) -> Tuple[dict, bytes]:
    if isinstance(t, np.ndarray):
        name = {"float32": "float32", "float16": "float16", "int64": "int64", "int32": "int32"}.get(t.dtype.name)
        if name is None:
            raise ProtocolError(f"unsupported array dtype {t.dtype}")
        return {"dtype": name, "shape": list(t.shape)}, np.ascontiguousarray(t, dtype=_NP[name]).tobytes()
    t = t.detach().to("cpu").contiguous()
    if t.dtype not in _NAME_OF:
        raise ProtocolError(f"unsupported tensor dtype {t.dtype}")
    name = _NAME_OF[t.dtype]
    bits = t.view(_DTYPES[name][1])
    return {"dtype": name, "shape": list(t.shape)}, bits.numpy().tobytes()


def tensor_from_bytes(meta: dict, payload: bytes):
    """A tensor where PyTorch is installed, a plain array where it is not."""
    name = meta.get("dtype")
    shape = meta.get("shape")
    if name not in _WIDTH or not isinstance(shape, list) or len(shape) > 8:
        raise ProtocolError("bad tensor metadata")
    if not all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d < (1 << 40) for d in shape):
        raise ProtocolError("bad tensor shape")
    count = 1
    for d in shape:
        count *= d
    if count * _WIDTH[name] != len(payload):
        raise ProtocolError("tensor payload size does not match its shape")
    if torch is None:
        flat = bfloat16_to_float32(payload) if name == "bfloat16" else np.frombuffer(payload, dtype=_NP[name])
        return flat.reshape(shape).copy()
    real, bits = _DTYPES[name]
    if count == 0:
        return torch.empty(shape, dtype=real)
    raw = torch.frombuffer(bytearray(payload), dtype=bits)
    return raw.view(real).reshape(shape).clone()


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_secret(path: str) -> bytes:
    with open(path, "rb") as f:
        secret = f.read().strip()
    if len(secret) < 16:
        raise ValueError("secret file must hold at least 16 characters")
    return secret


def new_secret() -> str:
    return os.urandom(32).hex()


MIN_GROUP_CODE = 8
PUBLIC_CODE = "public"   # the open lobby: anyone running the software can connect


def secret_from_group_code(code: str) -> bytes:
    """Turn a typed group code into key material. Same code in, same key out, on every machine.

    Spaces and letter case are ignored so a code read out loud still matches. The
    slow scrypt step makes guessing codes expensive, but it cannot rescue a code
    like "password": use a long one for anything that matters.
    """
    normal = "".join(code.split()).casefold()
    if normal != PUBLIC_CODE and len(normal) < MIN_GROUP_CODE:
        raise ValueError(f"group code must be at least {MIN_GROUP_CODE} characters (or exactly 'public')")
    return hashlib.scrypt(normal.encode(), salt=b"rosetta-jane-group-code-v1", n=1 << 15, r=8, p=1,
                          maxmem=1 << 27, dklen=32)   # about 32 MB of work per guess


def encryption_available() -> bool:
    return hasattr(ssl.SSLContext, "set_psk_server_callback")


_TLS_CIPHERS = "ECDHE-PSK-CHACHA20-POLY1305:DHE-PSK-AES256-GCM-SHA384"


def _tls_key(secret: bytes) -> bytes:
    return hmac.new(secret, b"rosetta-jane-tls-psk", hashlib.sha256).digest()


def _tls_wrap(sock: socket.socket, secret: bytes, server: bool) -> socket.socket:
    if not encryption_available():
        raise ProtocolError(
            "encryption needs Python 3.13 or newer. Upgrade Python, or run every machine with "
            "--no-encrypt inside a VPN such as Tailscale or WireGuard.")
    key = _tls_key(secret)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2   # the pre-shared-key suites used here are TLS 1.2 suites
    ctx.set_ciphers(_TLS_CIPHERS)
    if server:
        ctx.set_psk_server_callback(lambda identity: key)
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE             # the key, not a certificate, proves who the peer is
        ctx.set_psk_client_callback(lambda hint: ("rosetta-jane", key))
    try:
        return ctx.wrap_socket(sock, server_side=server)
    except ssl.SSLError as e:
        raise AuthError(
            "encrypted handshake failed: the other side has a different secret or group code, "
            "or is running with --no-encrypt") from e


class Channel:
    """One authenticated connection. Not thread-safe; use one per peer."""

    def __init__(self, sock: socket.socket, session_key: bytes):
        self.sock = sock
        self._key = session_key
        self._send_seq = 0
        self._recv_seq = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.tap = None                           # optional recorder: tap(direction, header, tensor, payload)

    # -- handshake -------------------------------------------------------
    @staticmethod
    def _session_key(secret: bytes, server_nonce: bytes, client_nonce: bytes) -> bytes:
        return hmac.new(secret, b"rosetta-jane-session" + server_nonce + client_nonce, hashlib.sha256).digest()

    @classmethod
    def accept(cls, sock: socket.socket, secret: bytes, encrypt: bool = True) -> "Channel":
        """Server side. Raises ProtocolError if the client does not know the secret."""
        if encrypt:
            sock = _tls_wrap(sock, secret, server=True)
        server_nonce = os.urandom(_NONCE_LEN)
        sock.sendall(MAGIC + bytes([WIRE_VERSION]) + server_nonce)
        hello = _recv_exact(sock, len(MAGIC) + 1 + _NONCE_LEN + _TAG_LEN)
        if hello[:4] != MAGIC or hello[4] != WIRE_VERSION:
            raise ProtocolError("not a Rosetta-Jane client, or a different protocol version")
        client_nonce = hello[5:5 + _NONCE_LEN]
        proof = hello[5 + _NONCE_LEN:]
        key = cls._session_key(secret, server_nonce, client_nonce)
        if not hmac.compare_digest(proof, hmac.new(key, b"client", hashlib.sha256).digest()):
            raise AuthError("client failed authentication (wrong secret)")
        sock.sendall(hmac.new(key, b"server", hashlib.sha256).digest())
        return cls(sock, key)

    @classmethod
    def connect(cls, host: str, port: int, secret: bytes, timeout: float = 30.0,
                encrypt: bool = True) -> "Channel":
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            if encrypt:
                sock = _tls_wrap(sock, secret, server=False)
            greeting = _recv_exact(sock, len(MAGIC) + 1 + _NONCE_LEN)
            if greeting[:4] != MAGIC or greeting[4] != WIRE_VERSION:
                raise ProtocolError("not a Rosetta-Jane node, or a different protocol version")
            server_nonce = greeting[5:]
            client_nonce = os.urandom(_NONCE_LEN)
            key = cls._session_key(secret, server_nonce, client_nonce)
            sock.sendall(MAGIC + bytes([WIRE_VERSION]) + client_nonce
                         + hmac.new(key, b"client", hashlib.sha256).digest())
            try:
                proof = _recv_exact(sock, _TAG_LEN)
            except ConnectionError:
                raise AuthError("worker rejected the connection (different secret or group code)") from None
            if not hmac.compare_digest(proof, hmac.new(key, b"server", hashlib.sha256).digest()):
                raise AuthError("worker failed authentication (different secret or group code)")
        except BaseException:
            sock.close()
            raise
        return cls(sock, key)

    # -- frames ----------------------------------------------------------
    def _tag(self, seq: int, header: bytes, payload: bytes) -> bytes:
        mac = hmac.new(self._key, struct.pack("!Q", seq), hashlib.sha256)
        mac.update(struct.pack("!I", len(header)))
        mac.update(header)
        mac.update(payload)
        return mac.digest()

    def send(self, header: dict, tensor=None, blob: Optional[bytes] = None) -> None:
        header = dict(header)
        payload = b""
        if tensor is not None:
            header["tensor"], payload = tensor_to_bytes(tensor)
        elif blob is not None:
            header["blob"] = len(blob)
            payload = blob
        raw = json.dumps(header, separators=(",", ":")).encode()
        if len(raw) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
            raise ProtocolError("message too large")
        tag = self._tag(self._send_seq, raw, payload)
        self._send_seq += 1
        self.sock.sendall(_FRAME.pack(MAGIC, WIRE_VERSION, len(raw), len(payload)) + raw)
        if payload:
            self.sock.sendall(payload)
        self.sock.sendall(tag)
        self.bytes_sent += _FRAME.size + len(raw) + len(payload) + _TAG_LEN
        if self.tap is not None:
            self.tap("sent", header, tensor, payload)

    def recv(self) -> Tuple[dict, Optional[object], Optional[bytes]]:
        magic, version, hlen, plen = _FRAME.unpack(_recv_exact(self.sock, _FRAME.size))
        if magic != MAGIC or version != WIRE_VERSION:
            raise ProtocolError("bad frame")
        if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
            raise ProtocolError("frame too large")
        raw = _recv_exact(self.sock, hlen)
        payload = _recv_exact(self.sock, plen) if plen else b""
        tag = _recv_exact(self.sock, _TAG_LEN)
        if not hmac.compare_digest(tag, self._tag(self._recv_seq, raw, payload)):
            raise ProtocolError("frame failed authentication")
        self._recv_seq += 1
        self.bytes_received += _FRAME.size + hlen + plen + _TAG_LEN
        try:
            header = json.loads(raw)
        except ValueError as e:
            raise ProtocolError("header is not JSON") from e
        if not isinstance(header, dict) or not isinstance(header.get("op"), str):
            raise ProtocolError("header has no op")
        tensor = blob = None
        if "tensor" in header:
            tensor = tensor_from_bytes(header["tensor"], payload)
        elif "blob" in header:
            blob = payload
        elif payload:
            raise ProtocolError("unexpected payload")
        if self.tap is not None:
            self.tap("received", header, tensor, payload)
        return header, tensor, blob

    def call(self, header: dict, tensor=None,
             blob: Optional[bytes] = None) -> Tuple[dict, Optional[object], Optional[bytes]]:
        """Send a request and wait for its reply. Raises RemoteError on an error reply."""
        self.send(header, tensor, blob)
        reply, rtensor, rblob = self.recv()
        if reply["op"] == "error":
            raise RemoteError(str(reply.get("message", "unknown remote error")))
        if reply["op"] != header["op"] + "_ok":
            raise ProtocolError(f"expected reply to {header['op']}, got {reply['op']}")
        return reply, rtensor, rblob

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
