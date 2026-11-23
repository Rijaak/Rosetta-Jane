"""Signatures, so a device can show that something really came from it.

Each device keeps a secret key (see identity.py). With it the device can sign a short
message, and anyone holding the matching public key can check the signature. That is what
lets a name travel with a device, and lets a leader hand out a receipt for teaching points
that other leaders can check without asking anybody.

The scheme is Ed25519 (RFC 8032). On a desktop the `cryptography` package does the work.
A phone does not have that package, so this file also carries the algorithm in plain
Python, taken step for step from the reference code in the RFC. Both give the same bytes
for the same input, and the tests check each against the other and against the RFC's own
examples. The plain version is slower (a few thousandths of a second per signature on a
desktop, more on a phone) and takes no care to run in constant time: fine for signing a
name or a points receipt a few times a session, not something to build a bank on.

Plain Python, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import hashlib

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _add(a, b):
    A, B = (a[1] - a[0]) * (b[1] - b[0]) % _P, (a[1] + a[0]) * (b[1] + b[0]) % _P
    C, D = 2 * a[3] * b[3] * _D % _P, 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return E * F % _P, G * H % _P, F * G % _P, E * H % _P


def _mul(s: int, point):
    out = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            out = _add(out, point)
        point = _add(point, point)
        s >>= 1
    return out


def _same(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    return _P - x if (x & 1) != sign else x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(point) -> bytes:
    z = _inv(point[2])
    x, y = point[0] * z % _P, point[1] * z % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decompress(raw: bytes):
    y = int.from_bytes(raw, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _expand(seed: bytes):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    return (a & ((1 << 254) - 8)) | (1 << 254), h[32:]


def _hash_q(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % _Q


def pure_public(seed: bytes) -> bytes:
    return _compress(_mul(_expand(seed)[0], _G))


def pure_sign(seed: bytes, message: bytes) -> bytes:
    a, prefix = _expand(seed)
    public = _compress(_mul(a, _G))
    r = _hash_q(prefix + message)
    first = _compress(_mul(r, _G))
    s = (r + _hash_q(first + public + message) * a) % _Q
    return first + s.to_bytes(32, "little")


def pure_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A, R = _decompress(public), _decompress(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if A is None or R is None or s >= _Q:
        return False
    h = _hash_q(signature[:32] + public + message)
    return _same(_mul(s, _G), _add(R, _mul(h, A)))


try:                                     # a desktop: let the vetted library do it
    from cryptography.exceptions import InvalidSignature as _Invalid
    from cryptography.hazmat.primitives import serialization as _ser
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _lib
    FAST = True
except Exception:                        # a phone: the plain version above
    FAST = False


def public_key(seed: bytes) -> bytes:
    """The public key that goes with a 32-byte secret."""
    if len(seed) != 32:
        raise ValueError("a secret key is 32 bytes")
    if FAST:
        return _lib.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            _ser.Encoding.Raw, _ser.PublicFormat.Raw)
    return pure_public(seed)


def sign(seed: bytes, message: bytes) -> bytes:
    """A 64-byte signature of the message, made with the secret."""
    if len(seed) != 32:
        raise ValueError("a secret key is 32 bytes")
    if FAST:
        return _lib.Ed25519PrivateKey.from_private_bytes(seed).sign(message)
    return pure_sign(seed, message)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Was this signature made for this message by the holder of the secret behind this public key?"""
    if not (isinstance(public, bytes) and isinstance(signature, bytes)) or len(public) != 32 or len(signature) != 64:
        return False
    if FAST:
        try:
            _lib.Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
            return True
        except (_Invalid, ValueError):
            return False
    return pure_verify(public, message, signature)
