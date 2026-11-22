"""Rosetta-Jane for devices that cannot run PyTorch, such as Android phones.

Everything in this package uses only Python and NumPy. It speaks the same
protocol as the desktop program, so a phone is an ordinary peer: it can join a
group, run the p2p service for a stretch of layers, and help with training.

Nothing here may import torch, transformers, safetensors or peft, directly or
through another module. A test runs this package with those imports blocked.
"""
