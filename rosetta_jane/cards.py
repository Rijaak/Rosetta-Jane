"""Which graphics cards this install can really use. Each card is tried, not guessed at.

PyTorch comes in builds, and each build works with some generations of card. A card the driver can see
is not necessarily one this build can calculate on: a GTX 1080 Ti shows up under the newest NVIDIA
build and then fails at the first calculation. So this runs a small calculation on every card.
"""
from __future__ import annotations

from typing import List

TURING = (7, 5)        # RTX 20-series, Quadro RTX, GTX 16-series. Older than this needs the older-cards build.


def version() -> str:
    """The PyTorch in use, e.g. '2.15.0+cu132' or '2.15.0+cpu'."""
    import torch
    return torch.__version__


def check() -> List[dict]:
    """One entry per card PyTorch can see: its name, its memory, and whether a small calculation runs on it."""
    import warnings

    import torch
    cards = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # PyTorch's own notes about old cards; ours are in plain words
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(count):
            card = {"index": i, "name": "a card that could not be read", "memory_gb": 0.0, "capability": None,
                    "works": False, "why": ""}
            try:
                card["name"] = str(torch.cuda.get_device_name(i))
                card["capability"] = tuple(int(v) for v in torch.cuda.get_device_capability(i))
                card["memory_gb"] = torch.cuda.get_device_properties(i).total_memory / 2 ** 30
                x = torch.ones(64, 64, device=f"cuda:{i}")
                card["works"] = float((x @ x).sum().item()) == 64.0 ** 3
                if not card["works"]:
                    card["why"] = "a test calculation gave the wrong answer"
            except Exception as e:                   # the usual one: "no kernel image is available"
                card["why"] = (str(e).strip().splitlines() or [type(e).__name__])[0][:160]
            cards.append(card)
    return cards


def describe(cards: List[dict], torch_version: str) -> List[str]:
    """What check() found, and what to do about it, in plain words."""
    processor_only = torch_version.endswith("+cpu")
    if not cards:
        lines = [f"No graphics card can be used by this install (PyTorch {torch_version}), "
                 "so models run on the processor."]
        if processor_only:
            lines.append("This is the processor-only build. To use a graphics card, install again and choose its maker.")
        else:
            lines += ["If the machine has an NVIDIA or AMD card, check that its driver is installed and working.",
                      "GTX 10-series and older NVIDIA cards need NVIDIA's 580 driver series and the older-cards build."]
        return lines
    lines = [f"Graphics cards on this machine (PyTorch {torch_version}):"]
    for c in cards:
        size = f", {c['memory_gb']:.0f} GB" if c["memory_gb"] else ""
        lines.append(f"  card {c['index']}: {c['name']}{size}: "
                     f"{'ready' if c['works'] else 'cannot be used by this install'}")
    ready = [c for c in cards if c["works"]]
    failed = [c for c in cards if not c["works"]]
    for c in failed:
        if c["capability"] is not None and c["capability"] < TURING:
            lines += [f"Card {c['index']} is from an older generation than this build of PyTorch supports.",
                      "  To use it, install the older-cards build: run the installer again and choose "
                      "'NVIDIA, older cards'",
                      "  (sh install.sh --nvidia-older). That build also runs RTX 20-, 30- and 40-series cards",
                      "  beside it, so one install covers them all. It cannot run RTX 50-series cards."]
        else:
            lines += [f"Card {c['index']} could not run a test calculation: {c['why'] or 'no reason given'}",
                      "  If it is an RTX 50-series card and this is the older-cards build, the two cannot share",
                      "  one install. Otherwise check the card's driver."]
    if ready:
        lines.append("To run a model on the ready cards, add  --device-map auto  to serve, gateway, ask or train.")
        lines.append("The model is spread across them. Without it the model runs on the processor.")
        if len(ready) > 1:
            lines.append("Cards of different generations work side by side. To cap each one:  "
                         "--max-memory 0=14GiB,1=6GiB")
        if failed:
            keep = ",".join(str(c["index"]) for c in ready)
            lines += ["Until every card is ready, leave the others out, or loading a model will fail:",
                      f"  CUDA_VISIBLE_DEVICES={keep} rosetta_jane serve ... --device-map auto"]
    return lines


def report() -> List[str]:
    return describe(check(), version())
