"""Graphics cards: each one is tried with a real calculation, and the advice names the right build."""
import types

import pytest
import torch

from rosetta_jane import cards
from rosetta_jane.cli import main

RTX5000 = ("Quadro RTX 5000", (7, 5), 16)
RTX4000 = ("Quadro RTX 4000", (7, 5), 8)
GTX1080TI = ("NVIDIA GeForce GTX 1080 Ti", (6, 1), 11)
RTX5090 = ("NVIDIA GeForce RTX 5090", (12, 0), 32)
NO_KERNEL = "CUDA error: no kernel image is available for execution on the device\nCUDA kernel errors might be"


def machine(monkeypatch, fitted, build_supports):
    """Pretend this machine holds these cards, under a PyTorch build that can calculate on some of them."""
    real_ones = torch.ones

    def ones(*shape, device=None, **kw):
        if device is not None and str(device).startswith("cuda:"):
            name, capability, _ = fitted[int(str(device).split(":")[1])]
            if not build_supports(capability):
                raise RuntimeError(NO_KERNEL)
            return real_ones(*shape, **kw)           # the sum is worked out for real, on the processor
        return real_ones(*shape, **kw)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(fitted))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(fitted))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: fitted[i][0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: fitted[i][1])
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(total_memory=fitted[i][2] * 2 ** 30))
    monkeypatch.setattr(torch, "ones", ones)


newest_build = lambda capability: capability >= (7, 5)                     # CUDA 13: Turing and newer
older_build = lambda capability: (5, 0) <= capability <= (9, 0)            # CUDA 12.6: Maxwell to RTX 40-series


def test_an_older_card_beside_newer_ones_is_caught_and_the_right_build_is_named(monkeypatch):
    machine(monkeypatch, [RTX5000, RTX4000, GTX1080TI], newest_build)
    found = cards.check()
    assert [c["works"] for c in found] == [True, True, False]
    assert found[2]["why"].startswith("CUDA error: no kernel image") and "\n" not in found[2]["why"]
    text = "\n".join(cards.describe(found, "2.15.0+cu132"))
    assert "card 0: Quadro RTX 5000, 16 GB: ready" in text
    assert "card 2: NVIDIA GeForce GTX 1080 Ti, 11 GB: cannot be used by this install" in text
    assert "Card 2 is from an older generation" in text and "--nvidia-older" in text
    assert "one install covers them all" in text and "cannot run RTX 50-series" in text
    assert "CUDA_VISIBLE_DEVICES=0,1 rosetta_jane" in text       # how to carry on with the two that work


def test_the_older_build_runs_all_three_and_says_how_to_use_them(monkeypatch):
    machine(monkeypatch, [RTX5000, RTX4000, GTX1080TI], older_build)
    found = cards.check()
    assert all(c["works"] for c in found)
    text = "\n".join(cards.describe(found, "2.14.1+cu126"))
    assert text.count(": ready") == 3 and "cannot be used" not in text and "CUDA_VISIBLE_DEVICES" not in text
    assert "--device-map auto" in text and "different generations work side by side" in text
    assert "--max-memory 0=14GiB,1=6GiB" in text and "Without it the model runs on the processor" in text


def test_a_card_too_new_for_the_older_build_is_not_called_old(monkeypatch):
    machine(monkeypatch, [GTX1080TI, RTX5090], older_build)
    found = cards.check()
    assert [c["works"] for c in found] == [True, False]
    text = "\n".join(cards.describe(found, "2.14.1+cu126"))
    assert "Card 1 could not run a test calculation" in text and "RTX 50-series" in text
    assert "older generation" not in text and "CUDA_VISIBLE_DEVICES=0 rosetta_jane" in text
    assert "--max-memory" not in text                             # only one card is ready


def test_no_card_at_all(monkeypatch):
    machine(monkeypatch, [], newest_build)
    assert cards.check() == []
    plain = "\n".join(cards.describe([], "2.15.0+cpu"))
    assert "models run on the processor" in plain and "processor-only build" in plain
    with_libraries = "\n".join(cards.describe([], "2.15.0+cu132"))
    assert "check that its driver is installed" in with_libraries and "580 driver series" in with_libraries


def test_a_card_that_cannot_even_be_read_does_not_stop_the_report(monkeypatch):
    machine(monkeypatch, [RTX5000, RTX4000], newest_build)
    def broken(i):
        if i == 1:
            raise RuntimeError("")
        return (7, 5)
    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    found = cards.check()
    assert found[0]["works"] and not found[1]["works"] and found[1]["why"] == "RuntimeError"
    assert "Card 1 could not run a test calculation: RuntimeError" in "\n".join(cards.describe(found, "x"))


def test_the_cards_command(monkeypatch, capsys):
    machine(monkeypatch, [RTX5000], newest_build)
    assert main(["cards"]) == 0
    said = capsys.readouterr().out
    assert "card 0: Quadro RTX 5000, 16 GB: ready" in said and "--device-map auto" in said
