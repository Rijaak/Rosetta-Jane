"""The Model Adapter Score document: every score on a device, by model then adapter, best first."""
import json
import os
import re

from rosetta_jane.cli import main
from rosetta_jane.scorebook import TITLE, build, collect, document


def card(right_before, right_after, loss_before, loss_after, pairs=40, held=True, **more):
    return {"adapter": "x", "base_model_fingerprint": "fp", "scored": "2026-11-22 09:00:00", "pairs": pairs,
            "answer_tokens": pairs * 9, "never_trained_on": held,
            "before": {"loss": loss_before, "right_next_word": right_before},
            "after": {"loss": loss_after, "right_next_word": right_after},
            "work_checked": 60, "work_unchecked": 20, "verdict": "better with the adapter", **more}


def adapter(work, name, facts):
    folder = os.path.join(work, "adapters", name)
    os.makedirs(folder)
    open(os.path.join(folder, "adapter_model.safetensors"), "wb").write(b"w")
    json.dump(facts, open(os.path.join(folder, "rosetta-jane-run.json"), "w"))


def sample(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    big = {"base_model_fingerprint": "aaaa1111", "base_model_name": "big-model"}
    small = {"base_model_fingerprint": "bbbb2222", "base_model_name": "small-model"}
    adapter(work, "farming", {**big, "score": card(0.40, 0.52, 2.1, 1.7)})                       # +12 points
    adapter(work, "first-aid", {**big, "score": card(0.40, 0.45, 2.1, 2.0),                      # +5 from training,
                                "my_score": card(0.38, 0.61, 2.2, 1.4, pairs=25, held=False)})   # +23 on my questions
    adapter(work, "worse", {**big, "received": True, "my_score": card(0.40, 0.31, 2.1, 2.6)})    # -9 points
    adapter(work, "unscored", {**big})
    adapter(work, "repair", {**small, "score": card(0.30, 0.34, 2.9, 2.8)})                      # +4 points
    json.dump({**card(0.30, 0.60, 2.9, 1.9), "adapter": "radio", "base_model_fingerprint": "bbbb2222"},
              open(os.path.join(work, "score-card-radio-20261122-090000.json"), "w"))            # +30, kept elsewhere
    return work


def test_models_then_adapters_with_the_best_first(tmp_path):
    models = collect(sample(tmp_path))
    assert [m["name"] for m in models] == ["small-model", "big-model"]      # small-model's best adapter gained most
    assert [a["name"] for a in models[0]["adapters"]] == ["radio", "repair"]
    assert [a["name"] for a in models[1]["adapters"]] == ["first-aid", "farming", "worse", "unscored"]
    first_aid = models[1]["adapters"][0]
    assert [round(e["gain"], 2) for e in first_aid["entries"]] == [0.23, 0.05]       # its best scoring leads
    assert first_aid["entries"][0]["scored_by"] == "You, on your own questions"
    assert models[1]["adapters"][3]["entries"] == [] and models[1]["adapters"][2]["received"] is True


def test_the_document_puts_each_adapter_beside_the_unchanged_model(tmp_path):
    work = sample(tmp_path)
    page = build(work)
    assert f"<title>{TITLE}</title>" in page and f"<h1>{TITLE}</h1>" in page
    assert "size: landscape" in page                                        # laid out for landscape paper
    assert page.index("small-model") < page.index("big-model")
    order = [page.index(f'class="adapter">{n}<') for n in ("first-aid", "farming", "worse", "unscored")]
    assert order == sorted(order)
    assert "Without the adapter" in page and "With the adapter" in page
    row = page[page.index('class="adapter">farming<'):page.index('class="adapter">worse<')]
    assert "40.0%" in row and "52.0%" in row and "2.100" in row and "1.700" in row and "+12.0 points" in row
    assert "never trained on" in row and "60 of 80" in row and "The run that trained it" in row
    worse = page[page.index('class="adapter">worse<'):page.index('class="adapter">unscored<')]
    assert "-9.0 points" in worse and 'class="down"' in worse and "shared with you" in worse
    assert "Not scored yet" in page and "kept in another folder" in page
    assert "not strictly comparable" in page and "Nothing in\nRosetta-Jane applies one" in page
    ranks = re.findall(r'<td class="rank"[^>]*>(\d+)</td>', page)
    assert ranks == ["1", "2", "1", "2", "3"]                                # ranking restarts for each model
    assert not re.search(r'(src|href)="http', page) and "<script" not in page   # one self-contained file
    empty = document([])
    assert "no adapters on this device yet" in empty


def test_what_other_people_wrote_cannot_run_or_break_the_page(tmp_path):
    work = str(tmp_path / "work")
    os.makedirs(work)
    nasty = '<script>alert(1)</script><img src=x onerror=alert(2)>"\'&'
    adapter(work, "from-a-stranger", {
        "base_model_fingerprint": nasty, "base_model_name": nasty, "received": True,
        "score": {**card(0.2, 0.9, 2.0, 0.5), "verdict": nasty, "scored": nasty, "pairs": "many",
                  "work_checked": float("inf"), "work_unchecked": -5},
        "my_score": {"before": {"loss": "low", "right_next_word": 0.1}, "after": {"loss": 1, "right_next_word": 0.2}}})
    adapter(work, "odd-numbers", {"base_model_fingerprint": "cccc", "score": card(-3, 7, float("nan"), 1.0)})
    adapter(work, "too-big", {"base_model_fingerprint": "cccc", "score": card(-3.0, 7.0, 2.0, 1.0)})
    json.dump([1, 2, 3], open(os.path.join(work, "score-card-list.json"), "w"))
    open(os.path.join(work, "score-card-broken.json"), "w").write("{")
    json.dump("not an object", open(os.path.join(work, "adapters", "odd-numbers", "rosetta-jane-run.json"), "w"))
    page = build(work)
    assert "<script" not in page and "onerror=alert" not in page.replace("&lt;img src=x onerror=alert(2)&gt;", "")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    models = {m["fingerprint"]: m for m in collect(work)}
    stranger = next(m for m in models.values() if m["adapters"][0]["name"] == "from-a-stranger")["adapters"][0]
    assert len(stranger["entries"]) == 1                    # the card with a non-number in it was dropped
    assert stranger["entries"][0]["pairs"] == 0 and stranger["entries"][0]["checked"] == 0
    names = {a["name"]: a for m in models.values() for a in m["adapters"]}
    assert names["odd-numbers"]["entries"] == []            # unreadable facts: listed, not scored
    capped = names["too-big"]["entries"][0]
    assert capped["right_before"] == 0.0 and capped["right_after"] == 1.0      # shares are held within 0 to 100%
    assert "0.0%" in page and "100.0%" in page and "700" not in page


def test_the_scores_command_writes_the_document(tmp_path, capsys):
    work = sample(tmp_path)
    out = str(tmp_path / "out.html")
    assert main(["scores", "--work-dir", work, "--out", out]) == 0
    assert f"<h1>{TITLE}</h1>" in open(out, encoding="utf-8").read()
    said = capsys.readouterr().out
    assert "6 adapters across 2 models" in said and "landscape" in said


def test_the_result_comes_from_the_numbers_not_from_what_a_card_says_about_itself(tmp_path):
    models = collect(sample(tmp_path))
    worse = next(a for a in models[1]["adapters"] if a["name"] == "worse")
    assert worse["entries"][0]["verdict"] == "worse with the adapter"       # the card itself claimed "better"
    page = build(str(tmp_path / "work"))
    row = page[page.index('class="adapter">worse<'):page.index('class="adapter">unscored<')]
    assert "worse with the adapter" in row and "better with the adapter" not in row


def test_the_gateway_hands_the_document_over_only_with_the_token(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from rosetta_jane.api import Gateway
    work = sample(tmp_path)
    gateway = Gateway.__new__(Gateway)
    gateway.control_token, gateway.work_dir, gateway.job = "secret-token", work, None
    server = gateway.serve("127.0.0.1:0")
    base = f"http://127.0.0.1:{server.server_address[1]}/api/rosetta/model-adapter-score"
    try:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            try:
                urllib.request.urlopen(urllib.request.Request(base, headers=headers), timeout=10)
                raise AssertionError("served without the token")
            except urllib.error.HTTPError as e:
                assert e.code == 401
        with urllib.request.urlopen(urllib.request.Request(base, headers={"Authorization": "Bearer secret-token"}),
                                    timeout=10) as r:
            assert r.headers["Content-Type"].startswith("text/html")
            assert r.read().decode() .count("<h2>") == 2
    finally:
        server.shutdown()
