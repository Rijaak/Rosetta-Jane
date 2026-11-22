"""The "Model Adapter Score" document: every score held on this device, side by side.

One landscape page a person can read, print or pass on. It is arranged by model,
then by adapter: the models whose best adapter did best come first, and within a
model the adapters that improved it most come first. Each adapter shows the model
without it beside the model with it.

It is built only from the adapters and score cards on this device. Scores made on
different sets of questions are not strictly comparable, and the document says so
on every row by stating who scored it, on how many pairs, and whether the adapter
had seen them.

Facts files can come from other people (inside a shared adapter), so every piece
of text is escaped and every number is checked before it reaches the page.

Plain Python, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import html
import json
import math
import os
import time
from typing import Dict, List, Optional

TITLE = "Model Adapter Score"
FACTS = "rosetta-jane-run.json"


from .scoring import verdict

def _num(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _text(value, limit: int = 80) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _entry(adapter: str, who: str, card, facts: dict) -> Optional[dict]:
    """One scoring of one adapter, with every field checked. None if the card is not usable."""
    if not isinstance(card, dict) or not isinstance(card.get("before"), dict) or not isinstance(card.get("after"), dict):
        return None
    values = [_num(card["before"].get("right_next_word")), _num(card["after"].get("right_next_word")),
              _num(card["before"].get("loss")), _num(card["after"].get("loss"))]
    if any(v is None for v in values):
        return None
    right_before, right_after = (min(1.0, max(0.0, v)) for v in values[:2])
    pairs = _num(card.get("pairs")) or 0
    return {"adapter": adapter, "scored_by": who, "pairs": int(pairs),
            "never_trained_on": card.get("never_trained_on") is True,
            "right_before": right_before, "right_after": right_after,
            "loss_before": values[2], "loss_after": values[3],
            "gain": right_after - right_before, "loss_gain": values[2] - values[3],
            "checked": int(_num(card.get("work_checked")) or 0), "unchecked": int(_num(card.get("work_unchecked")) or 0),
            "scored": _text(card.get("scored"), 19),
            # Worked out again from the numbers. The sentence inside a shared card is somebody's claim.
            "verdict": verdict({"loss": values[2]}, {"loss": values[3]}, int(pairs)),
            "received": facts.get("received") is True}


def collect(work_dir: str) -> List[dict]:
    """Every model that has adapters here, each with its adapters and their scorings, best first."""
    folder = os.path.join(work_dir, "adapters")
    models: Dict[str, dict] = {}
    names = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    for name in names:
        if not os.path.exists(os.path.join(folder, name, "adapter_model.safetensors")):
            continue
        try:
            with open(os.path.join(folder, name, FACTS)) as f:
                facts = json.load(f)
        except (OSError, ValueError):
            facts = {}
        if not isinstance(facts, dict):
            facts = {}
        fingerprint = _text(facts.get("base_model_fingerprint"), 32) or "unknown"
        model = models.setdefault(fingerprint, {"fingerprint": fingerprint, "name": "", "adapters": []})
        if not model["name"]:
            model["name"] = _text(facts.get("base_model_name"), 80)
        entries = [e for e in (_entry(name, "The run that trained it", facts.get("score"), facts),
                               _entry(name, "You, on your own questions", facts.get("my_score"), facts)) if e]
        entries.sort(key=_rank)
        model["adapters"].append({"name": name, "received": facts.get("received") is True, "entries": entries,
                                  "pairs_trained": int(_num(facts.get("pairs")) or 0)})
    # Score cards saved by the `score` command for adapters kept somewhere else.
    loose = sorted(n for n in (os.listdir(work_dir) if os.path.isdir(work_dir) else [])
                   if n.startswith("score-card-") and n.endswith(".json"))
    for filename in loose[-500:]:
        try:
            with open(os.path.join(work_dir, filename)) as f:
                card = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(card, dict):
            continue
        name = _text(card.get("adapter"), 40) or "adapter"
        entry = _entry(name, "You, on your own questions", card, {})
        if entry is None:
            continue
        fingerprint = _text(card.get("base_model_fingerprint"), 32) or "unknown"
        model = models.setdefault(fingerprint, {"fingerprint": fingerprint, "name": "", "adapters": []})
        adapter = next((a for a in model["adapters"] if a["name"] == name), None)
        if adapter is None:
            adapter = {"name": name, "received": False, "elsewhere": True, "entries": [], "pairs_trained": 0}
            model["adapters"].append(adapter)
        adapter["entries"].append(entry)
        adapter["entries"].sort(key=_rank)
    for model in models.values():                  # adapters that improved the model most come first;
        model["adapters"].sort(key=lambda a: (_rank(a["entries"][0]) if a["entries"] else (1, 0, 0), a["name"]))
    return sorted(models.values(),                 # models whose best adapter did best come first
                  key=lambda m: ((_rank(m["adapters"][0]["entries"][0]) if m["adapters"] and m["adapters"][0]["entries"]
                                  else (1, 0, 0)), m["name"], m["fingerprint"]))


def _rank(entry: dict):
    """Sort key: scored before unscored, then the biggest gain in right answers, then the biggest drop in loss."""
    return (0, -entry["gain"], -entry["loss_gain"])


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _bar(share: float, tone: str) -> str:
    width = max(0.0, min(100.0, share * 100))
    return f'<div class="bar"><div class="fill {tone}" style="width:{width:.1f}%"></div></div>'


def document(models: List[dict], generated: Optional[str] = None) -> str:
    """The whole document as one self-contained HTML page, laid out for landscape paper."""
    e = html.escape
    generated = generated or time.strftime("%Y-%m-%d %H:%M")
    out = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(TITLE)}</title>
<style>
  @page {{ size: landscape; margin: 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #1f2419; background: #fff;
         margin: 0; padding: 24px; font-size: 13px; line-height: 1.4; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; color: #424c36; }}
  h2 {{ font-size: 18px; margin: 28px 0 2px; color: #424c36; }}
  .sub {{ color: #5b6352; margin: 0 0 10px; }}
  .note {{ background: #fff4d6; border: 1px solid #e6d28f; border-radius: 6px; padding: 10px 12px; margin: 12px 0 0; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 980px; margin-top: 8px; }}
  th, td {{ border: 1px solid #d9dccf; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #eef0e6; font-weight: 600; }}
  th.group {{ text-align: center; }}
  th.without, td.without {{ background: #f6f6f1; }}
  th.with, td.with {{ background: #eef5e4; }}
  td.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  td.rank {{ text-align: center; font-weight: 600; width: 44px; }}
  .adapter {{ font-weight: 600; font-size: 14px; }}
  .tag {{ display: inline-block; font-size: 11px; color: #5b6352; border: 1px solid #c9cdbd; border-radius: 10px;
          padding: 0 7px; margin-left: 6px; font-weight: 400; }}
  .bar {{ height: 8px; background: #e4e6da; border-radius: 4px; margin-top: 4px; min-width: 90px; }}
  .fill {{ height: 8px; border-radius: 4px; }}
  .fill.plain {{ background: #9aa08c; }} .fill.tuned {{ background: #5d7a3a; }}
  .up {{ color: #3d6b1f; font-weight: 600; }} .down {{ color: #a33a2b; font-weight: 600; }} .flat {{ color: #5b6352; }}
  .small {{ color: #5b6352; font-size: 12px; }}
  tr {{ break-inside: avoid; }} h2 {{ break-after: avoid; }}
  @media print {{ body {{ padding: 0; }} .scroll {{ overflow: visible; }} table {{ min-width: 0; }} }}
</style></head><body>
<h1>{e(TITLE)}</h1>
<p class="sub">Every score held on this device, made {e(generated)}. Arranged by model, then by adapter:
models whose best adapter did best come first, and within a model the adapters that improved it most come first.</p>
<p class="note"><strong>How to read this.</strong> Each row puts the unchanged model beside the same model with one
adapter, on the same questions. "Right next word" is the share of answer words predicted correctly; loss is lower when
the model is surer of the right answer. These show whether an adapter learned the answers it was scored on, not whether
those answers are good. Scores made on different questions are not strictly comparable, so each row says who scored
it, on how many pairs, and whether the adapter had seen them. A score informs the decision to use an adapter. Nothing in
Rosetta-Jane applies one.</p>"""]
    if not any(m["adapters"] for m in models):
        out.append('<p class="sub" style="margin-top:24px">There are no adapters on this device yet.</p>')
    for model in models:
        title = model["name"] or "Model"
        out.append(f'<h2>{e(title)}</h2><p class="sub">Model ID {e(model["fingerprint"])} &middot; '
                   f'{len(model["adapters"])} adapter{"" if len(model["adapters"]) == 1 else "s"}</p>')
        out.append('<div class="scroll"><table><thead><tr><th rowspan="2">Rank</th><th rowspan="2">Adapter</th>'
                   '<th rowspan="2">Scored by</th><th rowspan="2">Pairs</th>'
                   '<th colspan="2" class="group without">Without the adapter</th>'
                   '<th colspan="2" class="group with">With the adapter</th>'
                   '<th rowspan="2">Change</th><th rowspan="2">Scoring checked</th><th rowspan="2">Result</th></tr>'
                   '<tr><th class="without">Right next word</th><th class="without">Loss</th>'
                   '<th class="with">Right next word</th><th class="with">Loss</th></tr></thead><tbody>')
        rank = 0
        for adapter in model["adapters"]:
            tag = '<span class="tag">shared with you</span>' if adapter["received"] else \
                '<span class="tag">kept in another folder</span>' if adapter.get("elsewhere") else \
                '<span class="tag">trained here</span>'
            label = f'<span class="adapter">{e(adapter["name"])}</span>{tag}'
            if not adapter["entries"]:
                out.append(f'<tr><td class="rank">&ndash;</td><td>{label}</td><td colspan="9" class="small">'
                           'Not scored yet. Score it on your own questions to place it here.</td></tr>')
                continue
            rank += 1
            span = len(adapter["entries"])
            for i, s in enumerate(adapter["entries"]):
                points = s["gain"] * 100
                tone = "up" if points > 0.05 else "down" if points < -0.05 else "flat"
                change = f'<span class="{tone}">{points:+.1f} points</span><br><span class="small">loss {-s["loss_gain"]:+.3f}</span>'
                seen = "never trained on" if s["never_trained_on"] else "chosen by the scorer"
                total = s["checked"] + s["unchecked"]
                checked = (f'{s["checked"]} of {total}' if total else "&ndash;")
                first = (f'<td class="rank" rowspan="{span}">{rank}</td><td rowspan="{span}">{label}</td>' if i == 0 else "")
                out.append(
                    f'<tr>{first}<td>{e(s["scored_by"])}<br><span class="small">{e(s["scored"])}</span></td>'
                    f'<td class="num">{s["pairs"]}<br><span class="small">{seen}</span></td>'
                    f'<td class="num without">{_pct(s["right_before"])}{_bar(s["right_before"], "plain")}</td>'
                    f'<td class="num without">{s["loss_before"]:.3f}</td>'
                    f'<td class="num with">{_pct(s["right_after"])}{_bar(s["right_after"], "tuned")}</td>'
                    f'<td class="num with">{s["loss_after"]:.3f}</td>'
                    f'<td class="num">{change}</td><td class="num">{checked}</td><td>{e(s["verdict"])}</td></tr>')
        out.append("</tbody></table></div>")
    out.append('<p class="small" style="margin-top:20px">Made by Rosetta-Jane from the score cards on one device. '
               'It was not sent anywhere. Share it only if you choose to.</p></body></html>')
    return "\n".join(out)


def build(work_dir: str) -> str:
    return document(collect(work_dir))
