"""Command line for Rosetta-Jane. Run `rosetta_jane --help`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from . import __version__

NAME = "Rosetta-Jane"


def _max_memory(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"--max-memory entries look like 0=14GiB, got {part!r}")
        out[int(key) if key.strip().isdigit() else key.strip()] = value.strip()
    return out


def _net_args(p: argparse.ArgumentParser, default_listen: str) -> None:
    p.add_argument("--group-code", default=None,
                   help="everyone who enters the same code forms a private network, like a co-op game "
                        "lobby. 'public' joins the open network. Or set ROSETTA_JANE_GROUP_CODE.")
    p.add_argument("--secret-file", default=None, help="a key file in place of a group code")
    p.add_argument("--peer", action="append", default=[], metavar="IP:PORT",
                   help="address of any node already in the network; repeat for more")
    p.add_argument("--listen", default=default_listen, help="address and port other nodes reach this one on")
    p.add_argument("--lan", action="store_true", help="also find nodes on the local network automatically")
    p.add_argument("--no-encrypt", action="store_true", help="only inside a VPN or a network you trust")
    p.add_argument("--work-dir", default="rosetta-jane-data")
    p.add_argument("--no-handoff-log", action="store_true",
                   help="do not keep fingerprints of what passes between devices (handoffs.jsonl in the work "
                        "folder; used to find the cause when a run across machines goes wrong)")


def _load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="folder holding the base model (safetensors)")
    p.add_argument("--load-4bit", action="store_true", help="load in 4-bit to save memory (GPU, bitsandbytes)")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", default=None, help="'auto' to spread across this machine's GPUs")
    p.add_argument("--max-memory", default=None, help="per-card cap, e.g. 0=14GiB,1=6.5GiB")
    p.add_argument("--allow-untested-model", action="store_true")


def _node(args, quiet: bool = False):
    from .net import Node, swarm_secret
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    if not code and not args.secret_file:
        raise SystemExit("give --group-code (or set ROSETTA_JANE_GROUP_CODE), or --secret-file")
    try:
        secret = swarm_secret(None if args.secret_file else code, args.secret_file)
    except ValueError as e:
        raise SystemExit(str(e))
    if code and "".join(code.split()).casefold() == "public" and not args.secret_file:
        print("NOTE: 'public' is the open network. Anyone can connect to this node.", file=sys.stderr)
    os.makedirs(args.work_dir, exist_ok=True)
    node = Node(secret, args.listen, encrypt=not args.no_encrypt, bootstrap=args.peer, beacon=args.lan,
                peers_file=os.path.join(args.work_dir, "peers.json"), quiet=quiet)
    from . import identity
    identity.of(node, args.work_dir)               # this device's own key, kept in its work folder
    if not args.no_handoff_log:
        from .handoff import HandoffLog
        node.handoffs = HandoffLog(os.path.join(args.work_dir, "handoffs.jsonl"), node.node_id)
    return node


def _load_kwargs(args) -> dict:
    return dict(dtype=args.dtype, load_4bit=args.load_4bit, device_map=args.device_map,
                max_memory=_max_memory(args.max_memory), allow_untested=args.allow_untested_model)


def _serve_range(text: str, span: int, node, model_dir: str):
    """Layers to serve: as given, or with 'auto' wherever the swarm is thinnest right now."""
    if text != "auto":
        try:
            start, end = (int(v) for v in text.split(":"))
            return start, end
        except ValueError:
            raise SystemExit("layers look like 8:16 (first layer, one past the last), or 'auto'")
    from .layers import model_fingerprint, model_shape
    from .swarm import choose_layers
    node.refresh(3)
    total = model_shape(model_dir)[1]
    start, end = choose_layers(node.table.peers(), model_fingerprint(model_dir), total, span)
    print(f"{NAME}: the swarm is thinnest at layers {start}-{end - 1}; serving those.", file=sys.stderr)
    return start, end


def cmd_serve(args) -> int:
    from .layers import LayerServer
    node = _node(args)
    node.start()
    start, end = _serve_range(args.layers, args.span, node, args.model)
    server = LayerServer(args.model, start, end, args.work_dir, max_sessions=args.max_sessions,
                         training=args.training, **_load_kwargs(args))
    server.attach(node)
    from .helpers import Sharing
    from .team import TeamService
    sharing = Sharing(node, server.helped).attach()
    teams = TeamService(node, server, codes=args.team, any_open=args.team_open, any_invited=args.team_invited,
                        say=lambda text: print(text, flush=True))
    print(f"{NAME}: serving layers {start}-{end - 1} of model {server.fingerprint} on port {node.port}. "
          + ("Also helping with training. " if args.training else "") + "Ctrl-C to stop.")
    try:
        ticks, told = 0, set()
        while True:
            time.sleep(5)
            ticks += 1
            teams.tick()
            for offer in sharing.offers():
                if offer["run"] not in told:
                    told.add(offer["run"])
                    print(f"  An adapter this device helped train ({offer['examples']} pieces of work) is being "
                          f"shared with its helpers. To look, or take a copy, run in another terminal:\n"
                          f"    rosetta_jane shared --model {args.model} --work-dir {args.work_dir} "
                          f"--group-code ...   (add --take {offer['run']} to take it)", flush=True)
            if ticks % 12 == 0:
                print(f"  peers known: {len(node.table.peers())}   answers helped: {server.steps_served} steps, "
                      f"{server.tokens_served} tokens   training examples helped: {server.examples_trained}"
                      + (f"   teams: {len(teams.links)}" if teams.links else ""), flush=True)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        teams.leave_all()
        server.helped.save()
        node.stop()
    return 0


def _checking(args):
    """The asking side's setup shared by answering and training: node, own layers, and the checks."""
    from transformers import AutoTokenizer

    from .layers import LocalAuditor, LocalEnds
    node = _node(args, quiet=True)
    ends = LocalEnds(args.model, args.own_layers, args.work_dir, **_load_kwargs(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    auditor = LocalAuditor(args.model, os.path.join(args.work_dir, "audit"), **_load_kwargs(args)) \
        if args.audit else None
    if args.redundancy != "auto" and not (args.redundancy.isdigit() and int(args.redundancy) >= 1):
        raise SystemExit("--redundancy is 'auto' or a whole number from 1 up")
    redundancy = None if args.redundancy == "auto" else int(args.redundancy)
    if redundancy == 1 and args.require_checked:
        raise SystemExit("--require-checked needs --redundancy 2 or more")
    from .canary import CanaryShelf
    canaries = None
    if not args.no_known_answers:
        same = [group.split(",") for group in args.treat_as_same]
        shelf = CanaryShelf(args.work_dir, ends.fingerprint, same)
        if shelf.refused:
            print(f"NOTE: ignoring known answers for {', '.join(shelf.refused)}: they were made on another "
                  "device. Known answers cannot be shared; run `rosetta_jane calibrate` here.", file=sys.stderr)
        if not shelf.books and not args.no_auto_calibrate:
            # Without known answers a lone peer's work cannot be checked at all, so they are
            # prepared the first time rather than left as a step people skip.
            print("First run with this model: preparing known-answer tests so peers can be checked. "
                  "This happens once.", file=sys.stderr)
            _known_answers(args, ends, tokenizer, 24, 4, 48, out=sys.stderr)
            shelf = CanaryShelf(args.work_dir, ends.fingerprint, same)
        canaries = shelf or None
    if canaries is not None:
        print(f"known-answer tests ready for peers running: {', '.join(sorted(canaries.books))}", file=sys.stderr)
    if canaries is None and not args.no_known_answers:
        print("NOTE: no known-answer tests prepared for this model. Run `rosetta_jane calibrate` so new "
              "peers can be tested before they are trusted.", file=sys.stderr)
    serve = getattr(args, "serve", None)
    args._service = None
    node.start()
    if serve:                                      # one program that asks, serves and trains
        from .layers import LayerServer
        start, end = _serve_range(serve, args.serve_span, node, args.model)
        args._service = LayerServer(args.model, start, end, args.work_dir, training=args.serve_training,
                                   **_load_kwargs(args))
        args._service.attach(node)
    return node, ends, tokenizer, auditor, canaries, redundancy


def _gateway(args):
    from .api import Gateway
    from .control import load_or_make_token
    node, ends, tokenizer, auditor, canaries, redundancy = _checking(args)
    token = None
    if getattr(args, "controls", False):
        token = load_or_make_token(os.path.join(args.work_dir, "control-token"))
    return node, Gateway(node, ends, tokenizer, args.name, service=args._service, control_token=token,
                         model_dir=args.model, work_dir=args.work_dir, variant=args.variant,
                         max_new_tokens=args.max_new_tokens, redundancy=redundancy, auditor=auditor,
                         canaries=canaries, canary_rate=args.known_answer_rate,
                         require_verified=args.require_checked, tolerance=args.tolerance,
                         audit_rate=args.audit_rate if args.audit else 0.0)


def cmd_gateway(args) -> int:
    node, gateway = _gateway(args)
    gateway.serve(args.api)
    print(f"{NAME}: answering on http://{args.api} as model '{args.name}'.\n"
          f"Your device keeps layers 0-{args.own_layers - 1} and the output layer; the swarm runs the rest.\n"
          "Point NOMAD's remote AI server address here. Ctrl-C to stop.")
    if gateway.service is not None:
        print(f"Running the p2p service for layers {gateway.service.start}-{gateway.service.end - 1}"
              + (" (training help included)." if gateway.service.training else "."))
    if gateway.control_token:
        print("Controls are on. A front end needs the token in "
              f"{os.path.join(args.work_dir, 'control-token')} to change anything.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        node.stop()
    return 0


def cmd_ask(args) -> int:
    from .swarm import NoRoute
    node, gateway = _gateway(args)
    try:
        node.refresh(3)
        ids = gateway.chat_ids([{"role": "user", "content": args.prompt}])
        final = gateway.answer(ids, {"temperature": args.temperature},
                               lambda text: print(text, end="", flush=True))
        print(f"\n[{final['eval_count']} tokens through {final['peers_used']} peers, "
              f"{final['reroutes']} reroutes, {final['checked_steps']} checked and "
              f"{final['unchecked_steps']} unchecked steps, {final['known_answer_tests']} known-answer "
              f"tests, {final['peers_blocked']} peers blocked]")
    except NoRoute as e:
        raise SystemExit(f"the swarm cannot answer right now: {e}")
    finally:
        node.stop()
    return 0


def _known_answers(args, ends, tokenizer, count: int, chunk: int, max_tokens: int, out=sys.stdout):
    import random

    from .canary import calibrate, make_prompts
    prompts = []
    for text in make_prompts(count, random.Random()):
        try:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                                add_generation_prompt=True)
            ids = list(getattr(ids, "input_ids", ids))
        except Exception:
            ids = list(tokenizer(text)["input_ids"])
        prompts.append(ids[:max_tokens])
    print(f"{NAME}: working out known answers for {len(prompts)} private test prompts, "
          f"{chunk} layers at a time. This is slow by design.", file=out)
    kw = _load_kwargs(args)
    return calibrate(args.model, args.work_dir, ends, prompts, chunk=chunk, dtype=kw["dtype"],
                     load_4bit=kw["load_4bit"], device_map=kw["device_map"], max_memory=kw["max_memory"],
                     allow_untested=kw["allow_untested"],
                     progress=lambda a, n: print(f"  layers {a} of {n}", file=out, flush=True))


def cmd_calibrate(args) -> int:
    from transformers import AutoTokenizer

    from .layers import LocalEnds
    os.makedirs(args.work_dir, exist_ok=True)
    ends = LocalEnds(args.model, args.own_layers, args.work_dir, **_load_kwargs(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    book = _known_answers(args, ends, tokenizer, args.count, args.chunk, args.max_tokens)
    print(f"done: {book.count} tests for peers running precision '{book.variant}', saved in {args.work_dir}. "
          "To check peers of another precision too, run calibrate again with that precision's options. "
          "Keep these files private; anyone who has them can pass your tests.")
    return 0


def _precision(text: str) -> dict:
    if text == "4bit":
        return {"load_4bit": True, "dtype": "float16"}
    if text in ("float32", "float16", "bfloat16"):
        return {"dtype": text}
    raise SystemExit(f"precision is one of float32, float16, bfloat16, 4bit; got {text!r}")


def cmd_measure(args) -> int:
    import random
    import shutil

    from transformers import AutoTokenizer

    from .canary import make_prompts, measure_precisions
    from .layers import LocalEnds
    a, b = _precision(args.a), _precision(args.b)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [list(tokenizer(text)["input_ids"])[:args.max_tokens]
               for text in make_prompts(args.count, random.Random())]
    scratch = os.path.join(args.work_dir, "measure")

    def ends_for(folder, opts):
        return LocalEnds(args.model, args.own_layers, folder, dtype=opts.get("dtype"),
                         load_4bit=bool(opts.get("load_4bit")), device_map=args.device_map,
                         max_memory=_max_memory(args.max_memory), allow_untested=args.allow_untested_model)

    for opts in (a, b):
        opts.update(device_map=args.device_map, max_memory=_max_memory(args.max_memory))
    print(f"{NAME}: running the same {len(prompts)} prompts through every layer at {args.a} and at {args.b}.")
    try:
        gaps = measure_precisions(args.model, scratch, ends_for, prompts, a, b, chunk=args.chunk,
                                  allow_untested=args.allow_untested_model)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    layers = sorted(gaps)
    shown = sorted(set(layers[::max(1, len(layers) // 8)] + layers[-1:]))
    for layer in shown:
        print(f"  after layer {layer - 1:>3}: {gaps[layer] * 100:6.2f}% apart")
    worst = max(gaps.values())
    print(f"largest gap: {worst * 100:.2f}%   (peers are compared with a tolerance of {args.tolerance * 100:.0f}%)")
    if worst < args.tolerance / 2:
        print(f"Well inside the tolerance. These two can share known answers: --treat-as-same {args.a},{args.b}")
    elif worst < args.tolerance:
        print("Inside the tolerance, but with little margin. Keep separate known answers for each.")
    else:
        print("Outside the tolerance. These must never be compared; keep separate known answers for each.")
    return 0


def cmd_compare(args) -> int:
    from .handoff import compare
    for path in args.files:
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {path}")
    r = compare(args.files, args.tolerance)
    for path, count in r["files"].items():
        print(f"{path}: {count} handoffs recorded")
    print(f"\n{r['devices']} devices. {r['delivered_intact']} of {r['handoffs_sent']} handoffs arrived intact "
          "(same fingerprint on both ends).")
    if r["unmatched"]:
        print(f"{len(r['unmatched'])} were sent but have no matching arrival. Either the other device's file "
              "was not given, or the data was lost or changed on the way. First few:")
        for u in r["unmatched"][:5]:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(u['t']))}  {u['node']} -> {u['peer']}  {u['op']}"
                  f"  layers {u['layers']}  {u['hash']}")
    print(f"\n{r['work_compared']} pieces of work were given to more than one peer.")
    print(f"  {r['rounding_differences']} differed only by rounding (within {args.tolerance:.0%}).")
    print(f"  {len(r['real_differences'])} differed by more than that.")
    for d in r["real_differences"][:10]:
        gap = "unknown" if d["norm_gap"] is None else f"{d['norm_gap']:.2%}"
        print(f"  {time.strftime('%H:%M:%S', time.localtime(d['t']))}  {d['op']}  layers {d['layers']}  "
              f"{d['peers'][0]} vs {d['peers'][1]}: overall size differs by {gap}, "
              f"{d['top_shared']} of 8 largest values in the same places")
    return 1 if r["real_differences"] else 0


def cmd_peers(args) -> int:
    node = _node(args, quiet=True).start()
    try:
        node.refresh(args.rounds)
        peers = node.table.peers()
        print(f"{len(peers)} peers reachable")
        for p in sorted(peers, key=lambda p: p.addr):
            offers = ", ".join(f"{s['model'][:8]} layers {s['layers'][0]}-{s['layers'][1] - 1} "
                               f"({s['variant']}, {s['free']} free)" for s in p.services) or "not serving"
            print(f"  {p.addr:<24} {offers}")
    finally:
        node.stop()
    return 0


def cmd_train(args) -> int:
    from .lan.data import read_pairs, tokenize_pairs
    from .swarm import NoRoute
    from .train import SwarmTrainer
    pairs = read_pairs(args.data)
    node, ends, tokenizer, auditor, canaries, redundancy = _checking(args)
    link = None
    try:
        from .scoring import describe, split_holdout
        examples, cut = tokenize_pairs(pairs, tokenizer, args.max_tokens)
        examples, held = split_holdout(examples)
        if held:
            print(f"{len(held)} of your pairs are kept back and never trained on, to score the result with.")
        steps = args.steps or max(1, round(len(examples) * args.epochs / args.accum))
        print(f"{NAME}: training on {len(examples)} pairs for {steps} steps"
              + (f" ({cut} pairs cut to {args.max_tokens} tokens)" if cut else "") + ".\n"
              f"Your device keeps layers 0-{args.own_layers - 1}, the output layer and every trained weight. "
              "Peers that help can see the hidden states for your text and the trained weights for their own "
              "layers.")
        common = dict(total_steps=steps, accum=args.accum, lr=args.lr, warmup_steps=args.warmup_steps,
                      seed=args.seed, data_seed=args.seed, save_every=args.save_every, run_id=args.run_id,
                      redundancy=redundancy, tolerance=args.tolerance, auditor=auditor,
                      audit_rate=args.audit_rate if args.audit else 0.0, canaries=canaries,
                      canary_rate=args.known_answer_rate, variant=args.variant)
        if args.team:                              # train inside a team: only its signed-on devices do the work
            from .team import NoTeam, TeamLink
            try:
                link = TeamLink(node, model=ends.fingerprint, code=args.team, want=[[0, args.own_layers]]).join()
            except (NoTeam, PermissionError) as e:
                raise ValueError(f"team {args.team}: {e}") from None
            if link.kind != "teaching":
                raise ValueError(f"team {args.team} is for sessions. Training together happens in a teaching "
                                 "team: `rosetta_jane team lead --teaching`.")
            common["team"] = link.id
            print("Training inside the teaching team: only devices signed on to it, and helping with training, "
                  "do the work.")
        if args.from_adapter:                      # carry an adapter further instead of starting from nothing
            trainer = SwarmTrainer.continuing(node, ends, args.model, args.work_dir, args.from_adapter, **common)
            print(f"Continuing from the adapter in {args.from_adapter} (rank {trainer.rank}, alpha "
                  f"{trainer.alpha:g}; its own settings are kept). It is not changed; the result is a new adapter.")
        else:
            trainer = SwarmTrainer(node, ends, args.model, args.work_dir, rank=args.rank, alpha=args.alpha,
                                   targets=args.targets.split(","), **common)
        node.refresh(3)
        stats = trainer.train(examples)
        score = trainer.compare(held, name=os.path.basename(os.path.normpath(args.out)), held_out=True) if held else None
        from .control import flattery_examples
        from .flattery import describe as describe_flattery
        flattery = trainer.flattery(flattery_examples(tokenizer, args.max_tokens))
        out = trainer.export(args.out, args.base_model_name or os.path.abspath(args.model))
        with open(os.path.join(out, "rosetta-jane-run.json"), "w") as f:
            json.dump({"name": os.path.basename(os.path.normpath(args.out)), "base_model_fingerprint": ends.fingerprint,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": len(examples),
                       "base_model_name": os.path.basename(os.path.normpath(args.model)),
                       "pairs_kept_back": len(held), "steps": stats["steps"], "final_loss": stats["final_loss"],
                       "work_checked": stats["checked_examples"], "work_unchecked": stats["unchecked_examples"],
                       "peers_blocked": stats["peers_blocked"], "score": score, "flattery": flattery,
                       **trainer.record()}, f, indent=2)
        print("\n".join(describe(score) + describe_flattery(flattery)))
        helpers = trainer.record().get("helpers", 0)
        if helpers:
            print(f"{helpers} other device{'s' if helpers != 1 else ''} did work on this adapter. Whether they get "
                  f"a copy is your choice. To offer it: rosetta_jane share --adapter {out} --group-code ...")
        print(f"done: {stats['steps']} steps, final loss {stats['final_loss']:.4f}. "
              f"{stats['checked_examples']} checked and {stats['unchecked_examples']} unchecked pieces of work, "
              f"{stats['peers_blocked']} peers blocked, {stats['redone']} examples redone.\n"
              f"Adapter saved to {out}")
    except NoRoute as e:
        raise SystemExit(f"training stopped: {e}\nProgress is saved; run the same command to carry on.")
    except ValueError as e:
        raise SystemExit(str(e))
    finally:
        if link is not None:
            link.leave()
        node.stop()
    return 0


def _with_adapter(args):
    """The asking side plus a trainer holding the adapter in --adapter, for scoring it or trying it."""
    from .train import SwarmTrainer
    node, ends, tokenizer, auditor, canaries, redundancy = _checking(args)
    try:
        trainer = SwarmTrainer.from_adapter(
            node, ends, args.model, os.path.join(args.work_dir, "scoring"), args.adapter, redundancy=redundancy,
            tolerance=args.tolerance, auditor=auditor, canaries=canaries, canary_rate=args.known_answer_rate,
            variant=args.variant)
    except ValueError as e:
        node.stop()
        raise SystemExit(str(e))
    node.refresh(3)
    return node, ends, tokenizer, trainer, os.path.basename(os.path.normpath(args.adapter))


def cmd_score(args) -> int:
    """Score an adapter on your own questions: the unchanged model against the model with the adapter."""
    from .lan.data import read_pairs, tokenize_pairs
    from .scoring import describe
    from .swarm import NoRoute
    pairs = read_pairs(args.data)
    node, ends, tokenizer, trainer, name = _with_adapter(args)
    try:
        examples, _ = tokenize_pairs(pairs, tokenizer, args.max_tokens)
        result = trainer.compare(examples, name=name, held_out=False)
        out = os.path.join(args.work_dir, f"score-card-{name}-{time.strftime('%Y%m%d-%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print("\n".join(describe(result)))
        print(f"Score card saved to {out}. It stays on this device unless you choose to share it.\n"
              "A score informs the decision to use an adapter. Nothing applies one for you.")
    except NoRoute as e:
        raise SystemExit(f"scoring stopped: {e}")
    except ValueError as e:
        raise SystemExit(str(e))
    finally:
        node.stop()
    return 0


def cmd_flattery(args) -> int:
    """Measure whether an adapter goes along with false statements more than the unchanged model does."""
    from .control import flattery_examples, save_fact
    from .flattery import describe
    from .swarm import NoRoute
    node, ends, tokenizer, trainer, name = _with_adapter(args)
    try:
        result = trainer.flattery(flattery_examples(tokenizer, args.max_tokens))
        print("\n".join(describe(result)))
        try:
            save_fact(args.adapter, "flattery", result)
            print("Kept with the adapter, so it is shown to people before they join a team that uses it.")
        except OSError:
            pass
    except NoRoute as e:
        raise SystemExit(f"the check stopped: {e}")
    finally:
        node.stop()
    return 0


def cmd_scores(args) -> int:
    """Write the Model Adapter Score document: every score held on this device, side by side."""
    from .scorebook import TITLE, build, collect
    out = args.out or f"{TITLE}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(build(args.work_dir))
    models = collect(args.work_dir)
    print(f"{TITLE}: {sum(len(m['adapters']) for m in models)} adapters across {len(models)} models. "
          f"Saved to {out}. Open it in a browser; it prints in landscape.")
    return 0


def cmd_cards(args) -> int:
    """Say which graphics cards this install can use. Each card is tried with a small calculation."""
    from .cards import report
    print("\n".join(report()))
    return 0


def cmd_try(args) -> int:
    """Ask one question and see both answers: the unchanged model, and the model with the adapter."""
    from .swarm import NoRoute
    node, ends, tokenizer, trainer, name = _with_adapter(args)
    try:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True,
                                            add_generation_prompt=True)
        ids = list(getattr(ids, "input_ids", ids))
        stops = [t for t in {ends.eos_token_id, getattr(tokenizer, "eos_token_id", None)} if isinstance(t, int)]
        without, with_it = trainer.try_out(ids, args.max_new_tokens, stops)
        print("Without the adapter:\n  " + tokenizer.decode(without, skip_special_tokens=True))
        print(f"\nWith \"{name}\":\n  " + tokenizer.decode(with_it, skip_special_tokens=True))
        print("\nThis was a trial. The adapter has not been applied to anything.")
    except NoRoute as e:
        raise SystemExit(f"the trial stopped: {e}")
    finally:
        node.stop()
    return 0


# -- teams: an adapter applied for a group because its members signed on ------------------------------
def _team_session(node, gateway, args) -> int:
    """The terminal's team screen: a few commands, and everything else is a question for the model."""
    import threading

    from .swarm import NoRoute
    from .team import JOINING_WORDS
    leading = gateway.team.role == "leader"
    done = threading.Event()

    def watch():                                   # say so when the team begins, grows or ends
        seen = None
        while not done.wait(1.0):
            t = gateway.team
            now = (t.begun, t.ended, len(t.roster()) if leading else t.members)
            for r in t.requests() if leading and t.ended is None else []:
                if r["id"] not in told:
                    told.add(r["id"])
                    print(f"\n[team] {r['who']} asks to join" + (f" ({r['about']})" if r.get("about") else "")
                          + f". Type: approve {r['id']}   or   decline {r['id']}", flush=True)
            if seen is not None and now != seen:
                if now[1] is not None:
                    print(f"\n[team] The team is over: {now[1]}. Answers are from the unchanged model again. "
                          + ("Type rate to answer a few questions about the adapter, or press Enter to finish."
                             if rates and t.rated is None else "Press Enter to finish."), flush=True)
                    return
                if now[0] != seen[0] and now[0]:
                    print(f"\n[team] Begun. Answers now come from the model with \"{t.name}\".", flush=True)
                if now[2] != seen[2]:
                    print(f"\n[team] {now[2]} device{'s' if now[2] != 1 else ''} in the team.", flush=True)
            seen = now
    threading.Thread(target=watch, daemon=True).start()
    teaching = gateway.team.kind == "teaching"
    rates = not leading and not teaching and gateway.team.desc.get("adapter") is not None
    me = gateway.desk.who()
    print(f"On team screens you are {me['shown']}. The tag in brackets comes from this device's own key; "
          "the name is whatever you chose.")
    print("Commands: " + (("" if teaching else "begin, ") + "who, devices, invite ADDRESS, requests, approve N, "
                          "decline N, remove N, open | ask | invisible (who can sign on), "
                          + ("" if teaching else "honour teamed | anyone | mine (whose teaching points the team "
                                                "takes), ") + "end. "
                          if leading else "who, " + ("rate (answer a few questions about the adapter), blind [QUESTION] "
                                              "(two unlabelled answers; you say which was right), " if rates else "")
                          + "leave. ")
          + ("A teaching team does not query: this device asks no questions until it leaves." if teaching else
             "Anything else is a question.")
          + ("" if leading or teaching else "\nTeaching points: type points to see what this device holds, "
             "boost N to spend N (each is 1% more of the team's computing power, for an hour)."))
    told = set()
    try:
        for line in sys.stdin:
            text = line.strip()
            t = gateway.team
            word, _, rest = text.partition(" ")
            try:
                if rates and word == "rate":       # the community rating; still possible just after the team ends
                    _ask_marks(gateway)
                    if t.ended is not None:
                        break
                    continue
                if rates and word == "blind" and t.ended is None:
                    got = gateway.blind({"messages": [{"role": "user", "content": rest.strip()}]}
                                        if rest.strip() else {})["blind"]
                    print(f"Question: {got['question']}\nAnswer A:\n  {got['a']}\nAnswer B:\n  {got['b']}\n"
                          "One of these is the adapter's and one is the unchanged model's. Which was right? "
                          "Type a, b, both or neither.", flush=True)
                    shown = gateway.team_control("pick", {"id": got["id"],
                                                          "choice": sys.stdin.readline().strip().lower()})["revealed"]
                    print(f"Answer {shown['adapter_was'].upper()} was the adapter's. Your answer joins its "
                          "community rating.")
                    continue
                if t.ended is not None:
                    break
                if not text:
                    continue
                if not leading and (text == "points" or word == "boost"):
                    got = gateway.team_control("points" if text == "points" else "boost",
                                               {"points": int(rest) if rest.strip().isdigit() else 0})["points"]
                    held = ", ".join(f"{row['left']} from {row['from']}" for row in got["from"])
                    print(f"Teaching points this device holds: {got['balance']}" + (f" ({held})" if held else "")
                          + ". "
                          + (f"In force now: {got['boost']} (your share of the team's computing power is "
                             f"{100 + got['boost']} to everyone else's 100), for {got['minutes']} more minutes."
                             if got["boost"] else "None in force."))
                    if got.get("note"):
                        print(got["note"])
                    if any(not t.get("confirmed") for t in got.get("taken", [])):
                        print("Some of those points were taken without their own leader confirming them, because "
                              "that leader is not in the group right now.")
                    continue
                if text in ("end", "leave", "quit", "exit"):
                    if rates and t.rated is None:
                        print("Before you go, a few questions about the adapter.")
                        _ask_marks(gateway)
                    break
                if text == "who":
                    r = gateway.team_report()
                    print(f"Team {r['code'] or r['team']}: adapter \"{r['name']}\", "
                          f"{'begun' if r['begun'] else 'not begun'}, {JOINING_WORDS.get(r['joining'], '')}.")
                    for card in r.get("roster", []):
                        role = ", ".join(x for x in ("asks" if card["asks"] else "",
                                                     f"serves layers {card['serves'][0]}-{card['serves'][1] - 1}"
                                                     if card["serves"] else "") if x)
                        print(f"  {card['who']}: {card['state']}" + (f" ({role})" if role else "")
                              + (f"   +{card['boost']}% share" if card.get("boost") else "")
                              + (f"   {card['about']}" if card.get("about") else "")
                              + (f"   [remove {card['id']}]" if card.get("id") else ""))
                    for q in r.get("requests", []):
                        print(f"  Asking to join: {q['who']}" + (f" ({q['about']})" if q.get("about") else "")
                              + f"   [approve {q['id']} / decline {q['id']}]")
                    if not leading:
                        print(f"  {r['members']} devices in the team.")
                    if r.get("missing"):
                        print(f"  No signed-on device serves layers {r['missing']} yet.")
                    elif r.get("missing_for_you"):
                        print(f"  For your own questions, another device has to sign on for layers "
                              f"{r['missing_for_you']}: a device does not route through the layers it serves itself.")
                elif leading and word == "begin":
                    gateway.team_control("begin", {"force": rest.strip() == "anyway"})
                elif leading and text in ("open", "ask", "invisible"):
                    gateway.team_control("settings", {"joining": text})
                    print(f"Who can sign on: {JOINING_WORDS[text]}. Remembered for your next team.")
                elif leading and text == "requests":
                    waiting = gateway.team_report()["requests"]
                    print("\n".join(f"  {q['who']}" + (f" ({q['about']})" if q.get("about") else "")
                                    + f": approve {q['id']} / decline {q['id']}"
                                    for q in waiting) or "  Nobody is asking to join.")
                elif leading and word == "honour" and not teaching:
                    from .team import HONOURS_WORDS
                    gateway.team_control("settings", {"honours": rest.strip()})
                    print(f"This team now takes {HONOURS_WORDS[rest.strip()]}. Remembered for your next team.")
                elif leading and word in ("approve", "decline"):
                    gateway.team_control(word, {"request": rest.strip()})
                    print("Approved: it is signed on and supplied with the adapter at its next check."
                          if word == "approve" else "Declined.")
                elif leading and word == "remove":
                    gateway.team_control("remove", {"member": rest.strip()})
                    print("Removed. It drops the adapter at its next check and cannot rejoin this team.")
                elif leading and word == "invite":
                    gateway.team_control("invite", {"addr": rest.strip()})
                    print(f"Invited {rest.strip()}. Its owner decides.")
                elif leading and text == "devices":
                    for d in gateway.team_control("devices", {})["devices"]:
                        what = f"serves layers {d['serves'][0]}-{d['serves'][1] - 1}" if d["serves"] else "not serving"
                        print(f"  {d['addr']:<24} {(d['who'] or '-'):<24} {what}"
                              + ("  in the team" if d["in_team"] else "  invited" if d["invited"] else
                                 "  in another team" if d["busy"] else "")
                              + (f"  ({d['about']})" if d.get("about") else ""))
                else:
                    ids = gateway.chat_ids([{"role": "user", "content": text}])
                    final = gateway.answer(ids, {"temperature": args.temperature},
                                           lambda piece: print(piece, end="", flush=True))
                    how = "unchanged model: the team has not begun" if not final["team"] else \
                        "unchanged model, on the team's devices" if t.use().plain else \
                        ('with the adapter "' + t.name + '"')
                    print(f"\n[{how}, through {final['peers_used']} peers, {final['checked_steps']} checked and "
                          f"{final['unchecked_steps']} unchecked steps]")
            except NoRoute as e:
                print(f"The team cannot answer right now: {e}")
            except ValueError as e:
                print(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        done.set()
        gateway.leave_team()
        if leading:
            gateway.team.settle()                  # let the members hear it before this program goes
        if gateway.service is not None:
            gateway.service.helped.save()
        node.stop()
    print("Team ended. Nothing stays applied." if leading else "Left the team. Nothing stays applied.")
    return 0


def _ask_marks(gateway) -> None:
    """Put the community rating's questions to the person, one at a time. Each can be skipped."""
    from .team import QUESTIONS
    given = {}
    for key, question in QUESTIONS.items():
        print(f"{question} Type a mark from 1 (rarely) to 5 (nearly always) and press Enter, or just press "
              "Enter to skip.", flush=True)
        mark = sys.stdin.readline().strip()
        if mark.isdigit():
            given[key] = int(mark)
    if not given:
        print("No marks given.")
        return
    try:
        gateway.team_control("rate", {"marks": given})
        print(f"Thank you. Your {len(given)} mark{'s' if len(given) != 1 else ''} join the adapter's community "
              "rating, apart from its teacher rating.")
    except ValueError as e:
        print(str(e))


def cmd_team_lead(args) -> int:
    """Lead a team with an adapter you hold. Others sign on; you begin; answers then use the adapter."""
    from .team import JOINING_WORDS, fact_lines
    node, gateway = _gateway(args)
    try:
        node.refresh(3)
        if args.invisible_device:
            gateway.desk.hidden = gateway.desk.invitations.closed = True     # for this run only
        if args.called is not None:
            gateway.team_control("settings", {"name": args.called})
        if args.teaching and args.adapter:
            raise ValueError("a teaching team is not led with an adapter. To train an adapter further, start "
                             "the run with `train --team CODE --from-adapter FOLDER`.")
        if args.honour is not None:
            gateway.team_control("settings", {"honours": args.honour})
        host = gateway.desk.lead(args.adapter, os.path.basename(os.path.normpath(args.adapter)) if args.adapter
                                 else None, args.joining, args.limit, kind="teaching" if args.teaching else "session")
    except ValueError as e:
        node.stop()
        raise SystemExit(str(e))
    print(f"{NAME}: you are leading a team "
          + ("for teaching: its devices work together on training an adapter.\n"
             "  To train, from another terminal: rosetta_jane train ... --team \"" + host.code + "\"\n"
             "  When the run is done, offering the adapter to the devices that helped is your choice "
             "(rosetta_jane share).\n" if host.kind == "teaching" else
             f"with the adapter \"{host.name}\".\n" if not host.plain else
             "with no adapter: it pools its devices to run the unchanged model.\n") + ""
          f"  Team code: {host.code}\n  Who can sign on: {JOINING_WORDS[host.joining]}.\n"
          f"  This device is at port {node.port}.\n"
          "What people are shown before they sign on:\n    " + "\n    ".join(fact_lines(host.describe())) + "\n"
          "Devices that serve layers must sign on too."
          + ("" if host.kind == "teaching" else " Type 'begin' once the layers are covered."))
    return _team_session(node, gateway, args)


def cmd_team_join(args) -> int:
    """Sign on to a team: by its code, by waiting for an invitation, or the first open team for your model."""
    from .team import NoTeam, teams_in_group
    node, gateway = _gateway(args)
    try:
        node.refresh(3)
        if args.code:
            req = {"code": args.code}
        elif args.invited:
            print(f"Waiting for an invitation. This device is at port {node.port}; the leader's 'devices' "
                  "list shows its address.")
            deadline, found = time.time() + args.invited * 60, []
            while not found and time.time() < deadline:
                time.sleep(1.0)
                found = gateway.invitations.list(gateway.ends.fingerprint)
            if not found:
                raise SystemExit("no invitation arrived")
            req = {"team": found[0]["team"]}
        else:                                      # the first team listed for this model: open ones first
            found = teams_in_group(node, gateway.ends.fingerprint)
            if not found:
                raise SystemExit("no team for this model is listed in the group right now. Give a team code, "
                                 "or lead a team yourself with `rosetta_jane team lead`.")
            req = {"team": found[0]["team"]}
        if args.invisible_device:
            gateway.desk.hidden = gateway.desk.invitations.closed = True     # for this run only
        if args.called is not None:
            gateway.team_control("settings", {"name": args.called})
        look = gateway.look_team(req)
        print("Before you sign on:\n  " + "\n  ".join(look["lines"]))
        if not look["fits"]:
            raise SystemExit("That team is for a different model than the one on this device.")
        asks = look["joining"] == "ask" and "code" not in req and not args.invited
        if not args.yes:
            print(("Ask this team's leader to let this device in? If they approve, it is signed on and supplied "
                   "with the adapter at once. " if asks else "Sign this device on? ")
                  + "Answers then come from this team, once the leader begins, until you leave. "
                  "Type y and press Enter; anything else does not.", flush=True)
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                raise SystemExit("Not signed on. Nothing was applied.")
        gateway.join_team(req)
        if gateway.team is None or gateway.team.ended is not None:      # a request: wait for the leader
            print("Asked. Waiting for the leader to approve.", flush=True)
            while gateway.desk.pending is not None and gateway.desk.pending["state"] == "waiting":
                time.sleep(0.5)
            if gateway.desk.pending is not None:
                raise SystemExit(f"Not signed on: {gateway.desk.pending['state']}.")
    except (ValueError, NoTeam, PermissionError) as e:
        node.stop()
        raise SystemExit(str(e))
    except SystemExit:
        node.stop()
        raise
    print(f"Signed on. {'The team has begun.' if gateway.team.begun else 'Waiting for the leader to begin.'}")
    return _team_session(node, gateway, args)


def cmd_teams(args) -> int:
    """List the teams anyone in this group can see: open ones, and ones that take an invitation."""
    from .team import JOINING_WORDS, teams_in_group
    node = _node(args, quiet=True).start()
    try:
        node.refresh(args.rounds)
        found = teams_in_group(node)
        print(f"{len(found)} team{'s' if len(found) != 1 else ''} visible in this group "
              "(invisible teams are not listed; they take their code or an invitation)")
        for lobby, title in (("session", "Teams running sessions (querying the model):"),
                             ("teaching", "Teaching teams (training an adapter):")):
            if any(t["kind"] == lobby for t in found):
                print(title)
            for t in (t for t in found if t["kind"] == lobby):
                _print_team(t, JOINING_WORDS)
    finally:
        node.stop()
    return 0


def _print_team(t: dict, words: dict) -> None:
    facts = t["facts"]
    community = facts.get("community") or {}
    print(f"  Model {(t['model_name'] or '') + ' '}({t['model']}). "
          + (f"Adapter \"{t['name']}\" ({t['adapter']})." if t["adapter"] else "No adapter.") + "\n"
          f"      {t['members']}/{t['limit']} devices, {'begun' if t['begun'] else 'gathering'}, "
          f"{words[t['joining']]}."
          + ("" if not t["adapter"] else
             f"\n      training work checked {facts.get('work_checked', 'unknown')}, "
             f"unchecked {facts.get('work_unchecked', 'unknown')}"
             f"\n      teacher rating: {(facts.get('score') or {}).get('verdict', 'not scored')}; "
             "community rating: " + (f"{community['average']} out of 5 from {community['ratings']}"
                                     if community.get("ratings", 0) >= 5 else
                                     f"{community['ratings']} marks, too few to rely on"
                                     if community.get("ratings") else "none yet")))


def cmd_share(args) -> int:
    """Offer a finished adapter to the devices that helped train it, for as long as this runs."""
    from .helpers import Sharing
    node = _node(args, quiet=True).start()
    try:
        model = "0" * 16
        if args.model:
            from .layers import model_fingerprint
            model = model_fingerprint(args.model)
        sharing = Sharing(node).attach()
        row = sharing.open(args.adapter, model)
        print(f"{NAME}: offering \"{row['name']}\" to the {row['helpers']} device"
              f"{'s' if row['helpers'] != 1 else ''} whose work went into it. Each owner decides whether to take "
              "a copy. Their own model stays unchanged either way. Ctrl-C to stop offering.")
        taken = 0
        while True:
            time.sleep(2)
            now = sharing.report()[0]["taken"]
            if now != taken:
                taken = now
                print(f"  {taken} of {row['helpers']} have taken a copy.", flush=True)
    except ValueError as e:
        raise SystemExit(str(e))
    except KeyboardInterrupt:
        print("stopped offering")
    finally:
        node.stop()
    return 0


def cmd_shared(args) -> int:
    """See the adapters this device helped train that are on offer, and take a copy of one."""
    from .helpers import HelpLog, Sharing
    from .layers import model_fingerprint
    from .team import fact_lines
    node = _node(args, quiet=True).start()
    try:
        sharing = Sharing(node, HelpLog(os.path.join(args.work_dir, "helped.json")))
        node.refresh(args.rounds)
        offers = sharing.offers()
        if args.take:
            kept = sharing.take(args.take, args.work_dir, model_fingerprint(args.model), args.name)
            print(f"Stored as \"{kept}\" in {os.path.join(args.work_dir, 'adapters', kept)}. Nothing has been "
                  "applied. You can try it, score it, lead a team with it, or train it further.")
            return 0
        print(f"{len(offers)} adapter{'s' if len(offers) != 1 else ''} this device helped train "
              f"{'is' if len(offers) == 1 else 'are'} on offer right now.")
        for offer in offers:
            found = sharing.look(offer["run"])
            print(f"  run {offer['run']}: \"{found['name']}\", {offer['examples']} pieces of work done here.\n    "
                  + "\n    ".join(fact_lines(found)[1:]) + f"\n    To take a copy: add --take {offer['run']}")
    except (KeyError, PermissionError, ValueError) as e:
        raise SystemExit(str(e.args[0]) if e.args else str(e))
    finally:
        node.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rosetta_jane", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"{NAME}: one model across many devices, peer to peer, with no central server.\n"
                    "Answer questions and train, through strangers' devices or across your own PCs.",
        epilog="training:\n"
               "  train        through peers anywhere; your device keeps every trained weight\n"
               "  lan plan     across PCs you own: decide which PC holds which layers\n"
               "  lan worker   across PCs you own: run on each helping PC\n"
               "  lan train    across PCs you own: run on the PC that holds the training text\n"
               "teams (pool devices to ask with an adapter or with none, or to teach):\n"
               "  team lead    lead a team; people and devices sign on. --teaching for a teaching team\n"
               "  team join    sign on to a team (by code, by invitation, by request, or any open team)\n"
               "  teams        list the teams visible in your group, in both lobbies\n"
               "  share        offer a finished adapter to the devices that helped train it\n"
               "  shared       see what is on offer to this device, and take a copy\n"
               "Every command takes --help.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("lan", help="train across several PCs you own: `lan plan`, `lan worker`, `lan train`")

    s = sub.add_parser("serve", help="serve this device: run a range of a model's layers for others")
    _load_args(s)
    s.add_argument("--layers", default="auto", metavar="FIRST:END",
                   help="e.g. 8:16 serves layers 8 to 15. 'auto' (the default) looks at the swarm and takes "
                        "the layers nobody is serving, or the ones with the fewest serving peers")
    s.add_argument("--span", type=int, default=8, help="with 'auto', how many layers to serve")
    s.add_argument("--max-sessions", type=int, default=4, help="answers this device helps with at once")
    s.add_argument("--training", action="store_true",
                   help="also help other people train (heavier work than answering; off unless you say so)")
    s.add_argument("--team", action="append", default=[], metavar="CODE",
                   help="sign this device on to the team with this code, so it applies that team's adapter "
                        "for that team's work; repeat for more teams")
    s.add_argument("--team-open", action="store_true",
                   help="sign this device on to any open team for its model (off unless you say so)")
    s.add_argument("--team-invited", action="store_true",
                   help="say yes in advance: sign this device on to any team whose leader invites it, and let "
                        "leaders see it in their lists of devices to invite (off unless you say so)")
    _net_args(s, "0.0.0.0:7700")
    s.set_defaults(fn=cmd_serve)

    def ask_side(s):
        _load_args(s)
        s.add_argument("--own-layers", type=int, default=2,
                       help="how many of the first layers stay on this device (more is more private)")
        s.add_argument("--variant", default=None, help="only use peers running this precision, e.g. 4bit")
        s.add_argument("--name", default="rosetta", help="model name shown to programs that connect")
        s.add_argument("--max-new-tokens", type=int, default=512)
        s.add_argument("--redundancy", default="auto",
                       help="how many independent peers compute each stretch of layers, to catch wrong "
                            "results. 'auto' (the default) uses 2 whenever at least 3 peers are available; "
                            "1 turns the comparison off.")
        s.add_argument("--no-known-answers", action="store_true",
                       help="do not test peers with the known answers prepared by `calibrate`")
        s.add_argument("--no-auto-calibrate", action="store_true",
                       help="do not prepare known-answer tests automatically on the first run")
        s.add_argument("--known-answer-rate", type=float, default=0.2,
                       help="how often an already-tested peer is tested again")
        s.add_argument("--treat-as-same", action="append", default=[], metavar="A,B",
                       help="precisions that may share one set of known answers, e.g. float16,bfloat16. "
                            "Only after `measure-precision` shows their gap is well under the tolerance.")
        s.add_argument("--audit", action="store_true",
                       help="recompute work on this device to see who is right: always when peers disagree, "
                            "and at random even when they agree (needs room to load those layers here)")
        s.add_argument("--audit-rate", type=float, default=0.1,
                       help="with --audit, the share of steps checked at random even when peers agree")
        s.add_argument("--require-checked", action="store_true",
                       help="refuse to answer unless every stretch of layers has that many peers")
        s.add_argument("--tolerance", type=float, default=0.02,
                       help="how far two peers' results may differ and still count as agreeing")
        _net_args(s, "0.0.0.0:0")

    s = sub.add_parser("gateway", help="answer through the swarm; NOMAD connects to this")
    ask_side(s)
    s.add_argument("--api", default="127.0.0.1:11435", help="where programs on this device connect")
    s.add_argument("--serve", default=None, metavar="FIRST:END",
                   help="also serve layers to the swarm from this same program: 8:16, or 'auto' to take "
                        "whichever layers the swarm is shortest of")
    s.add_argument("--serve-span", type=int, default=8, help="with --serve auto, how many layers to serve")
    s.add_argument("--serve-training", action="store_true", help="with --serve, also help other people train")
    s.add_argument("--controls", action="store_true",
                   help="let a front end (NOMAD's settings page) switch serving on and off and run training. "
                        "It must present the token saved in the work folder.")
    s.set_defaults(fn=cmd_gateway)

    s = sub.add_parser("ask", help="ask one question from the terminal")
    ask_side(s)
    s.add_argument("prompt")
    s.add_argument("--temperature", type=float, default=0.7)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("train", help="train a LoRA adapter on your question/answer pairs through the swarm")
    ask_side(s)
    s.add_argument("--data", required=True, help="JSONL file with 'question' and 'answer' on each line")
    s.add_argument("--out", default="rosetta-jane-adapter", help="folder for the finished adapter")
    s.add_argument("--epochs", type=float, default=3.0)
    s.add_argument("--steps", type=int, default=0, help="optimizer steps; overrides --epochs")
    s.add_argument("--accum", type=int, default=16, help="pairs per optimizer step")
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--warmup-steps", type=int, default=20)
    s.add_argument("--rank", type=int, default=16)
    s.add_argument("--alpha", type=float, default=32.0)
    s.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    s.add_argument("--max-tokens", type=int, default=1024, help="longest pair, in tokens")
    s.add_argument("--save-every", type=int, default=25, help="steps between saves on this device")
    s.add_argument("--run-id", default="run", help="name of this run; the same name carries on a stopped run")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--base-model-name", default=None, help="model name to record in the adapter")
    s.add_argument("--team", default=None, metavar="CODE",
                   help="train inside the team with this code: only devices signed on to it do the work")
    s.add_argument("--from-adapter", default=None, metavar="FOLDER",
                   help="carry on from a finished adapter (yours, or one shared with you) instead of starting "
                        "from nothing; the result is a new adapter and the old one is kept")
    s.set_defaults(fn=cmd_train)

    def serve_too(s):
        s.add_argument("--serve", default=None, metavar="FIRST:END",
                       help="also serve layers from this same program (8:16, or 'auto'); they sign on with it")
        s.add_argument("--serve-span", type=int, default=8)
        s.add_argument("--serve-training", action="store_true")
        s.add_argument("--temperature", type=float, default=0.7)
        s.add_argument("--invisible-device", action="store_true",
                       help="keep this device out of leaders' lists of devices to invite")
        s.add_argument("--called", default=None, metavar="NAME",
                       help="the name you go by on team screens (a label; nothing checks it). Remembered.")

    team = sub.add_parser("team", help="pool devices as a team, to ask with an adapter or with none, or to teach: "
                                        "`team lead`, `team join`")
    tsub = team.add_subparsers(dest="team_command", required=True)
    s = tsub.add_parser("lead", help="lead a team: with an adapter you hold, with none, or (--teaching) for training")
    ask_side(s)
    serve_too(s)
    s.add_argument("--adapter", default=None,
                   help="folder holding the adapter. Leave out for a team with no adapter, which pools its "
                        "devices' computing power to run the unchanged model")
    s.add_argument("--joining", choices=["open", "ask", "invisible"], default=None,
                   help="who can sign on: open (anyone in the group), ask (anyone can ask; you approve), "
                        "invisible (not listed; code or invitation only). Default: what you chose last time, "
                        "or open.")
    s.add_argument("--limit", type=int, default=None, help="most devices in the team")
    s.add_argument("--honour", choices=["teamed", "anyone", "mine"], default=None,
                   help="whose teaching points this team takes: teamed (points from leaders whose teams this "
                        "device has signed on to), anyone, or mine (only points earned with you). Default: what "
                        "you chose last time, or teamed.")
    s.add_argument("--teaching", action="store_true",
                   help="lead a teaching team, for training an adapter together. It is listed in its own lobby, "
                        "apart from teams that query the model")
    s.set_defaults(fn=cmd_team_lead)
    s = tsub.add_parser("join", help="sign on to a team")
    ask_side(s)
    serve_too(s)
    s.add_argument("code", nargs="?", default=None, help="the team's code; leave out to take the first open team")
    s.add_argument("--invited", type=float, default=0, metavar="MINUTES",
                   help="wait this long for an invitation from a leader instead")
    s.add_argument("--yes", action="store_true", help="sign on without being asked (you have read the facts)")
    s.set_defaults(fn=cmd_team_join)

    s = sub.add_parser("teams", help="list the teams visible in your group")
    _net_args(s, "0.0.0.0:0")
    s.add_argument("--rounds", type=int, default=4)
    s.set_defaults(fn=cmd_teams)

    s = sub.add_parser("share", help="offer a finished adapter to the devices that helped train it")
    s.add_argument("--adapter", required=True, help="folder holding the adapter you trained")
    s.add_argument("--model", default=None, help="the base model folder (only needed for adapters with no record)")
    _net_args(s, "0.0.0.0:0")
    s.set_defaults(fn=cmd_share)

    s = sub.add_parser("shared", help="adapters this device helped train that are on offer; take a copy")
    s.add_argument("--model", required=True, help="the base model folder on this device")
    s.add_argument("--take", default=None, metavar="RUN", help="take a copy of the adapter from this run")
    s.add_argument("--name", default=None, help="name to store it under")
    s.add_argument("--rounds", type=int, default=4)
    _net_args(s, "0.0.0.0:0")
    s.set_defaults(fn=cmd_shared)

    s = sub.add_parser("score", help="score an adapter on your own questions: the model with it against "
                                     "the model without it")
    ask_side(s)
    s.add_argument("--adapter", required=True, help="folder holding the adapter (as downloaded and unzipped)")
    s.add_argument("--data", required=True, help="JSONL file of question/answer pairs the adapter was not trained on")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("flattery", help="measure whether an adapter goes along with false statements more than "
                                        "the unchanged model does")
    ask_side(s)
    s.add_argument("--adapter", required=True, help="folder holding the adapter")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.set_defaults(fn=cmd_flattery)

    s = sub.add_parser("scores", help="write the Model Adapter Score document: every score on this device, "
                                      "side by side, best first")
    s.add_argument("--work-dir", default="rosetta-jane-data")
    s.add_argument("--out", default=None, help="where to save it (default: 'Model Adapter Score.html')")
    s.set_defaults(fn=cmd_scores)

    s = sub.add_parser("cards", help="list this machine's graphics cards and say which ones this install can use")
    s.set_defaults(fn=cmd_cards)

    s = sub.add_parser("try", help="ask one question and see both answers: without an adapter and with it")
    ask_side(s)
    s.add_argument("--adapter", required=True, help="folder holding the adapter (as downloaded and unzipped)")
    s.add_argument("prompt")
    s.set_defaults(fn=cmd_try, max_new_tokens=48)

    s = sub.add_parser("calibrate", help="prepare private known-answer tests for checking peers")
    _load_args(s)
    s.add_argument("--own-layers", type=int, default=2, help="the same value you ask with")
    s.add_argument("--count", type=int, default=24, help="how many tests to prepare")
    s.add_argument("--chunk", type=int, default=4, help="layers this device loads at one time")
    s.add_argument("--max-tokens", type=int, default=48, help="longest test prompt, in tokens")
    s.add_argument("--work-dir", default="rosetta-jane-data")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("measure-precision",
                       help="measure how far two precisions differ on your model, layer by layer")
    s.add_argument("--model", required=True)
    s.add_argument("--a", default="float32")
    s.add_argument("--b", default="bfloat16")
    s.add_argument("--own-layers", type=int, default=2)
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--chunk", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=48)
    s.add_argument("--tolerance", type=float, default=0.02)
    s.add_argument("--device-map", default=None)
    s.add_argument("--max-memory", default=None)
    s.add_argument("--allow-untested-model", action="store_true")
    s.add_argument("--work-dir", default="rosetta-jane-data")
    s.set_defaults(fn=cmd_measure)

    s = sub.add_parser("compare-handoffs",
                       help="lay several devices' handoff files side by side to see where numbers stopped matching")
    s.add_argument("files", nargs="+", help="handoffs.jsonl from each device's work folder")
    s.add_argument("--tolerance", type=float, default=0.02)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("peers", help="list the peers this device can reach and what they serve")
    _net_args(s, "0.0.0.0:0")
    s.add_argument("--rounds", type=int, default=4)
    s.set_defaults(fn=cmd_peers)
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["lan"]:                       # the multi-PC trainer lives under `rosetta_jane lan ...`
        from .lan.cli import main as lan_main
        return lan_main(argv[1:])
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
