"""Start the p2p service without PyTorch: python -m rosetta_jane.mobile serve --model DIR --layers 4:8

This is what the Android app runs underneath. It is also a way to try the phone
engine on any computer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


from ..routing import choose_layers as choose  # noqa: E402  (the same rule desktop peers use)


def ask_ids(args) -> int:
    import json

    from ..net import Node, swarm_secret
    from ..swarm import Generation
    from .ends import Ends
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    node = Node(swarm_secret(code), "127.0.0.1:0", bootstrap=args.peer, quiet=True).start()
    try:
        node.refresh(3)
        ends = Ends(args.model, args.own_layers)
        scores = []
        real = ends.logits
        ends.logits = lambda h: scores.append(real(h)) or scores[-1]
        gen = Generation(node, ends, find_wait=10.0,
                         redundancy=None if args.redundancy == "auto" else int(args.redundancy))
        out = gen.run([int(v) for v in args.ids.split(",")], args.new)
        print("RESULT " + json.dumps({
            "ids": out, "scores": [[float(v) for v in s] for s in scores], "checked": gen.verified_steps,
            "unchecked": gen.unverified_steps, "disputes": gen.disputes, "blocked": sorted(gen.caught),
            "reroutes": gen.reroutes, "peers": sum(len(h.replicas()) for h in gen.route)}), flush=True)
    finally:
        node.stop()
    return 0


def try_ids(args) -> int:
    import json

    from ..net import Node, swarm_secret
    from .ends import Ends
    from .train import Trainer
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    node = Node(swarm_secret(code), "127.0.0.1:0", bootstrap=args.peer, quiet=True).start()
    out = {}
    try:
        node.refresh(3)
        try:
            trainer = Trainer.from_adapter(node, Ends(args.model, args.own_layers), args.work_dir, args.adapter,
                                           find_wait=10.0,
                                           redundancy=None if args.redundancy == "auto" else int(args.redundancy))
        except ValueError as e:
            print("RESULT " + json.dumps({"refused": str(e)}), flush=True)
            return 0
        if args.ids:
            out["without"], out["with"] = trainer.try_out([int(v) for v in args.ids.split(",")], args.new)
        if args.examples:
            with open(args.examples) as f:
                out["score"] = trainer.compare([(list(i), list(l)) for i, l in json.load(f)], name="tried")
        out["blocked"] = sorted(trainer.caught)
        print("RESULT " + json.dumps(out), flush=True)
    finally:
        node.stop()
    return 0


def train_ids(args) -> int:
    import json

    from ..net import Node, swarm_secret
    from . import st
    from .ends import Ends
    from .train import Trainer
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    with open(args.examples) as f:
        examples = [(list(ids), list(labels)) for ids, labels in json.load(f)]
    init = None
    if args.init:
        with open(args.init, "rb") as f:
            init = st.loads(f.read())
    node = Node(swarm_secret(code), "127.0.0.1:0", bootstrap=args.peer, quiet=True).start()
    try:
        node.refresh(3)
        ends = Ends(args.model, args.own_layers)
        trainer = Trainer(node, ends, args.work_dir, total_steps=args.steps, accum=args.accum, lr=args.lr,
                          warmup_steps=args.warmup_steps, rank=args.rank, alpha=args.alpha,
                          targets=args.targets.split(","), save_every=args.save_every, find_wait=10.0, init=init,
                          redundancy=None if args.redundancy == "auto" else int(args.redundancy))

        class Halt(Exception):
            pass

        def on_step(step):
            if args.stop_after and step == args.stop_after:
                raise Halt()
        try:
            held = []
            if args.hold_out:
                from ..scoring import split_holdout
                examples, held = split_holdout(examples, args.hold_out)
            stats = trainer.train(examples, on_step=on_step)
            score = trainer.compare(held, name=os.path.basename(args.out), held_out=True) if held else None
            trainer.export(args.out, len(examples), os.path.abspath(args.model), score, len(held))
            stats["score"] = score
        except Halt:
            stats = {"stopped_at": trainer.step}
        stats["blocked"] = sorted(trainer.caught)
        print("RESULT " + json.dumps(stats), flush=True)
    finally:
        node.stop()
    return 0


def main(argv=None) -> int:
    from ..handoff import HandoffLog
    from ..net import Node, swarm_secret
    from .engine import Unsupported, fingerprint, layout, read_config, scan
    from .peer import NumpyLayerServer
    p = argparse.ArgumentParser(prog="rosetta_jane.mobile", description="Rosetta-Jane's p2p service without PyTorch.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="run the p2p service for a stretch of a model's layers")
    s.add_argument("--model", required=True, help="folder holding the model (safetensors)")
    s.add_argument("--layers", default="auto", help="FIRST:END, or 'auto' for wherever the swarm is thinnest")
    s.add_argument("--span", type=int, default=2, help="with 'auto', how many layers to serve")
    s.add_argument("--group-code", default=None)
    s.add_argument("--peer", action="append", default=[])
    s.add_argument("--listen", default="0.0.0.0:7700")
    s.add_argument("--lan", action="store_true")
    s.add_argument("--no-encrypt", action="store_true")
    s.add_argument("--training", action="store_true", help="also help other people train")
    s.add_argument("--max-sessions", type=int, default=2)
    s.add_argument("--work-dir", default="rosetta-jane-data")
    q = sub.add_parser("ask-ids", help="answer a prompt given as token numbers (for testing)")
    q.add_argument("--model", required=True)
    q.add_argument("--own-layers", type=int, default=2)
    q.add_argument("--ids", required=True, help="comma-separated token numbers")
    q.add_argument("--new", type=int, default=8)
    q.add_argument("--redundancy", default="auto")
    q.add_argument("--group-code", default=None)
    q.add_argument("--peer", action="append", default=[])
    q.add_argument("--work-dir", default="rosetta-jane-data")
    t = sub.add_parser("train-ids", help="train on examples given as token numbers (for testing)")
    t.add_argument("--model", required=True)
    t.add_argument("--own-layers", type=int, default=2)
    t.add_argument("--examples", required=True, help="JSON file: [[ids, labels], ...]")
    t.add_argument("--init", default=None, help="safetensors file of starting adapters")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup-steps", type=int, default=1)
    t.add_argument("--rank", type=int, default=4)
    t.add_argument("--alpha", type=float, default=8.0)
    t.add_argument("--targets", default="q_proj,v_proj")
    t.add_argument("--redundancy", default="auto")
    t.add_argument("--stop-after", type=int, default=0, help="stop early after this many steps (to test resuming)")
    t.add_argument("--save-every", type=int, default=25)
    t.add_argument("--hold-out", type=int, default=0, help="keep every Nth example back and score the result on them")
    t.add_argument("--out", required=True)
    t.add_argument("--group-code", default=None)
    t.add_argument("--peer", action="append", default=[])
    t.add_argument("--work-dir", default="rosetta-jane-data")
    y = sub.add_parser("try-ids", help="try or score an adapter given token numbers (for testing)")
    y.add_argument("--model", required=True)
    y.add_argument("--own-layers", type=int, default=2)
    y.add_argument("--adapter", required=True, help="folder holding the adapter")
    y.add_argument("--ids", default="", help="comma-separated token numbers to try")
    y.add_argument("--examples", default="", help="JSON file [[ids, labels], ...] to score on")
    y.add_argument("--new", type=int, default=6)
    y.add_argument("--redundancy", default="auto")
    y.add_argument("--group-code", default=None)
    y.add_argument("--peer", action="append", default=[])
    y.add_argument("--work-dir", default="rosetta-jane-data")
    args = p.parse_args(argv)
    if args.command == "try-ids":
        return try_ids(args)
    if args.command == "ask-ids":
        return ask_ids(args)
    if args.command == "train-ids":
        return train_ids(args)
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    if not code:
        raise SystemExit("give --group-code (or set ROSETTA_JANE_GROUP_CODE)")
    try:
        read_config(args.model)
        found = scan(args.model)
        total = layout(found.keys())[1]
        model_id = fingerprint(args.model, found)
        secret = swarm_secret(code)
    except (Unsupported, ValueError, OSError) as e:
        raise SystemExit(str(e))
    os.makedirs(args.work_dir, exist_ok=True)
    node = Node(secret, args.listen, encrypt=not args.no_encrypt, bootstrap=args.peer, beacon=args.lan,
                peers_file=os.path.join(args.work_dir, "peers.json"))
    node.handoffs = HandoffLog(os.path.join(args.work_dir, "handoffs.jsonl"), node.node_id)
    node.start()
    try:
        if args.layers == "auto":
            node.refresh(3)
            start, end = choose(node.table.peers(), model_id, total, args.span)
        else:
            start, end = (int(v) for v in args.layers.split(":"))
        server = NumpyLayerServer(args.model, start, end, training=args.training, max_sessions=args.max_sessions)
    except (Unsupported, ValueError) as e:
        node.stop()
        raise SystemExit(str(e))
    server.attach(node)
    print(f"Rosetta-Jane (no PyTorch): p2p service for layers {start}-{end - 1} of model {server.fingerprint} "
          f"on port {node.port}." + (" Also helping with training." if args.training else "") + " Ctrl-C to stop.",
          flush=True)
    try:
        while True:
            time.sleep(60)
            print(f"  peers known: {len(node.table.peers())}   answer steps helped: {server.steps_served}   "
                  f"training examples helped: {server.examples_trained}", flush=True)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
