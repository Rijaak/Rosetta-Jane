"""Command line for the multi-PC trainer. Run `rosetta_jane lan --help`."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

from . import __version__


def _max_memory(text: Optional[str]) -> Optional[dict]:
    """'0=14GiB,1=6.5GiB' -> {0: '14GiB', 1: '6.5GiB'}"""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"--max-memory entries look like 0=14GiB, got {part!r}")
        out[int(key) if key.strip().isdigit() else key.strip()] = value.strip()
    return out


def _add_secret_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--secret-file", default=None, help="file holding the run's shared secret")
    p.add_argument("--group-code", default=None,
                   help="a code everyone in the group types in, like a co-op game lobby. 'public' means "
                        "anyone may connect. You can also set ROSETTA_JANE_GROUP_CODE to keep it out of "
                        "your shell history.")
    p.add_argument("--no-encrypt", action="store_true",
                   help="turn off built-in encryption (only inside a VPN or on a network you trust)")


def _secret(args) -> bytes:
    from .protocol import PUBLIC_CODE, load_secret, secret_from_group_code
    code = args.group_code or os.environ.get("ROSETTA_JANE_GROUP_CODE")
    if args.secret_file and code:
        raise SystemExit("give either --secret-file or a group code, not both")
    if args.secret_file:
        return load_secret(args.secret_file)
    if not code:
        raise SystemExit("give --secret-file, --group-code, or set ROSETTA_JANE_GROUP_CODE")
    try:
        secret = secret_from_group_code(code)
    except ValueError as e:
        raise SystemExit(str(e))
    if "".join(code.split()).casefold() == PUBLIC_CODE:
        print("WARNING: group code 'public' lets anyone who can reach this machine connect to it.",
              file=sys.stderr)
    return secret


def _add_load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--load-4bit", action="store_true",
                   help="load this machine's layers in 4-bit to save memory (needs bitsandbytes and a GPU)")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                   help="number format for this machine's layers")
    p.add_argument("--device-map", default=None, help="'auto' to spread across this machine's GPUs")
    p.add_argument("--max-memory", default=None, help="per-card cap, e.g. 0=14GiB,1=6.5GiB")


def cmd_secret(args) -> int:
    from .protocol import new_secret
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists; refusing to overwrite it")
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(new_secret() + "\n")
    print(f"wrote {args.out}. Copy it to every machine in the run, by hand or USB stick, "
          "and do not post it anywhere.")
    return 0


def cmd_fingerprint(args) -> int:
    from .slicing import model_fingerprint
    print(model_fingerprint(args.model))
    return 0


def cmd_plan(args) -> int:
    from .slicing import make_plan
    guests = []
    for spec in args.guest:
        parts = spec.split("=")
        if len(parts) != 3:
            raise SystemExit("--guest looks like NAME=IP:PORT=MEMORY_GB, e.g. tower=192.168.1.20:7000=8")
        guests.append({"name": parts[0], "addr": parts[1], "mem_gb": float(parts[2])})
    plan = make_plan(args.model, args.host_mem, guests, host_reserve_gb=args.host_reserve,
                     allow_untested=args.allow_untested_model)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"model {plan['fingerprint']}: {plan['num_layers']} layers")
    for s in plan["stages"]:
        a, b = s["layers"]
        where = "this machine" if s["name"] == "host" else s["addr"]
        print(f"  {s['name']:<12} layers {a:>3}-{b - 1:<3} ({b - a} layers)  {where}")
    print(f"wrote {args.out}. Give the same file to every machine.")
    return 0


def cmd_worker(args) -> int:
    from .worker import Worker
    worker = Worker(args.model, args.plan, args.name, args.work_dir, _secret(args),
                    load_4bit=args.load_4bit, dtype=args.dtype, device_map=args.device_map,
                    max_memory=_max_memory(args.max_memory), encrypt=not args.no_encrypt)
    worker.serve(args.listen, ready_file=args.ready_file)
    return 0


def cmd_train(args) -> int:
    from transformers import AutoTokenizer

    from .data import read_pairs, tokenize_pairs
    from .host import Host, TrainConfig
    from .slicing import load_plan
    from .stage import LoraSettings

    plan = load_plan(args.plan)
    secret = _secret(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    examples, cut = tokenize_pairs(read_pairs(args.pairs), tokenizer, args.max_tokens)
    print(f"{len(examples)} training pairs" + (f" ({cut} cut short at {args.max_tokens} tokens)" if cut else ""))
    holdout = None
    if args.holdout:
        holdout, _ = tokenize_pairs(read_pairs(args.holdout), tokenizer, args.max_tokens)
        print(f"{len(holdout)} holdout pairs")
    total = args.steps or math.ceil(args.epochs * len(examples) / args.accum)
    settings = LoraSettings(rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                            target_modules=args.targets.split(","), weight_decay=args.weight_decay,
                            seed=args.seed, grad_checkpointing=args.grad_checkpointing)
    cfg = TrainConfig(total_steps=total, accum=args.accum, lr=args.lr, warmup_steps=args.warmup,
                      schedule=args.schedule, max_grad_norm=args.max_grad_norm,
                      save_every=args.save_every, fp16_scaling=args.fp16_scaling,
                      eval_every=args.eval_every, run_id=args.run_id, data_seed=args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    host = Host(args.model, plan, args.work_dir, secret, settings, cfg,
                load_4bit=args.load_4bit, dtype=args.dtype, device_map=args.device_map,
                max_memory=_max_memory(args.max_memory), encrypt=not args.no_encrypt)
    from .protocol import AuthError, MismatchError
    try:
        result = host.train(examples, holdout)
        out = host.export(os.path.join(args.work_dir, f"adapter-{args.run_id}"))
    except (AuthError, MismatchError) as e:
        raise SystemExit(f"cannot start: {e}")
    finally:
        host.close()
    print(json.dumps(result, indent=1))
    print(f"finished. Adapter for the full model written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rosetta_jane lan", description="LoRA fine-tuning split across several PCs.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("secret", help="make a shared secret file for a run")
    s.add_argument("--out", default="secret.txt")
    s.set_defaults(fn=cmd_secret)

    s = sub.add_parser("fingerprint", help="print the ID of a local model copy")
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_fingerprint)

    s = sub.add_parser("plan", help="decide which machine holds which layers")
    s.add_argument("--model", required=True)
    s.add_argument("--host-mem", type=float, required=True, help="GPU memory on the host, in GB")
    s.add_argument("--host-reserve", type=float, default=2.0,
                   help="GB of host memory kept back for embeddings and the output layer")
    s.add_argument("--guest", action="append", default=[], metavar="NAME=IP:PORT=GB",
                   help="one per guest machine, in chain order; repeat for more break points")
    s.add_argument("--out", default="plan.json")
    s.add_argument("--allow-untested-model", action="store_true",
                   help="plan for a model family that has not been verified to split correctly")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("worker", help="run on each guest machine")
    s.add_argument("--model", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--name", required=True, help="this machine's name in the plan")
    s.add_argument("--listen", default="0.0.0.0:7000")
    _add_secret_args(s)
    s.add_argument("--work-dir", default="jane-train-worker")
    s.add_argument("--ready-file", default=None, help=argparse.SUPPRESS)
    _add_load_args(s)
    s.set_defaults(fn=cmd_worker)

    s = sub.add_parser("train", help="run on the host machine, after the workers are up")
    s.add_argument("--model", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--pairs", required=True, help="JSONL file with 'question' and 'answer' on each line")
    s.add_argument("--holdout", default=None)
    _add_secret_args(s)
    s.add_argument("--work-dir", default="jane-train-run")
    s.add_argument("--run-id", default="run")
    s.add_argument("--epochs", type=float, default=3.0)
    s.add_argument("--steps", type=int, default=0, help="optimizer steps; overrides --epochs")
    s.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    s.add_argument("--max-grad-norm", type=float, default=1.0)
    s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--save-every", type=int, default=200)
    s.add_argument("--eval-every", type=int, default=0)
    s.add_argument("--rank", type=int, default=16)
    s.add_argument("--alpha", type=int, default=32)
    s.add_argument("--dropout", type=float, default=0.05)
    s.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    s.add_argument("--weight-decay", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--grad-checkpointing", action="store_true", help="trade speed for memory")
    s.add_argument("--fp16-scaling", action="store_true",
                   help="scale the loss to keep half-precision gradients from vanishing")
    _add_load_args(s)
    s.set_defaults(fn=cmd_train)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
