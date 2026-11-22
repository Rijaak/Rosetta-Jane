"""Shared helpers for the tests: a tiny random model, fake data, and a single-machine trainer
that serves as the reference the multi-machine run must match."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

from rosetta_jane.lan.data import Schedule
from rosetta_jane.lan.host import TrainConfig
from rosetta_jane.lan.slicing import make_plan
from rosetta_jane.lan.stage import LoraSettings
from rosetta_jane.lan.worker import Worker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRET = b"test-secret-0123456789abcdef"
VOCAB = 96


FAMILIES = {
    "llama": ("LlamaConfig", "LlamaForCausalLM"),
    "qwen2": ("Qwen2Config", "Qwen2ForCausalLM"),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
    "mistral": ("MistralConfig", "MistralForCausalLM"),
    "phi3": ("Phi3Config", "Phi3ForCausalLM"),
    "gemma": ("GemmaConfig", "GemmaForCausalLM"),
    "gemma2": ("Gemma2Config", "Gemma2ForCausalLM"),
    "gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM"),
    "mixtral": ("MixtralConfig", "MixtralForCausalLM"),
    "starcoder2": ("Starcoder2Config", "Starcoder2ForCausalLM"),
}


def make_model(path: str, layers: int = 6, seed: int = 1, family: str = "llama") -> str:
    import transformers
    cfg_name, model_name = FAMILIES[family]
    LlamaConfig, LlamaForCausalLM = getattr(transformers, cfg_name), getattr(transformers, model_name)
    torch.manual_seed(seed)
    extra = {"head_dim": 8} if family.startswith("gemma") else {}
    cfg = LlamaConfig(**extra, vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


def make_examples(n: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        length = int(torch.randint(10, 28, (1,), generator=g))
        ids = torch.randint(3, VOCAB, (1, length), generator=g)
        labels = ids.clone()
        labels[0, : int(torch.randint(2, 6, (1,), generator=g))] = -100
        out.append((ids, labels))
    return out


def settings(**kw) -> LoraSettings:
    base = dict(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"], seed=3)
    base.update(kw)
    return LoraSettings(**base)


def train_cfg(**kw) -> TrainConfig:
    base = dict(total_steps=6, accum=3, lr=5e-3, warmup_steps=2, save_every=2, max_grad_norm=0.05,
                rpc_timeout=60, retry_wait=0.3, max_outage=60, run_id="t", data_seed=11)
    base.update(kw)
    return TrainConfig(**base)


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def plan_for(model_dir: str, guest_mems, ports):
    guests = [{"name": f"g{i + 1}", "addr": f"127.0.0.1:{p}", "mem_gb": m}
              for i, (m, p) in enumerate(zip(guest_mems, ports))]
    return make_plan(model_dir, 3.0, guests, host_reserve_gb=1.0)


def start_thread_worker(model_dir, plan_path, name, work_dir, port, secret=SECRET):
    worker = Worker(model_dir, plan_path, name, work_dir, secret)
    t = threading.Thread(target=worker.serve, args=(f"127.0.0.1:{port}",), daemon=True)
    t.start()
    wait_port(port)
    return worker


def start_process_worker(model_dir, plan_path, name, work_dir, port, secret_file):
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "rosetta_jane.lan", "worker", "--model", model_dir, "--plan", plan_path,
         "--name", name, "--listen", f"127.0.0.1:{port}", "--secret-file", secret_file,
         "--work-dir", work_dir],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_port(port, proc)
    return proc


def wait_port(port: int, proc=None, timeout: float = 120.0) -> None:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("worker process exited early")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("worker did not start listening")


def reference_train(model_dir: str, init_state: dict, examples, s: LoraSettings, cfg: TrainConfig, lr_at):
    """Plain single-machine LoRA training with the same recipe. Returns the adapter weights."""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(model, LoraConfig(r=s.rank, lora_alpha=s.alpha, lora_dropout=0.0,
                                          target_modules=list(s.target_modules), bias="none"))
    set_peft_model_state_dict(pm, init_state)
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, betas=(s.beta1, s.beta2), eps=s.eps, weight_decay=s.weight_decay)
    order = Schedule(len(examples), cfg.data_seed)
    for step in range(cfg.total_steps):
        for micro in range(cfg.accum):
            ids, labels = examples[order.index(step * cfg.accum + micro)]
            logits = pm(input_ids=ids).logits
            loss = F.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=-100)
            (loss / cfg.accum).backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(pm).items()}, pm


def max_diff(a: dict, b: dict) -> float:
    assert set(a) == set(b), (sorted(set(a) ^ set(b))[:4])
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
