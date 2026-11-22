"""One machine's part of the model: load its slice, attach LoRA, run it, save it.

A Stage is used the same way on the host and on a guest. The only difference is
that the host stage also owns the two ends of the model: the token embeddings at
the front and the final norm plus output head at the back.
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load as st_load
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save as st_save
from safetensors.torch import save_file as st_save_file


@dataclass
class LoraSettings:
    """Settings every machine in a run must share. The host sends them to guests."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 0
    grad_checkpointing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoraSettings":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        s = cls(**known)
        if not (1 <= int(s.rank) <= 1024) or not isinstance(s.target_modules, list):
            raise ValueError("bad LoRA settings")
        if not all(isinstance(m, str) and m.replace("_", "").isalnum() for m in s.target_modules):
            raise ValueError("bad LoRA target module names")
        return s


def _find_decoder(model: nn.Module) -> nn.Module:
    """Find the module that owns the stack of decoder layers and the final norm."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0 and hasattr(module, "norm"):
            return module
    raise RuntimeError(
        "unsupported model layout: expected a decoder with '.layers' and a final '.norm' "
        "(the LLaMA, Qwen and Mistral family)")


class Stage:
    def __init__(self, slice_dir: str, is_host: bool, settings: LoraSettings, *,
                 load_4bit: bool = False, dtype: Optional[str] = None,
                 device_map=None, max_memory: Optional[dict] = None):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        self.is_host = is_host
        self.settings = settings
        kwargs: dict = {}
        if dtype:
            kwargs["dtype"] = getattr(torch, dtype)
        if device_map is not None:
            kwargs["device_map"] = device_map
        if max_memory:
            kwargs["max_memory"] = max_memory
        if load_4bit:
            from transformers import BitsAndBytesConfig
            compute = getattr(torch, dtype) if dtype else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
            kwargs.setdefault("device_map", "auto")
        model = AutoModelForCausalLM.from_pretrained(slice_dir, **kwargs)
        for p in model.parameters():
            p.requires_grad_(False)

        decoder = _find_decoder(model)
        # Hidden states handed to the next machine must be the raw output of the
        # last layer here. The final norm belongs only at the very end of the
        # model, so take it out of the decoder and keep it for the host's tail.
        self._tail_norm = decoder.norm if is_host else None
        decoder.norm = nn.Identity()
        self._decoder = decoder
        self._embed = decoder.embed_tokens if is_host else None
        self._head = model.get_output_embeddings() if is_host else None

        torch.manual_seed(settings.seed)
        cfg = LoraConfig(r=settings.rank, lora_alpha=settings.alpha, lora_dropout=settings.dropout,
                         target_modules=list(settings.target_modules), bias="none")
        self.model = get_peft_model(model, cfg)
        if settings.grad_checkpointing:
            enable = (getattr(self.model, "activation_checkpointing_enable", None)
                      or self.model.gradient_checkpointing_enable)  # name changed across versions
            enable()
        self.model.train()
        self._params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if not self._params:
            raise RuntimeError(f"no layers matched LoRA targets {settings.target_modules}")
        self.optimizer = torch.optim.AdamW(
            self._params, lr=0.0, betas=(settings.beta1, settings.beta2),
            eps=settings.eps, weight_decay=settings.weight_decay)
        self._h_in: Optional[torch.Tensor] = None
        self._h_out: Optional[torch.Tensor] = None
        # 4-bit layers store integers, so their weights do not reveal the working format.
        self._act_dtype: Optional[torch.dtype] = (
            (getattr(torch, dtype) if dtype else torch.float16) if load_4bit else None)

    # -- helpers ---------------------------------------------------------
    @property
    def _first_device(self) -> torch.device:
        return next(self._decoder.layers[0].parameters()).device

    @property
    def act_dtype(self) -> torch.dtype:
        """Number format of the hidden states this machine's layers work in."""
        if self._act_dtype is not None:
            return self._act_dtype
        for name, p in self._decoder.layers[0].named_parameters():
            if p.is_floating_point() and "lora_" not in name:
                return p.dtype
        return torch.float32

    def _run_layers(self, h: torch.Tensor) -> torch.Tensor:
        return self._decoder(inputs_embeds=h, use_cache=False).last_hidden_state

    def drop_graph(self) -> None:
        self._h_in = None
        self._h_out = None

    # -- guest: middle of the model -------------------------------------
    def forward(self, h_in: torch.Tensor, keep_graph: bool = True) -> torch.Tensor:
        h_in = h_in.to(self._first_device)
        if not keep_graph:
            with torch.no_grad():
                return self._run_layers(h_in).detach()
        h_in = h_in.detach().requires_grad_(True)
        self._h_in = h_in
        self._h_out = self._run_layers(h_in)
        return self._h_out.detach()

    def backward(self, g_out: torch.Tensor) -> torch.Tensor:
        if self._h_out is None:
            raise RuntimeError("backward called with no forward pass waiting")
        self._h_out.backward(g_out.to(self._h_out.device, self._h_out.dtype))
        g_in = self._h_in.grad
        self.drop_graph()
        return g_in

    # -- host: both ends of the model ------------------------------------
    def head_forward(self, input_ids: torch.Tensor, keep_graph: bool = True) -> torch.Tensor:
        ids = input_ids.to(self._embed.weight.device)
        if not keep_graph:
            with torch.no_grad():
                return self._run_layers(self._embed(ids)).detach()
        emb = self._embed(ids)
        if self.settings.grad_checkpointing:
            emb = emb.detach().requires_grad_(True)  # checkpointing needs an input that tracks gradients
        self._h_out = self._run_layers(emb)
        return self._h_out.detach()

    def head_backward(self, g_out: torch.Tensor) -> None:
        if self._h_out is None:
            raise RuntimeError("head_backward called with no forward pass waiting")
        self._h_out.backward(g_out.to(self._h_out.device, self._h_out.dtype))
        self.drop_graph()

    def tail_loss(self, h_last: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = self._head.weight.device
        logits = self._head(self._tail_norm(h_last.to(device)))
        return F.cross_entropy(logits[0, :-1].float(), labels[0, 1:].to(device), ignore_index=-100)

    def tail(self, h_last: torch.Tensor, labels: torch.Tensor, multiplier: float) -> Tuple[float, torch.Tensor]:
        """Loss on the last hidden states, and the gradient to send back down the chain."""
        h = h_last.to(self._head.weight.device).detach().requires_grad_(True)
        loss = self.tail_loss(h, labels)
        (loss * multiplier).backward()
        return float(loss.detach()), h.grad

    @torch.no_grad()
    def tail_eval(self, h_last: torch.Tensor, labels: torch.Tensor) -> float:
        return float(self.tail_loss(h_last, labels))

    # -- optimizer steps, coordinated by the host -------------------------
    def prepare(self, scale: float) -> Tuple[bool, float]:
        """Undo loss scaling, then report whether gradients are finite and their squared norm."""
        sq = 0.0
        finite = True
        for p in self._params:
            if p.grad is None:
                continue
            if scale != 1.0:
                p.grad.div_(scale)
            g = p.grad.float()
            if not torch.isfinite(g).all():
                finite = False
            else:
                sq += float(g.pow(2).sum())
        return finite, sq

    def apply(self, clip_coef: float, lr: float) -> None:
        if clip_coef != 1.0:
            for p in self._params:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def skip(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.drop_graph()

    # -- adapter state ----------------------------------------------------
    def adapter_state(self) -> Dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict
        return {k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(self.model).items()}

    def set_adapter_state(self, state: Dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict
        own = self.adapter_state()
        if set(own) != set(state) or any(own[k].shape != state[k].shape for k in own):
            raise RuntimeError("adapter does not match this machine's layers and LoRA settings")
        set_peft_model_state_dict(self.model, state)

    def adapter_bytes(self) -> bytes:
        return st_save(self.adapter_state())

    def reinit(self) -> None:
        """Start the adapter and optimizer over, exactly as on a fresh run."""
        torch.manual_seed(self.settings.seed)
        for module in self.model.modules():
            if hasattr(module, "reset_lora_parameters") and hasattr(module, "lora_A"):
                for name in list(module.lora_A.keys()):
                    module.reset_lora_parameters(name, True)
        self.optimizer = torch.optim.AdamW(
            self._params, lr=0.0, betas=(self.settings.beta1, self.settings.beta2),
            eps=self.settings.eps, weight_decay=self.settings.weight_decay)
        self.skip()

    # -- checkpoints ------------------------------------------------------
    @staticmethod
    def _ckpt_dir(root: str, step: int) -> str:
        return os.path.join(root, f"step-{step:08d}")

    @staticmethod
    def list_checkpoints(root: str) -> List[int]:
        steps = []
        if os.path.isdir(root):
            for name in os.listdir(root):
                if name.startswith("step-") and name[5:].isdigit() and \
                        os.path.exists(os.path.join(root, name, "done.json")):
                    steps.append(int(name[5:]))
        return sorted(steps)

    def save(self, root: str, step: int, extra: Optional[dict] = None) -> None:
        final = self._ckpt_dir(root, step)
        tmp = final + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        st_save_file(self.adapter_state(), os.path.join(tmp, "adapter.safetensors"))
        torch.save(self.optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
        with open(os.path.join(tmp, "done.json"), "w") as f:
            json.dump({"step": step, "settings": self.settings.to_dict(), "extra": extra or {}}, f)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)

    def load(self, root: str, step: int) -> dict:
        d = self._ckpt_dir(root, step)
        with open(os.path.join(d, "done.json")) as f:
            meta = json.load(f)
        if meta["settings"] != self.settings.to_dict():
            raise RuntimeError("checkpoint was made with different LoRA settings")
        self.set_adapter_state(st_load_file(os.path.join(d, "adapter.safetensors")))
        # optimizer.pt is written and read only by this machine; weights_only
        # still refuses anything but plain tensors and numbers.
        self.optimizer.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), weights_only=True))
        self.skip()
        return meta.get("extra", {})

    @classmethod
    def prune(cls, root: str, keep: int) -> None:
        for step in cls.list_checkpoints(root)[:-keep]:
            shutil.rmtree(cls._ckpt_dir(root, step), ignore_errors=True)


def adapter_from_bytes(blob: bytes) -> Dict[str, torch.Tensor]:
    return st_load(blob)
