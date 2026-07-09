"""Minimal LoRA for timm ViT (no peft dependency; OSI-clean, full control).

Wraps target nn.Linear modules with a low-rank adapter:
    y = W0 x + b0  +  scaling * (dropout(x) @ A^T) @ B^T
W0/b0 frozen; A (r x in), B (out x r) trainable. B init 0 -> identity at start.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        lora = (self.drop(x) @ self.A.t()) @ self.B.t()
        return out + self.scaling * lora


def inject_lora(model: nn.Module, targets=("qkv", "proj", "fc1", "fc2"),
                r=16, alpha=32, dropout=0.05, only_last=None):
    """Replace Linear modules whose name ends with a target suffix with LoRALinear.

    only_last: if set to int N, only inject into the last N transformer blocks
    (cheaper, often enough). Returns count injected.
    """
    blocks = getattr(model, "blocks", None)
    keep_ids = None
    if only_last is not None and blocks is not None:
        keep_ids = {id(b) for b in list(blocks)[-only_last:]}

    n = 0
    for mod_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                if keep_ids is not None:
                    # only inject if this module is inside a kept block
                    blk = _owning_block(model, mod_name)
                    if blk is None or id(blk) not in keep_ids:
                        continue
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


def _owning_block(model, mod_name):
    # mod_name like 'blocks.23.attn'; return blocks[23]
    parts = mod_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return model.blocks[int(parts[1])]
        except Exception:
            return None
    return None


def mark_trainable(model: nn.Module, head_names=("head", "classifier")):
    """Freeze everything except LoRA params and the head; return param groups info."""
    n_train = 0
    for name, p in model.named_parameters():
        is_lora = (".A" in name or ".B" in name) and ("base" not in name)
        is_head = any(h in name for h in head_names)
        p.requires_grad_(bool(is_lora or is_head))
        if p.requires_grad:
            n_train += p.numel()
    return n_train
