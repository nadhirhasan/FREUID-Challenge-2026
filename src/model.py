"""DINOv2-L (Apache-2.0) + LoRA + MAC-style head for FREUID fraud detection."""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import torch
import torch.nn as nn
import timm

from lora import inject_lora, mark_trainable


class MACHead(nn.Module):
    """Multi-Aspect pooling: concat[CLS, mean(reg), mean(patch), max(patch)] -> MLP -> 1 logit."""
    def __init__(self, dim, n_prefix, hidden=512, p=0.3):
        super().__init__()
        self.n_prefix = n_prefix
        self.norm = nn.LayerNorm(dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens):
        cls = tokens[:, 0]
        reg = tokens[:, 1:self.n_prefix].mean(1)
        patch = tokens[:, self.n_prefix:]
        feat = torch.cat([cls, reg, patch.mean(1), patch.amax(1)], dim=1)
        return self.mlp(self.norm(feat)).squeeze(1)


class PatchHead(nn.Module):
    """Per-patch forgery scoring + MIL aggregation -> image logit.

    Forces the model onto LOCAL forgery artifacts (blend seams, compression/splice
    boundaries) that are UNIVERSAL across document types/countries, instead of global
    (country-specific) layout -> generalizes to unseen IDs. Returns (image_logit, patch_logits).
    """
    def __init__(self, dim, n_prefix, hidden=512, topk_frac=0.1, p=0.3):
        super().__init__()
        self.n_prefix = n_prefix
        self.topk_frac = topk_frac
        self.norm = nn.LayerNorm(dim)
        self.scorer = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, 1))
        self.attn = nn.Linear(dim, 1)

    def forward(self, tokens):
        patch = self.norm(tokens[:, self.n_prefix:])           # B, N, D
        plog = self.scorer(patch).squeeze(-1)                  # B, N  per-patch forgery logit
        k = max(1, int(self.topk_frac * plog.shape[1]))
        topk = plog.topk(k, dim=1).values.mean(1)              # B  (most-suspicious patches)
        aw = torch.softmax(self.attn(patch).squeeze(-1), dim=1)  # B, N
        attnm = (plog * aw).sum(1)                             # B  (attention-weighted)
        img = 0.5 * (topk + attnm)
        return img, plog


class FreuidModel(nn.Module):
    def __init__(self, backbone="vit_large_patch14_reg4_dinov2", pretrained=True,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, only_last=None,
                 head_hidden=512, head_dropout=0.3, head_type="patch"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, dynamic_img_size=True)
        self.n_prefix = self.backbone.num_prefix_tokens
        dim = self.backbone.embed_dim
        n = inject_lora(self.backbone, r=lora_r, alpha=lora_alpha,
                        dropout=lora_dropout, only_last=only_last)
        self.head_type = head_type
        self.head = (PatchHead(dim, self.n_prefix, head_hidden, p=head_dropout)
                     if head_type == "patch" else MACHead(dim, self.n_prefix, head_hidden, head_dropout))
        self.n_lora = n

    def trainable_params(self):
        return mark_trainable(self)

    def forward(self, x, return_patches=False):
        tok = self.backbone.forward_features(x)
        out = self.head(tok)
        if self.head_type == "patch":
            img, plog = out
            return (img, plog) if return_patches else img
        return out


if __name__ == "__main__":
    m = FreuidModel(only_last=None).cuda()
    nt = m.trainable_params()
    tot = sum(p.numel() for p in m.parameters())
    print(f"LoRA modules injected: {m.n_lora}")
    print(f"trainable params: {nt/1e6:.2f}M / total {tot/1e6:.1f}M  ({100*nt/tot:.2f}%)")
    x = torch.randn(2, 3, 322, 518).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(x)
    print("output:", tuple(out.shape), out.detach().float().cpu().numpy())
    out.sum().backward()
    g = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
    print(f"params with grad after backward: {len(g)} (sanity: LoRA+head only)")
