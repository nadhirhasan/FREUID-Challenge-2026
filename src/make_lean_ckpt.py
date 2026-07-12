"""Convert a full training checkpoint (backbone + LoRA + head, ~1.24GB) into a lean
checkpoint (trained parameters only: 192 LoRA A/B matrices + 8 head tensors, ~27MB).

The backbone is frozen during training, so its tensors are bit-identical to the timm
DINOv2-L release; the lean file therefore carries every byte of trained information.
Run with --verify to prove both properties on your machine:
  1) every backbone tensor in the full checkpoint equals the timm release exactly;
  2) a lean-loaded model produces bit-identical fp32 logits to the full checkpoint.

Usage:
  python src/make_lean_ckpt.py --full checkpoints/cv3_fold0.pt --out weights/cv3_fold0.pt --verify
"""
from __future__ import annotations
import argparse, os, sys

import torch
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FreuidModel

BACKBONE = "vit_large_patch14_reg4_dinov2"


def backbone_key_set() -> set[str]:
    ref = timm.create_model(BACKBONE, pretrained=True, num_classes=0, dynamic_img_size=True)
    keys = set()
    for k in ref.state_dict():
        keys.add("backbone." + k)
        parts = k.rsplit(".", 1)
        if len(parts) == 2:  # LoRA-wrapped linears keep the frozen base weight under .base.
            keys.add("backbone." + parts[0] + ".base." + parts[1])
    return keys, ref.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="full training checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="lean checkpoint to write")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.full, map_location="cpu", weights_only=False)
    bkeys, ref_sd = backbone_key_set()

    if args.verify:
        n_eq = n_cmp = 0
        sd = ck["model"]
        for k, v in ref_sd.items():
            cand = "backbone." + k
            if cand not in sd:
                parts = k.rsplit(".", 1)
                cand = "backbone." + parts[0] + ".base." + parts[1] if len(parts) == 2 else cand
            if cand in sd:
                n_cmp += 1
                n_eq += int(torch.equal(sd[cand], v))
        print(f"backbone check: {n_eq}/{n_cmp} tensors bit-identical to timm release")
        assert n_eq == n_cmp == len(ref_sd), "backbone differs from timm release; lean unsafe"

    lean = {k: v for k, v in ck["model"].items() if k not in bkeys}
    torch.save({"model_lean": lean, "args": ck.get("args"), "val": ck.get("val")}, args.out)
    print(f"wrote {args.out}: {len(lean)} tensors, {os.path.getsize(args.out)/1e6:.1f} MB")

    if args.verify and torch.cuda.is_available():
        lora_r = (ck.get("args") or {}).get("lora_r", 16)
        mf = FreuidModel(pretrained=False, lora_r=lora_r).cuda().eval()
        mf.load_state_dict(ck["model"])
        ml = FreuidModel(pretrained=True, lora_r=lora_r).cuda().eval()
        missing, unexpected = ml.load_state_dict(lean, strict=False)
        assert not unexpected, unexpected[:3]
        x = torch.randn(2, 3, 448, 728).cuda()
        with torch.no_grad():
            d = (mf(x).float() - ml(x).float()).abs().max().item()
        print(f"logit check on random input: max|diff| = {d:.2e}")
        assert d == 0.0, "lean-loaded model diverges from full checkpoint"


if __name__ == "__main__":
    main()
