"""Inference: load a trained FreuidModel checkpoint, predict public_test, write submission.

Usage:
  python src/infer.py --ckpt checkpoints/v1_hold_MAURITIUS_ID.pt --out submissions/sub_v1.csv
Optional multi-scale TTA (no flips: ID layout isn't flip-invariant).
"""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, DATA, test_path, letterbox, to_tensor_norm, load_rgb
from model import FreuidModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class TestDS(Dataset):
    def __init__(self, ids, H, W):
        self.ids = ids; self.H, self.W = H, W

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img = letterbox(load_rgb(test_path(self.ids[i])), self.H, self.W)
        return to_tensor_norm(img), i


@torch.no_grad()
def predict(model, ids, H, W, bs, workers, device):
    import time
    dl = DataLoader(TestDS(ids, H, W), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=(device == "cuda"))
    out = np.zeros(len(ids), np.float32)
    amp_dt = torch.float16 if device == "cuda" else torch.bfloat16
    t0 = time.time(); n = 0
    for x, idx in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device, dtype=amp_dt):
            p = torch.sigmoid(model(x)).float().cpu().numpy()
        out[idx.numpy()] = p
        n += x.size(0)
        if n % (bs * 10) < bs:
            print(f"  {n}/{len(ids)}  {n/(time.time()-t0):.1f} img/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tta", type=str, default="none", help="none | scale (multi-resolution avg)")
    ap.add_argument("--tta_scales", type=str, default="",
                    help="comma list of scale factors for multi-scale logit-avg TTA, e.g. 0.85,0.9,1.0,1.1 "
                         "(overrides --tta; scales are relative to the checkpoint's training res, "
                         "rounded down to /14 for the ViT patch grid)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    args = ap.parse_args()
    device = args.device
    if device == "cpu" and args.threads:
        torch.set_num_threads(args.threads)

    def to_logit(p):
        eps = 1e-6; p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = [i for i in sub.id.tolist() if os.path.exists(test_path(i))]
    print(f"predicting {len(ids)} local public_test images (of {len(sub)} rows)")

    ckpts = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    logit_sum = np.zeros(len(ids), np.float64)
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        cargs = ck.get("args", {})
        res = cargs.get("res", "322x518"); H, W = (int(v) for v in res.lower().split("x"))
        lora_r = cargs.get("lora_r", 16)
        print(f"-- {os.path.basename(c)} res={H}x{W} val={ck.get('val')}")
        if "model_lean" in ck:   # LoRA+head only -> rebuild DINOv2 from timm (pretrained)
            model = FreuidModel(pretrained=True, lora_r=lora_r).to(device).eval()
            missing, unexpected = model.load_state_dict(ck["model_lean"], strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:3]}"
        else:                    # full checkpoint (backbone included)
            model = FreuidModel(pretrained=False, lora_r=lora_r).to(device).eval()
            model.load_state_dict(ck["model"])
        if args.tta_scales:
            scales = [float(s) for s in args.tta_scales.split(",") if s.strip()]
            lg = np.zeros(len(ids), np.float64)
            for s in scales:
                Hs, Ws = int(H * s) // 14 * 14, int(W * s) // 14 * 14
                bs_s = args.eval_bs if s <= 1.0 else max(8, args.eval_bs // 2)
                print(f"   TTA scale {s:g} -> {Hs}x{Ws}")
                lg += to_logit(predict(model, ids, Hs, Ws, bs_s, args.workers, device))
            p = 1 / (1 + np.exp(-lg / len(scales)))
            logit_sum += to_logit(p)
            del model
            continue
        p = predict(model, ids, H, W, args.eval_bs, args.workers, device)
        if args.tta == "scale":
            H2, W2 = int(H * 1.2) // 14 * 14, int(W * 1.2) // 14 * 14
            print(f"   TTA scale {H2}x{W2}")
            p2 = predict(model, ids, H2, W2, max(8, args.eval_bs // 2), args.workers, device)
            p = 1 / (1 + np.exp(-(to_logit(p) + to_logit(p2)) / 2))
        logit_sum += to_logit(p)
        del model
    p = 1 / (1 + np.exp(-logit_sum / len(ckpts)))  # logit-space ensemble average

    sub["label"] = sub.id.map(dict(zip(ids, p))).fillna(0.5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  models={len(ckpts)}  filled={len(ids)}  "
          f"min/mean/max={p.min():.3f}/{p.mean():.3f}/{p.max():.3f}")


if __name__ == "__main__":
    main()
