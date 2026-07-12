"""Submission with re-aggregated PatchHead scoring (best config from patch_agg_grid:
img = 0.25 * mean(top-5% patch logits) + 0.75 * attention-weighted sum).
--tta_scales '' (default) = single scale 448x728, isolates the aggregation change.
--tta_scales 0.85,0.9,1.0,1.1 = combined pagg+TTA (logit-avg across scales)."""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, DATA, test_path
from model import FreuidModel
from infer import TestDS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CKPT = os.path.join(ROOT, "checkpoints", "cv5_full_ep2_IDNet0.0166.pt")
H, W = 448, 728
TOPK_FRAC, W_TOPK = 0.05, 0.25


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def predict_pass(model, head, ids, Hs, Ws):
    dl = DataLoader(TestDS(ids, Hs, Ws), batch_size=16, shuffle=False,
                    num_workers=8, pin_memory=True)
    out = np.zeros(len(ids), np.float32)
    t0 = time.time(); n = 0
    with torch.no_grad():
        for x, idx in dl:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tok = model.backbone.forward_features(x)
                patch = head.norm(tok[:, head.n_prefix:])
                plog = head.scorer(patch).squeeze(-1)                       # B, N
                k = max(1, int(TOPK_FRAC * plog.shape[1]))
                topk = plog.topk(k, dim=1).values.mean(1)
                aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                attnm = (plog * aw).sum(1)
                img = W_TOPK * topk + (1 - W_TOPK) * attnm
                p = torch.sigmoid(img).float().cpu().numpy()
            out[idx.numpy()] = p
            n += x.size(0)
            if n % 800 < 16:
                print(f"  {Hs}x{Ws}: {n}/{len(ids)}  {n/(time.time()-t0):.1f} img/s", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta_scales", type=str, default="",
                    help="comma list, e.g. 0.85,0.9,1.0,1.1; empty = single scale")
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--out", type=str,
                    default=os.path.join(ROOT, "submissions", "sub_cv5_full_ep2_pagg.csv"))
    args = ap.parse_args()
    scales = [float(s) for s in args.tta_scales.split(",") if s.strip()] or [1.0]

    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = [i for i in sub.id.tolist() if os.path.exists(test_path(i))]
    print(f"predicting {len(ids)} local public_test images (of {len(sub)} rows) "
          f"scales={scales}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])
    head = model.head

    lg = np.zeros(len(ids), np.float64)
    for s in scales:
        Hs, Ws = int(H * s) // 14 * 14, int(W * s) // 14 * 14
        lg += to_logit(predict_pass(model, head, ids, Hs, Ws))
    out = (1 / (1 + np.exp(-lg / len(scales)))).astype(np.float32)

    sub["label"] = sub.id.map(dict(zip(ids, out))).fillna(0.5)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  filled={len(ids)}  "
          f"min/mean/max={out.min():.3f}/{out.mean():.3f}/{out.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
