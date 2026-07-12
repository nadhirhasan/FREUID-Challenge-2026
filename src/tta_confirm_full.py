"""Confirm the winning TTA config (0.85+0.9+1.0+1.1 logit-avg) for cv5_ep2 on the FULL
EST+SVK external pool (35,874 images), against the single-scale baseline (0.0154)."""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from model import FreuidModel
from freuid_metric import freuid_score
from train import IDNetEvalDS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CKPT = os.path.join(ROOT, "checkpoints", "cv5_full_ep2_IDNet0.0166.pt")
BASE_H, BASE_W = 448, 728
SCALES = [0.85, 0.9, 1.0, 1.1]


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    pool = idn[idn.type.isin(["EST_scanned", "SVK_scanned"])].reset_index(drop=True)
    y = pool.label.values.astype(np.float32)
    print(f"full pool: {len(pool)} (gen={(y==0).sum()}, att={(y==1).sum()})", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])

    lg = np.zeros(len(pool), np.float64)
    for s in SCALES:
        H = int(BASE_H * s) // 14 * 14
        W = int(BASE_W * s) // 14 * 14
        dl = DataLoader(IDNetEvalDS(pool, H, W), batch_size=32, shuffle=False,
                        num_workers=16, pin_memory=True)
        t0 = time.time(); ps = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    ps.append(torch.sigmoid(model(x)).float().cpu())
        p = torch.cat(ps).numpy()
        f, a, apc = freuid_score(y, p)
        print(f"[scale {s:g} -> {H}x{W}] FREUID={f:.4f} AuDET={a:.4f} APCER@1%={apc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        lg += to_logit(p)
    p = 1 / (1 + np.exp(-lg / len(SCALES)))
    f, a, apc = freuid_score(y, p)
    print(f"[TTA {'+'.join(str(s) for s in SCALES)}] FULL-POOL FREUID={f:.4f} "
          f"AuDET={a:.4f} APCER@1%={apc:.4f}  (baseline single-scale was 0.0154)", flush=True)


if __name__ == "__main__":
    main()
