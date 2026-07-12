"""TTA grid for cv5_ep2 on a stratified external subsample (EST+SVK).
Predicts once per scale, then scores every logit-avg combination offline.
Research basis: DeepID'25 winners avoided resize TTA (artifact destruction);
generic forgery comps report multi-scale gains -- so we measure, not assume.
"""
from __future__ import annotations
import os, sys, time, itertools
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
SCALES = [0.85, 0.9, 1.0, 1.1, 1.2]
N_SUB = 12000
EVAL_BS = 32
WORKERS = 16


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    pool = idn[idn.type.isin(["EST_scanned", "SVK_scanned"])]
    sub = pool.groupby(["type", "label"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), N_SUB // 4), random_state=7)).reset_index(drop=True)
    y = sub.label.values.astype(np.float32)
    print(f"subsample: {len(sub)} (gen={(y==0).sum()}, att={(y==1).sum()})", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])

    preds = {}
    for s in SCALES:
        H = int(BASE_H * s) // 14 * 14
        W = int(BASE_W * s) // 14 * 14
        dl = DataLoader(IDNetEvalDS(sub, H, W), batch_size=EVAL_BS, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)
        t0 = time.time(); ps = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    ps.append(torch.sigmoid(model(x)).float().cpu())
        p = torch.cat(ps).numpy()
        preds[s] = p
        f, a, apc = freuid_score(y, p)
        print(f"[scale {s:.2f} -> {H}x{W}] FREUID={f:.4f} AuDET={a:.4f} APCER@1%={apc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(ROOT, "out", "tta_grid_preds.npz"),
             y=y, **{f"s{s}": p for s, p in preds.items()})

    print("\n=== combinations (logit-avg) ===", flush=True)
    rows = []
    combos = []
    for r in range(1, len(SCALES) + 1):
        combos += list(itertools.combinations(SCALES, r))
    for c in combos:
        lg = np.mean([to_logit(preds[s]) for s in c], axis=0)
        p = 1 / (1 + np.exp(-lg))
        f, a, apc = freuid_score(y, p)
        rows.append({"scales": "+".join(f"{s:g}" for s in c), "FREUID": f,
                     "AuDET": a, "APCER@1%": apc})
    df = pd.DataFrame(rows).sort_values("FREUID")
    print(df.to_string(index=False), flush=True)
    df.to_csv(os.path.join(ROOT, "out", "tta_grid_results.csv"), index=False)
    print("saved out/tta_grid_results.csv", flush=True)


if __name__ == "__main__":
    main()
