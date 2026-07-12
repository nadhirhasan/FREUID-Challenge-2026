"""Combined patch-agg (top5%, w=0.25) x 4-scale TTA evaluation on the external subsample.
Also re-scores each lever alone on the same subsample for a clean 2x2 comparison."""
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
TOPK_FRAC, W_TOPK = 0.05, 0.25
N_SUB = 12000


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    pool = idn[idn.type.isin(["EST_scanned", "SVK_scanned"])]
    sub = pool.groupby(["type", "label"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), N_SUB // 4), random_state=7)).reset_index(drop=True)
    y = sub.label.values.astype(np.float32)
    print(f"subsample: {len(sub)}", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])
    head = model.head

    # per-scale: standard image logit AND patch-agg logit, from the same forward pass
    std_lg, pagg_lg = {}, {}
    for s in SCALES:
        H = int(BASE_H * s) // 14 * 14
        W = int(BASE_W * s) // 14 * 14
        dl = DataLoader(IDNetEvalDS(sub, H, W), batch_size=32, shuffle=False,
                        num_workers=16, pin_memory=True)
        t0 = time.time(); std, pag = [], []
        with torch.no_grad():
            for x, _ in dl:
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = model.backbone.forward_features(x)
                    patch = head.norm(tok[:, head.n_prefix:])
                    plog = head.scorer(patch).squeeze(-1)
                    k = max(1, int(TOPK_FRAC * plog.shape[1]))
                    topk5 = plog.topk(k, dim=1).values.mean(1)
                    k10 = max(1, int(0.10 * plog.shape[1]))
                    topk10 = plog.topk(k10, dim=1).values.mean(1)
                    aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                    attnm = (plog * aw).sum(1)
                    std.append((0.5 * (topk10 + attnm)).float().cpu())      # training agg
                    pag.append((W_TOPK * topk5 + (1 - W_TOPK) * attnm).float().cpu())
        std_lg[s] = torch.cat(std).numpy().astype(np.float64)
        pagg_lg[s] = torch.cat(pag).numpy().astype(np.float64)
        print(f"scale {s:g} done ({time.time()-t0:.0f}s)", flush=True)

    def score(tag, lg):
        p = 1 / (1 + np.exp(-lg))
        f, a, apc = freuid_score(y, p)
        print(f"[{tag:24s}] FREUID={f:.4f} AuDET={a:.4f} APCER@1%={apc:.4f}", flush=True)

    print("\n=== 2x2 comparison (external subsample) ===", flush=True)
    score("std agg, scale 1.0", std_lg[1.0])
    score("pagg,    scale 1.0", pagg_lg[1.0])
    score("std agg + TTA4", np.mean([std_lg[s] for s in SCALES], axis=0))
    score("pagg    + TTA4", np.mean([pagg_lg[s] for s in SCALES], axis=0))


if __name__ == "__main__":
    main()
