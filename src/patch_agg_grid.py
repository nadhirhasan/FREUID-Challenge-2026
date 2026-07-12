"""Patch-aggregation post-processing grid for cv5_ep2 (FakeIDet2-grounded lever).

One GPU pass over the external subsample saving per-patch logits + the attention-branch
scalar, then offline sweep of image-score aggregation:
    img(w, k) = w * mean(top-k% patch logits) + (1 - w) * attn_weighted_sum
Baseline = (w=0.5, k=10%), exactly what PatchHead.forward computes in training.
Metric is rank-only, so aggregation is the ONLY head-side lever that can move it.
"""
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
H, W = 448, 728
N_SUB = 12000
EVAL_BS = 32
WORKERS = 16


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
    head = model.head

    plogs, attnms = [], []
    dl = DataLoader(IDNetEvalDS(sub, H, W), batch_size=EVAL_BS, shuffle=False,
                    num_workers=WORKERS, pin_memory=True)
    t0 = time.time()
    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tok = model.backbone.forward_features(x)
                patch = head.norm(tok[:, head.n_prefix:])
                plog = head.scorer(patch).squeeze(-1)            # B, N
                aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                attnm = (plog * aw).sum(1)                       # B
            plogs.append(plog.float().cpu()); attnms.append(attnm.float().cpu())
    plog = torch.cat(plogs).numpy(); attnm = torch.cat(attnms).numpy()
    print(f"forward done: plog {plog.shape}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(ROOT, "out", "patch_agg_raw.npz"),
                        y=y, plog=plog.astype(np.float16), attnm=attnm)

    n_patches = plog.shape[1]
    plog_sorted = np.sort(plog, axis=1)[:, ::-1]  # desc
    csum = np.cumsum(plog_sorted, axis=1)

    def topk_mean(frac):
        k = max(1, int(frac * n_patches))
        return csum[:, k - 1] / k

    rows = []
    fracs = [1.0 / n_patches, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    for frac in fracs:
        tk = topk_mean(frac)
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            img = w * tk + (1 - w) * attnm
            f, a, apc = freuid_score(y, img)
            rows.append({"topk_frac": round(frac, 4), "w_topk": w,
                         "FREUID": f, "AuDET": a, "APCER@1%": apc})
    df = pd.DataFrame(rows).sort_values("FREUID")
    base = df[(df.topk_frac == 0.10) & (df.w_topk == 0.5)]
    print("=== baseline (training aggregation, w=0.5 k=10%) ===", flush=True)
    print(base.to_string(index=False), flush=True)
    print("\n=== top 12 aggregations ===", flush=True)
    print(df.head(12).to_string(index=False), flush=True)
    df.to_csv(os.path.join(ROOT, "out", "patch_agg_results.csv"), index=False)
    print("saved out/patch_agg_results.csv", flush=True)


if __name__ == "__main__":
    main()
