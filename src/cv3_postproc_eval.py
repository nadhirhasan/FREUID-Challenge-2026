"""cv3 post-processing eval on its OWN leak-free gauge: fold-0 validation (13,871 FREUID
images cv3 never trained on), both CLEAN and HARD (corrupt_fixed, the public-LB proxy).

Compares: raw (training agg, scale 1.0) | pagg(0.05,0.25) | TTA4 | pagg+TTA4.
Patch logits + attention branch are extracted per scale, so all aggregations come from
the same forward passes (exact comparison, like pagg_tta_combo.py).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, train_path, letterbox, to_tensor_norm, load_rgb
from model import FreuidModel
from freuid_metric import freuid_score
from train import corrupt_fixed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CKPT = os.path.join(ROOT, "checkpoints", "cv3_fold0_ep1_LB00096.pt")
BASE_H, BASE_W = 448, 728
SCALES = [0.85, 0.9, 1.0, 1.1]
TOPK_FRAC, W_TOPK = 0.05, 0.25


class ValDS(Dataset):
    def __init__(self, ids, labels, H, W, hard):
        self.ids = ids; self.y = np.asarray(labels, np.float32)
        self.H, self.W = H, W; self.hard = hard

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img = load_rgb(train_path(self.ids[i]))
        if self.hard:
            img = corrupt_fixed(img, i)
        return to_tensor_norm(letterbox(img, self.H, self.W)), float(self.y[i])


def main():
    df = pd.read_csv(os.path.join(ROOT, "splits", "folds.csv"))
    va = df[df.strat_fold == 0]
    ids, y = va.id.tolist(), va.label.values.astype(np.float32)
    print(f"fold0 val: {len(ids)} (gen={(y==0).sum()}, att={(y==1).sum()})", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])
    head = model.head

    results = {}   # (variant, hard) -> per-scale logits
    for hard in [False, True]:
        std_lg, pagg_lg = {}, {}
        for s in SCALES:
            H = int(BASE_H * s) // 14 * 14
            W = int(BASE_W * s) // 14 * 14
            dl = DataLoader(ValDS(ids, y, H, W, hard), batch_size=32, shuffle=False,
                            num_workers=16, pin_memory=True)
            t0 = time.time(); std, pag = [], []
            with torch.no_grad():
                for x, _ in dl:
                    x = x.cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        tok = model.backbone.forward_features(x)
                        patch = head.norm(tok[:, head.n_prefix:])
                        plog = head.scorer(patch).squeeze(-1)
                        k5 = max(1, int(TOPK_FRAC * plog.shape[1]))
                        k10 = max(1, int(0.10 * plog.shape[1]))
                        topk5 = plog.topk(k5, dim=1).values.mean(1)
                        topk10 = plog.topk(k10, dim=1).values.mean(1)
                        aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                        attnm = (plog * aw).sum(1)
                        std.append((0.5 * (topk10 + attnm)).float().cpu())
                        pag.append((W_TOPK * topk5 + (1 - W_TOPK) * attnm).float().cpu())
            std_lg[s] = torch.cat(std).numpy().astype(np.float64)
            pagg_lg[s] = torch.cat(pag).numpy().astype(np.float64)
            print(f"{'HARD' if hard else 'CLEAN'} scale {s:g} done ({time.time()-t0:.0f}s)", flush=True)
        results[("std", hard)] = std_lg
        results[("pagg", hard)] = pagg_lg

    def report(tag, lg):
        p = 1 / (1 + np.exp(-lg))
        f, a, apc = freuid_score(y, p)
        print(f"[{tag:28s}] FREUID={f:.6f} AuDET={a:.6f} APCER@1%={apc:.6f}", flush=True)

    for hard in [False, True]:
        lbl = "HARD" if hard else "CLEAN"
        std_lg = results[("std", hard)]; pagg_lg = results[("pagg", hard)]
        print(f"\n=== cv3 fold0-val {lbl} ===", flush=True)
        report("raw (train agg, 1.0)", std_lg[1.0])
        report("pagg, 1.0", pagg_lg[1.0])
        report("std + TTA4", np.mean([std_lg[s] for s in SCALES], axis=0))
        report("pagg + TTA4", np.mean([pagg_lg[s] for s in SCALES], axis=0))


if __name__ == "__main__":
    main()
