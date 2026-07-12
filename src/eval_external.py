"""Unified external-generalization check: evaluate one or more checkpoints on the SAME
IDNet slice (full EST_scanned + SVK_scanned pool, both labels) using the exact competition
freuid_score() metric. Apples-to-apples across candidates trained with different recipes.

Caveat printed per-model: some candidates (e.g. cv5/cv6, --idnet_countries including
EST/SVK) may have partially trained on this slice -- their number here is optimistic,
not a blind test. Only candidates that never touched EST/SVK (cv3, cv4) get a clean read.

Usage:
  python src/eval_external.py --ckpt checkpoints/cv3_fold0_ep1_LB00096.pt:cv3,checkpoints/cv4_fold0_ep2_IDNet0.0933.pt:cv4,checkpoints/cv5_full_ep2_IDNet0.0166.pt:cv5_ep2
"""
from __future__ import annotations
import os, sys, argparse, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from model import FreuidModel
from freuid_metric import freuid_score
from train import IDNetEvalDS  # reuse exact same dataset class used during training eval

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                     help="comma-separated path:label pairs, e.g. a.pt:cv3,b.pt:cv4")
    ap.add_argument("--countries", type=str, default="EST_scanned,SVK_scanned")
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
    countries = args.countries.split(",")
    pool = idn[idn.type.isin(countries)].reset_index(drop=True)
    n_gen = (pool.label == 0).sum(); n_att = (pool.label == 1).sum()
    print(f"external eval pool: {countries} -> {len(pool)} images "
          f"(genuine={n_gen}, attack={n_att})", flush=True)

    results = []
    for spec in args.ckpt.split(","):
        spec = spec.strip()
        if not spec:
            continue
        path, _, tag = spec.partition(":")
        tag = tag or os.path.basename(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cargs = ck.get("args", {})
        res = cargs.get("res", "448x728")
        H, W = (int(v) for v in res.lower().split("x"))
        lora_r = cargs.get("lora_r", 16)
        model = FreuidModel(pretrained=False, lora_r=lora_r).to(device).eval()
        model.load_state_dict(ck["model"])

        dl = DataLoader(IDNetEvalDS(pool, H, W), batch_size=args.eval_bs, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
        t0 = time.time(); ps, ys = [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.sigmoid(model(x)).float().cpu()
                ps.append(p); ys.append(y)
        p = torch.cat(ps).numpy(); y = torch.cat(ys).numpy()
        f, a, apc = freuid_score(y, p)
        dt = time.time() - t0
        print(f"[{tag}] n={len(pool)} FREUID={f:.4f} AuDET={a:.4f} "
              f"APCER@1%BPCER={apc:.4f} ({dt:.0f}s)", flush=True)
        results.append({"tag": tag, "ckpt": path, "n": len(pool),
                         "FREUID": f, "AuDET": a, "APCER@1%BPCER": apc})
        del model
        torch.cuda.empty_cache()

    print("\n=== summary (lower FREUID = better) ===")
    df = pd.DataFrame(results).sort_values("FREUID")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
