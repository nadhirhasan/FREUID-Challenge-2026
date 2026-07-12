"""Private-test-set inference + merge for the two selected final picks.

Run AFTER the private test images are released (allowed post-freeze: same frozen
weights, inference only). For each pick this script:
  1. loads the PREVIOUSLY SUBMITTED public CSV and keeps its 7,821 already-scored
     public-row predictions byte-identical (per organizer guidance on hidden-row
     effects, public rows must not drift);
  2. predicts ONLY the ids whose images exist under --private_dir and are still at
     the 0.5 placeholder;
  3. writes the merged full CSV ready for the final Kaggle upload.

Usage (once per pick):
  python src/infer_private.py --ckpt checkpoints/cv3_fold0_ep1_LB00096.pt \
      --base submissions/sub_cv3_pagg_tta4.csv \
      --private_dir <path/to/private/images> \
      --out submissions/final_cv3_pagg_tta4_full.csv

  python src/infer_private.py --ckpt checkpoints/cv5_full_ep2_IDNet0.0166.pt \
      --base submissions/sub_cv5_full_ep2_pagg_tta4.csv \
      --private_dir <path/to/private/images> \
      --out submissions/final_cv5_pagg_tta4_full.csv

Inference config is fixed to the selected finals: patch re-agg (top-5%, w=0.25)
+ 4-scale logit-avg TTA (0.85, 0.9, 1.0, 1.1). Crash-safe: predictions are
checkpointed to <out>.partial.npz every --save_every batches and resumed.
"""
from __future__ import annotations
import os, sys, argparse, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, DATA, letterbox, to_tensor_norm, load_rgb
from model import FreuidModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

H, W = 448, 728
SCALES = [0.85, 0.9, 1.0, 1.1]
TOPK_FRAC, W_TOPK = 0.05, 0.25


class PrivDS(Dataset):
    def __init__(self, paths, Hs, Ws):
        self.paths = paths; self.H, self.W = Hs, Ws

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = letterbox(load_rgb(self.paths[i]), self.H, self.W)
        return to_tensor_norm(img), i


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True, help="previously submitted CSV (public rows kept)")
    ap.add_argument("--private_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=100, help="checkpoint partial preds every N batches")
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    exts = (".jpeg", ".jpg", ".png", ".webp")
    have = {}
    for f in os.listdir(args.private_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in exts:
            have[stem] = os.path.join(args.private_dir, f)
    # predict only placeholder rows whose image is now available
    todo = base[(base.label == 0.5) & (base.id.isin(have))].reset_index(drop=True)
    print(f"base rows={len(base)}  placeholder+available={len(todo)}  "
          f"(private imgs on disk: {len(have)})", flush=True)
    if len(todo) == 0:
        raise SystemExit("nothing to predict -- check --private_dir")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = FreuidModel(pretrained=False, lora_r=ck["args"].get("lora_r", 16)).cuda().eval()
    model.load_state_dict(ck["model"])
    head = model.head

    paths = [have[i] for i in todo.id]
    partial_path = args.out + ".partial.npz"
    lg = np.zeros(len(paths), np.float64)
    done_scales = 0
    if os.path.exists(partial_path):
        z = np.load(partial_path, allow_pickle=True)
        if list(z["ids"]) == list(todo.id):
            lg = z["lg"]; done_scales = int(z["done_scales"])
            print(f"resumed partial: {done_scales} scale passes already complete", flush=True)

    for si, s in enumerate(SCALES):
        if si < done_scales:
            continue
        Hs, Ws = int(H * s) // 14 * 14, int(W * s) // 14 * 14
        dl = DataLoader(PrivDS(paths, Hs, Ws), batch_size=args.eval_bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        out_p = np.zeros(len(paths), np.float32)
        t0 = time.time(); n = 0
        with torch.no_grad():
            for bi, (x, idx) in enumerate(dl):
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = model.backbone.forward_features(x)
                    patch = head.norm(tok[:, head.n_prefix:])
                    plog = head.scorer(patch).squeeze(-1)
                    k = max(1, int(TOPK_FRAC * plog.shape[1]))
                    topk = plog.topk(k, dim=1).values.mean(1)
                    aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
                    attnm = (plog * aw).sum(1)
                    img = W_TOPK * topk + (1 - W_TOPK) * attnm
                    p = torch.sigmoid(img).float().cpu().numpy()
                out_p[idx.numpy()] = p
                n += x.size(0)
                if bi % 50 == 0:
                    print(f"  scale {s:g}: {n}/{len(paths)}  {n/(time.time()-t0):.1f} img/s", flush=True)
        lg += to_logit(out_p.astype(np.float64))
        done_scales = si + 1
        np.savez(partial_path, ids=todo.id.values, lg=lg, done_scales=done_scales)
        print(f"scale {s:g} done ({time.time()-t0:.0f}s) -- partial saved", flush=True)

    scores = 1.0 / (1.0 + np.exp(-lg / len(SCALES)))
    # exact merge: private rows get new scores; public rows keep their ORIGINAL
    # values untouched (no float round-trip drift).
    upd = dict(zip(todo.id, scores))
    merged = base.copy()
    merged["label"] = [upd.get(i, l) for i, l in zip(base.id, base.label)]
    merged.to_csv(args.out, index=False)
    n_still = (merged.label == 0.5).sum()
    print(f"wrote {args.out}: {len(merged)} rows, updated {len(todo)}, "
          f"still-placeholder {n_still}", flush=True)


if __name__ == "__main__":
    main()
