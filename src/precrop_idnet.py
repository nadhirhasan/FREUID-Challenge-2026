"""Pre-crop ALL IDNet images once to tight FREUID-format cards (deskewed landscape,
no white background), saved to external/idnet_cropped/. Training then loads these
directly (fast) instead of cropping every epoch. Orientation is handled later by
augmentation (auto-upright is unreliable on stylized ID portraits).

  python src/precrop_idnet.py --workers 16
"""
from __future__ import annotations
import os, sys, argparse, cv2
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from external_data import crop_card_meta
cv2.setNumThreads(0)
SRC_IDX = os.path.join(ROOT, "external", "idnet_index.csv")
DST = os.path.join(ROOT, "external", "idnet_cropped")
OUT_IDX = os.path.join(ROOT, "external", "idnet_cropped_index.csv")
MAXW = 1100  # cap width to keep tight cards small on disk


def _one(args):
    src, rel = args
    dst = os.path.join(DST, rel)
    if os.path.exists(dst):
        return dst
    try:
        cr = crop_card_meta(src)  # JSON-deskew -> tight, upright FREUID-format card
        if cr is None or cr.size == 0:
            return None
        if cr.shape[1] > MAXW:
            s = MAXW / cr.shape[1]
            cr = cv2.resize(cr, (MAXW, max(1, int(cr.shape[0] * s))), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, cv2.cvtColor(cr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return dst
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    idx = pd.read_csv(SRC_IDX)
    # relative path under external/idnet -> mirror under idnet_cropped, as .jpg
    base = os.path.join(ROOT, "external", "idnet")
    jobs, keep = [], []
    for _, r in idx.iterrows():
        rel = os.path.relpath(r.path, base)
        rel = os.path.splitext(rel)[0] + ".jpg"
        jobs.append((r.path, rel)); keep.append((r, rel))
    print(f"pre-cropping {len(jobs)} IDNet images -> {DST} ({args.workers} procs)", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_one, jobs, chunksize=64))
    rows = []
    for (r, rel), res in zip(keep, results):
        if res is None:
            continue
        rows.append({"id": "idnetc/" + rel.replace(os.sep, "/"), "path": res,
                     "label": int(r.label), "type": r.type, "source": "idnet"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_IDX, index=False)
    print(f"cropped {len(df)}/{len(jobs)} -> {OUT_IDX}")
    print(df.groupby(["type", "label"]).size())


if __name__ == "__main__":
    main()
