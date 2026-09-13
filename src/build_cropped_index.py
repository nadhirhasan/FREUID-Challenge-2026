"""Rebuild external/idnet_cropped_index.csv DIRECTLY from the cropped image folder,
deriving the genuine/fraud label from the path ('fraud' -> 1, 'positive' -> 0) and the
type from the top-level country folder. This makes the label index reproducible from the
CROPPED data alone, so the bulky raw archives/extractions can be safely deleted without
ever losing the genuine/fake labels.

  python src/build_cropped_index.py
"""
from __future__ import annotations
import os, sys, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT

CROP = os.path.join(ROOT, "external", "idnet_cropped")
OUT = os.path.join(ROOT, "external", "idnet_cropped_index.csv")


def main():
    rows = []
    for p in glob.glob(os.path.join(CROP, "**", "*.jpg"), recursive=True):
        rel = os.path.relpath(p, CROP)
        low = p.lower()
        if "positive" in low:
            label = 0
        elif "fraud" in low:
            label = 1
        else:
            continue  # skip anything that isn't clearly genuine/fraud
        ctype = rel.replace("\\", "/").split("/")[0]   # e.g. ESP_scanned
        rows.append({"id": "idnetc/" + rel.replace(os.sep, "/"), "path": p,
                     "label": label, "type": ctype, "source": "idnet"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"rebuilt {len(df)} rows -> {OUT}")
    if len(df):
        print(df.groupby(["type", "label"]).size())


if __name__ == "__main__":
    main()
