"""Create reproducible CV splits for FREUID.

Outputs splits/folds.csv with columns:
  id, label, is_digital, type, strat_fold
Validation regimes used downstream:
  - STRATIFIED K-fold (strat_fold 0..K-1): balanced by type x label; for in-domain dev.
  - LEAVE-ONE-DOC-TYPE-OUT: derived from `type` at train time (train on 4 types,
    validate on the 5th) -> the north-star estimate of the unseen-type gap that
    dominates the private LB (private test has 7 types vs our 5).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "the-freuid-challenge-dataset")
OUT = os.path.join(ROOT, "splits")
os.makedirs(OUT, exist_ok=True)
SEED = 42
K = 5


def main():
    df = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    df = df[["id", "label", "is_digital", "type"]].copy()
    # combined stratification key: type x label (keeps both balanced per fold)
    strat_key = df["type"].astype(str) + "|" + df["label"].astype(str)
    df["strat_fold"] = -1
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(skf.split(df, strat_key)):
        df.iloc[val_idx, df.columns.get_loc("strat_fold")] = fold

    out_path = os.path.join(OUT, "folds.csv")
    df.to_csv(out_path, index=False)

    # report
    print("wrote", out_path, "rows", len(df))
    print("\n=== stratified fold sizes ===")
    print(df.groupby("strat_fold").size())
    print("\n=== label balance per stratified fold ===")
    print(pd.crosstab(df.strat_fold, df.label, normalize="index").round(4))
    print("\n=== type balance per stratified fold (counts) ===")
    print(pd.crosstab(df.strat_fold, df.type))
    print("\n=== Leave-one-type-out: held-out set sizes & fraud rate ===")
    for t in sorted(df.type.unique()):
        sub = df[df.type == t]
        print(f"  hold {t:14s}: n={len(sub):6d}  fraud_rate={sub.label.mean():.3f}")
    print("\nanalog (is_digital=False) count:", int((~df.is_digital).sum()))


if __name__ == "__main__":
    main()
