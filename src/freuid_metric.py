"""Exact FREUID Challenge metric (lower = better).

Definitions (from competition README):
  - Scores: higher => more likely FRAUD (attack, label 1). Genuine = label 0.
  - Threshold rule: predict ATTACK if score >= t.
      BPCER(t) = fraction of GENUINE (label 0) with score >= t   (bona-fide wrongly rejected)
      APCER(t) = fraction of ATTACK  (label 1) with score <  t   (attack wrongly accepted / missed)
  - DET curve: APCER (y) vs BPCER (x) as t sweeps. Both in [0,1].
  - AuDET = area under the DET curve (linear axes), bounded [0,1], lower better.
            Identity used/verified: AuDET == 1 - ROC_AUC (attack=positive).
  - APCER@1%BPCER = APCER at the threshold where BPCER == 1%.
  - g_audet = 1 - AuDET ; g_apcer = 1 - APCER@1%BPCER
  - FREUID  = 1 - 2*g_audet*g_apcer/(g_audet+g_apcer)    (lower better)
"""
from __future__ import annotations
import numpy as np


def _as_arrays(y_true, scores):
    y = np.asarray(y_true).astype(np.int64).ravel()
    s = np.asarray(scores).astype(np.float64).ravel()
    assert y.shape == s.shape, "shape mismatch"
    assert set(np.unique(y)).issubset({0, 1}), "labels must be 0/1"
    return y, s


def audet(y_true, scores) -> float:
    """Area under the Detection Error Trade-off curve (linear axes), in [0,1].

    AuDET = integral of APCER over BPCER = 1 - ROC_AUC (attack=positive).
    We use the exact ROC-AUC identity (fast, exact, robust to ties). The direct
    trapezoid sweep is in `_audet_trapz` and cross-checked against this in tests.
    """
    from sklearn.metrics import roc_auc_score
    y, s = _as_arrays(y_true, scores)
    assert (y == 0).any() and (y == 1).any(), "need both classes"
    return float(1.0 - roc_auc_score(y, s))


def _audet_trapz(y_true, scores) -> float:
    """Direct DET-area by threshold sweep (cross-check for `audet`)."""
    y, s = _as_arrays(y_true, scores)
    gen = s[y == 0]
    att = s[y == 1]
    ng, na = len(gen), len(att)
    assert ng > 0 and na > 0, "need both classes"
    # candidate thresholds = all distinct scores (+/- inf endpoints)
    thr = np.unique(s)
    thr = np.concatenate(([-np.inf], thr, [np.inf]))
    # BPCER(t)=mean(gen>=t); APCER(t)=mean(att<t). Vectorized via sorting.
    gen_sorted = np.sort(gen)
    att_sorted = np.sort(att)
    # count gen >= t  => ng - searchsorted(gen, t, 'left')
    bpcer = (ng - np.searchsorted(gen_sorted, thr, side="left")) / ng
    # count att < t   => searchsorted(att, t, 'left')
    apcer = np.searchsorted(att_sorted, thr, side="left") / na
    # sort by bpcer ascending for integration
    order = np.argsort(bpcer, kind="mergesort")
    x = bpcer[order]
    yv = apcer[order]
    area = np.trapz(yv, x)
    return float(area)


def apcer_at_bpcer(y_true, scores, bpcer_target: float = 0.01) -> float:
    """APCER at the threshold where BPCER == bpcer_target (default 1%).

    Threshold t* = (1 - bpcer_target) quantile of GENUINE scores: the value
    above which exactly `bpcer_target` of genuine lie. APCER = mean(attack < t*).
    Uses linear interpolation on the genuine score quantile for stability.
    """
    y, s = _as_arrays(y_true, scores)
    gen = np.sort(s[y == 0])
    att = s[y == 1]
    # threshold so that fraction of genuine >= t equals bpcer_target
    # => t is the (1 - bpcer_target) quantile of genuine scores.
    t = np.quantile(gen, 1.0 - bpcer_target, method="linear")
    apcer = float(np.mean(att < t))
    return apcer


def freuid_score(y_true, scores, bpcer_target: float = 0.01):
    """Return (freuid, audet, apcer_at_1pct_bpcer). FREUID lower = better."""
    a = audet(y_true, scores)
    p = apcer_at_bpcer(y_true, scores, bpcer_target)
    g_audet = 1.0 - a
    g_apcer = 1.0 - p
    denom = g_audet + g_apcer
    hm = 0.0 if denom == 0 else 2.0 * g_audet * g_apcer / denom
    freuid = 1.0 - hm
    return float(freuid), float(a), float(p)


# ---------------- self-tests ----------------
if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)

    def check(y, s, tag):
        a = audet(y, s)
        at = _audet_trapz(y, s)
        f, a2, p = freuid_score(y, s)
        print(f"[{tag}] AuDET={a:.5f}  trapz={at:.5f}  diff={abs(a-at):.2e}"
              f"  APCER@1%BPCER={p:.4f}  FREUID={f:.5f}")
        assert abs(a - at) < 2e-3, "AuDET (1-AUC) != trapz sweep"

    # 1) random scores ~ 0.5 AUC
    n = 20000
    y = rng.integers(0, 2, n)
    s = rng.random(n)
    check(y, s, "random")

    # 2) separable-ish: attacks higher
    y = rng.integers(0, 2, n)
    s = rng.normal(loc=y * 1.5, scale=1.0)
    check(y, s, "gaussian-sep")

    # 3) perfect separation
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    s = np.r_[rng.random(1000) * 0.4, 0.6 + rng.random(1000) * 0.4]
    f, a, p = freuid_score(y, s)
    print(f"[perfect] AuDET={a:.7f} APCER@1%={p:.4f} FREUID={f:.7f}")
    assert a < 1e-6 and p < 1e-6 and f < 1e-6, "perfect case should be ~0"

    # 4) monotonic-invariance: metric unchanged under monotone transform
    y = rng.integers(0, 2, n)
    s = rng.normal(loc=y * 1.2, scale=1.0)
    f1 = freuid_score(y, s)
    f2 = freuid_score(y, 1 / (1 + np.exp(-3 * s)))   # sigmoid
    f3 = freuid_score(y, s * 7.0 - 100.0)            # affine
    print("monotone-invariance:", np.round(f1, 6), np.round(f2, 6), np.round(f3, 6))
    assert np.allclose(f1, f2, atol=1e-6) and np.allclose(f1, f3, atol=1e-6)
    print("ALL METRIC TESTS PASSED")
