"""Generate all figures for the README and technical report:
  1. annotation_example.png  - manual face/field boxes drawn on a real MAURITIUS/ID
     sample, proving type_fields.json is geometric metadata (never a fraud/genuine
     label), directly addressing the "no manual labeling" audit question.
  2. score_distribution.png  - public vs private score histograms for cv5 (the
     winning model) and cv3 (the contrast case), showing the shift discussed in
     the report.
  3. architecture.png        - pipeline diagram: image -> DINOv2-L+LoRA -> per-patch
     MIL head -> pagg+TTA4 -> fraud score.
  4. leaderboard_result.png  - bar chart of public vs private scores for both
     picks, the single clearest visual of the whole story.
"""
import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = r"c:\Users\pc\Jupyter_Notebooks\Competitions\The FREUID Challenge"
V1 = ROOT + r"\V1"
OUT = ROOT + r"\freuid-submission\report\figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------------------
# 1. Annotation example
# ---------------------------------------------------------------------------
ann = json.load(open(ROOT + r"\annotations\type_fields.json"))
img_path = V1 + r"\the-freuid-challenge-dataset\train\train\000447addd8a4e4aabb62238ba3d559f.jpeg"
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
spec = ann["MAURITIUS/ID"]
vis = img.copy()
for (x, y, bw, bh) in spec["fields"]:
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + bw) * w), int((y + bh) * h)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 165, 0), 4)
for (x, y, bw, bh) in spec["faces"]:
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + bw) * w), int((y + bh) * h)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 170, 255), 4)

fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(vis)
ax.axis("off")
ax.set_title("annotations/type_fields.json — geometric template metadata\n"
              "(where a face/text field SITS on this document type — not a fraud/genuine label)",
              fontsize=12, pad=10)
orange_patch = mpatches.Patch(color=(1, 0.647, 0), label="text-field box (for erase-retype / text-edit attack placement)")
blue_patch = mpatches.Patch(color=(0, 0.667, 1), label="face box (for portrait-swap attack placement)")
ax.legend(handles=[orange_patch, blue_patch], loc="upper center",
          bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/annotation_example.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote annotation_example.png")

# ---------------------------------------------------------------------------
# 2. Score distribution: public vs private, cv5 (winner) and cv3 (contrast)
# ---------------------------------------------------------------------------
pairs = [("cv5 (Pick 2 — 1st place, private 0.058)",
          V1 + r"\submissions\sub_cv5_full_ep2_pagg_tta4.csv",
          V1 + r"\submissions\final_cv5_pagg_tta4_full.csv", "#2266cc"),
         ("cv3 (Pick 1 — collapsed to 0.23 private)",
          V1 + r"\submissions\sub_cv3_pagg_tta4.csv",
          V1 + r"\submissions\final_cv3_pagg_tta4_full.csv", "#cc4422")]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (name, presub, final, color) in zip(axes, pairs):
    pre = pd.read_csv(presub)
    pub_ids = set(pre.id[pre.label != 0.5])
    d = pd.read_csv(final)
    pub = d.label[d.id.isin(pub_ids)].values
    priv = d.label[~d.id.isin(pub_ids)].values
    bins = np.linspace(0, 1, 41)
    ax.hist(pub, bins=bins, alpha=0.55, label=f"public (n={len(pub):,})", color=color, density=True)
    ax.hist(priv, bins=bins, alpha=0.85, label=f"private (n={len(priv):,})", color=color,
             density=True, histtype="step", linewidth=2.2)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("predicted fraud score")
    ax.legend(fontsize=9)
axes[0].set_ylabel("density")
plt.suptitle("Public vs private score distributions — both picks shift together\n"
             "(evidence the private set genuinely differs, not that one model overfit)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/score_distribution.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote score_distribution.png")

# ---------------------------------------------------------------------------
# 3. Architecture / pipeline diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.axis("off")
boxes = [
    ("Input\ndocument\nimage", "#e8e8e8"),
    ("Letterbox\n448x728", "#e8e8e8"),
    ("DINOv2-L ViT\n(frozen)\n+ LoRA r=16", "#cfe3ff"),
    ("Per-patch MIL head\n1,664 patch logits\n+ attention branch", "#cfe3ff"),
    ("Patch re-agg\ntop-5% + attn\n(w=0.25/0.75)", "#ffe3b3"),
    ("4-scale TTA\n0.85/0.9/1.0/1.1\nlogit-avg", "#ffe3b3"),
    ("Fraud score\nsigma(logit)\n in [0,1]", "#c9f2c9"),
]
n = len(boxes)
bw_, bh_ = 1.55, 1.5
gap = 0.35
x = 0.2
for i, (label, color) in enumerate(boxes):
    rect = mpatches.FancyBboxPatch((x, 0.4), bw_, bh_, boxstyle="round,pad=0.05,rounding_size=0.08",
                                    linewidth=1.4, edgecolor="#333333", facecolor=color)
    ax.add_patch(rect)
    ax.text(x + bw_ / 2, 0.4 + bh_ / 2, label, ha="center", va="center", fontsize=9.3)
    if i < n - 1:
        ax.annotate("", xy=(x + bw_ + gap, 0.4 + bh_ / 2), xytext=(x + bw_, 0.4 + bh_ / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333333"))
    x += bw_ + gap
ax.set_xlim(0, x)
ax.set_ylim(0, 2.4)
ax.set_title("cv5 inference pipeline (winning submission) — trainable parameters only in the blue boxes (~7M of ~304M)",
             fontsize=11.5, pad=6)
plt.tight_layout()
plt.savefig(f"{OUT}/architecture.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote architecture.png")

# ---------------------------------------------------------------------------
# 4. Public vs private leaderboard result bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ["cv3\n(Pick 1)", "cv5\n(Pick 2 — WINNER)"]
public = [0.00060, 0.00191]
private = [0.23, 0.058]
x = np.arange(2)
width = 0.32
b1 = ax.bar(x - width/2, public, width, label="Public LB", color="#8fb8de")
b2 = ax.bar(x + width/2, private, width, label="Private LB (final)", color=["#cc4422", "#1a7a1a"])
ax.set_yscale("log")
ax.set_ylabel("FREUID score (log scale, lower = better)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_title("The leaderboard shake, in one picture:\npublic ranked cv3 first; private ranked cv5 first — by a wide margin",
             fontsize=12)
for bars, vals in [(b1, public), (b2, private)]:
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.5f}" if v < 0.01 else f"{v:.3f}", (bar.get_x() + bar.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.25, which="both")
plt.tight_layout()
plt.savefig(f"{OUT}/leaderboard_result.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote leaderboard_result.png")

print("\nall figures written to", OUT)
