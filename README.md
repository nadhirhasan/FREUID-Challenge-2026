# FREUID Challenge 2026 — Reproducibility Repository

Team: **nadhir hasan** (Nadhir Hasan) · Kaggle username: **nadhirhasan**

This repository contains the training code, inference code, and a runnable Docker
submission entrypoint for our FREUID Challenge 2026 solution. Two finalist
checkpoints are included (both selected as final Kaggle submissions); the Docker
image reproduces either one. See [`report/`](report/) for the full technical report
(method, data, results).

## The two selected final submissions

| Pick | Model | Training data | Public LB | Kaggle CSV (sha256) |
|---|---|---|---|---|
| 1 (**primary**) | `cv3` | FREUID only (fold-0 split, ~80% of train) | **0.00060** | `final_cv3_pagg_tta4_full.csv` — `d31f9b0163da7b3aa374b4c92cc2781b47650992c5655afcc46d381492c06048` |
| 2 | `cv5_ep2` | 100% FREUID + 80k IDNet images (all 10 countries) | 0.00191 | `final_cv5_pagg_tta4_full.csv` — `c757f5ce81388fcd2796170387d2a75b4c87b96b383c617aa8aea72f2d9e0c5a` |

Both picks use **identical inference-time options** — patch re-aggregation
(top-5% of patch logits, branch weight 0.25) plus 4-scale logit-averaged TTA
(0.85/0.9/1.0/1.1) — and come from the **same frozen commit**; they differ *only*
in the weights file, selected by one documented environment variable
(organizer-confirmed as inference orchestration under the code-freeze rules).
Checksums are of the exact final CSVs uploaded to Kaggle (public-row predictions
frozen since the pre-code-freeze probes, private rows predicted post-release with
the frozen weights via `src/infer_private.py`); bit-exact float reproduction
across different GPU hardware is not guaranteed (rank-identical scores are).

Rationale (details in the report): `cv3` is the strongest in-domain model — best
public score and near-perfect on a genuinely held-out, corruption-augmented 20%
FREUID slice (FREUID metric 0.00019). `cv5_ep2` is the robustness hedge — nearly
identical in-domain ranking (Pearson 0.994 vs `cv3` on public test) but additionally
scores 0.0116 (with TTA) on 35,874 real-world IDNet documents where FREUID-only
models are at chance level; it covers the test set's physical-capture and
unseen-document-type tail. The patch re-aggregation and TTA configurations were
selected by grid search validated on held-out external (IDNet) data, then confirmed
with public-LB probes (each lever tested in isolation and combined).

## Method summary

- **Architecture:** DINOv2-L ViT (`vit_large_patch14_reg4_dinov2`, Apache-2.0,
  loaded via [`timm`](https://github.com/huggingface/pytorch-image-models)) with a
  LoRA adapter (rank 16, our own minimal from-scratch implementation, no `peft`
  dependency — see [`src/lora.py`](src/lora.py)) and a per-patch MIL classification
  head ([`src/model.py`](src/model.py)).
- **Synthetic fraud generation:** an annotation-driven attack suite (portrait swap,
  cross-card region swap, text-field edit, erase-and-retype) placed using manually
  annotated face/field boxes per document type ([`annotations/type_fields.json`](annotations/type_fields.json)),
  implemented in [`src/augment.py`](src/augment.py).
- **Resolution:** 448×728, letterboxed (aspect-preserving).
- **Loss:** Focal BCE (α=0.25, γ=2.0) on the whole-image logit.
- **Validation / selection protocol:** in-domain FREUID validation does not reliably
  predict generalization (established repeatedly during development), so every
  design decision that could be validated (checkpoint selection for `cv5`, the TTA
  grid, patch-aggregation sweep) was scored with the exact competition metric
  ([`src/freuid_metric.py`](src/freuid_metric.py)) on held-out IDNet data. The metric
  is invariant to monotone score transforms, so no score calibration of any kind is
  applied (provably a no-op).

## Repository layout

```
src/                  Training + inference + evaluation code
  train.py            Main training script (see "Training" below for exact commands)
  model.py, lora.py   Model architecture
  data.py             Image loading / letterbox / normalization
  augment.py          Augmentation + annotation-driven synthetic attack generation
  freuid_metric.py    Exact competition metric (AuDET, APCER@1%BPCER, harmonic mean)
  infer.py            Batch inference -> Kaggle-format submission CSV (supports --tta_scales)
  eval_external.py    Unified external-generalization eval (IDNet EST+SVK pool)
  tta_grid.py         TTA scale-combination grid (how the 4-scale TTA was selected)
  tta_confirm_full.py Full-pool confirmation of the selected TTA config
  patch_agg_grid.py   Patch-aggregation sweep (negative result; kept for the record)
  make_splits.py      Regenerates splits/folds.csv (stratified 5-fold, seed 42)
annotations/
  type_fields.json    Manual per-document-type face/text-field bounding boxes
splits/
  folds.csv           Stratified 5-fold split of the FREUID training set (fold 0 used)
weights/
  cv3_fold0.pt        Pick 1 (primary): FREUID-only, fold 0, epoch 1 — LEAN checkpoint
  cv5_full_ep2.pt     Pick 2: full-data FREUID+IDNet, epoch 2 — LEAN checkpoint
Dockerfile            Reproducibility container (see "Docker / reproduction" below)
docker/
  prepare_submission.py  Container entrypoint (loads weights, runs inference, optional TTA)
  requirements.txt       Pinned inference dependencies
report/               Technical report (LaTeX source + compiled PDF)
```

## Weight format (lean checkpoints)

The `weights/*.pt` files are **lean checkpoints**: they contain only the parameters
that training actually changed — 192 LoRA A/B matrices + 8 head tensors, ~27 MB each
(plain git files, no Git LFS needed). The DINOv2-L backbone is frozen during
training, so at load time it is rebuilt from the official `timm` pretrained weights
(Apache-2.0): the Dockerfile downloads them **at build time** into the image's HF
cache; container runtime performs no network access.

This conversion is lossless and audited by [`src/make_lean_ckpt.py`](src/make_lean_ckpt.py)
(run with `--verify`): every backbone tensor in the full training checkpoints is
bit-identical to the timm release (343/343), and lean-loaded models produce
bit-identical fp32 logits to the full checkpoints.

## Data

- **FREUID Challenge dataset:** obtain from the official Kaggle competition page and
  place under `the-freuid-challenge-dataset/` at the repo root (`train/`,
  `train_labels.csv`, ...) matching [`src/data.py`](src/data.py)'s expected layout.
- **IDNet** (used only by `cv5_ep2` training and by the external validation
  protocol): obtain from the dataset's official release (CC BY 4.0) and build a flat
  index CSV (`id,path,label,type,source`) at `external/idnet_cropped_index.csv`;
  `type` values are per-country codes (e.g. `ESP_scanned`, `EST_scanned`, ...).

Neither dataset is redistributed in this repository (size + license terms); only the
code and our own field/face annotations are included.

## Training

**Finalist 1 — `cv3` (FREUID-only, fold 0):**

```bash
python src/train.py \
    --fold 0 --epochs 3 --bs 6 --accum 4 --eval_bs 8 \
    --res 448x728 --sbi 0.25 --attacks full \
    --idnet_countries "" \
    --workers 16 --tag cv3
```

The submitted checkpoint is **epoch 1** of this run (`weights/cv3_fold0.pt`).

**Finalist 2 — `cv5_ep2` (100% FREUID + all-10-country IDNet):**

```bash
python src/train.py \
    --full_data --epochs 5 --bs 6 --accum 4 --eval_bs 8 \
    --res 448x728 --sbi 0.25 --attacks full \
    --idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned,EST_scanned,SVK_scanned \
    --heldout_idnet "" --idn_val_from_unused \
    --lim_idn 80000 --idn_val_n 4000 --select_on idnet \
    --workers 16 --save_every 250 --tag cv5
```

This trains on ~149k images/epoch (69,352 FREUID + 80,000 balanced IDNet rows);
per-epoch validation uses 4,000 IDNet images sampled from the pool rows NOT selected
into the 80k training cap (same countries, zero row overlap). The submitted
checkpoint is **epoch 2** (`weights/cv5_full_ep2.pt`). Training used a single NVIDIA
RTX A4500 (20GB); ≈5h/epoch at this configuration.

## Local inference (outside Docker)

`src/infer_patchagg.py` implements the patch re-aggregation (top-5%, w=0.25) used by
both final picks:

```bash
# Pick 1 (primary): cv3 + patch re-agg + 4-scale TTA
python src/infer_patchagg.py --ckpt weights/cv3_fold0.pt \
    --tta_scales 0.85,0.9,1.0,1.1 --out sub_cv3_pagg_tta4.csv

# Pick 2: cv5_ep2 + patch re-agg + 4-scale TTA
python src/infer_patchagg.py --ckpt weights/cv5_full_ep2.pt \
    --tta_scales 0.85,0.9,1.0,1.1 --out sub_cv5_full_ep2_pagg_tta4.csv
```

## Docker / reproduction

One image reproduces both selected final picks; the pick is chosen by a single
documented environment variable (inference orchestration only — same frozen weights,
identical TTA and patch-aggregation settings for both picks).

```bash
docker build -t freuid-repro:local .

# PICK 1 (primary) — reproduces Kaggle submission "final_cv3_pagg_tta4_full.csv"
# (cv3 weights + patch re-agg top-5%/w=0.25 + 4-scale TTA).
# These are the image defaults, so no -e flags are needed:
docker run --rm --gpus all \
  --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local

# PICK 2 — reproduces Kaggle submission "final_cv5_pagg_tta4_full.csv"
# (cv5_ep2 weights; all other settings identical):
docker run --rm --gpus all \
  --network none \
  -e FREUID_MODEL_PATH=/models/cv5_full_ep2.pt \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Measured inference cost (RTX A4500, 20 GB): ~160 ms/image for the 4-scale TTA
configuration used by both picks (images decoded once, decode overlapped with GPU
compute via threaded prefetch) — ~6.3 h for the full 142,818-image hidden set on
our A4500, i.e. approximately 2–3 h on an A100 (2–3× faster on FP16 throughput
and memory bandwidth), within the 6 h limit. Batch size and decode threads are
tunable via `FREUID_BATCH_SIZE` (default 32) and `FREUID_DECODE_THREADS`
(default 8) if more headroom is desired.

`docker/prepare_submission.py` discovers every image directly under `/data/`
(`.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff`), runs the same DINOv2-L+LoRA model and
preprocessing used in training/validation, and writes `/submissions/submission.csv`
with columns `id,label` (`id` = filename stem, `label` = fraud score, higher = more
confident fraudulent). No network access is required or used at inference time; both
finalist weights are baked into the image under `/models/`.

## Licenses

- This repository's code: **MIT** (see [`LICENSE`](LICENSE)).
- DINOv2 backbone weights/architecture (via `timm`): **Apache-2.0**.
- IDNet training data: **CC BY 4.0** (not redistributed here; see "Data" above).
- We deliberately avoided any non-OSI-licensed code or weights (e.g. TruFor,
  ConvNeXt V2, DINOv3) anywhere in the pipeline used for this submission.
