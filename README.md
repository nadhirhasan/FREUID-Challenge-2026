# FREUID Challenge 2026 — Reproducibility Repository

Team: **\<FILL IN TEAM NAME>** · Kaggle usernames: **\<FILL IN>**

This repository contains the training code, inference code, and a runnable Docker
submission entrypoint for our FREUID Challenge 2026 solution (tag `cv4`). See
[`report/`](report/) for the full technical report (method, data, results).

## Method summary

- **Architecture:** DINOv2-L ViT (`vit_large_patch14_reg4_dinov2`, Apache-2.0,
  loaded via [`timm`](https://github.com/huggingface/pytorch-image-models)) with a
  LoRA adapter (rank 16, our own minimal from-scratch implementation, no `peft`
  dependency — see [`src/lora.py`](src/lora.py)) and a per-patch MIL classification
  head ([`src/model.py`](src/model.py)).
- **Training data:** the FREUID Challenge training set (all 5 document types) mixed
  with 8 countries from the [IDNet](https://arxiv.org/abs/2409.10472) dataset
  (CC BY 4.0) for cross-document-type generalization; 2 further IDNet countries
  (Estonia, Slovakia) held out entirely and used only to select the checkpoint
  (never trained on — see "Validation" below).
- **Synthetic fraud generation:** an annotation-driven attack suite (portrait swap,
  cross-card region swap, text-field edit, erase-and-retype) placed using manually
  annotated face/field boxes per document type ([`annotations/type_fields.json`](annotations/type_fields.json)),
  implemented in [`src/augment.py`](src/augment.py).
- **Resolution:** 448×728, letterboxed (aspect-preserving).
- **Loss:** Focal BCE (α=0.25, γ=2.0) on the whole-image logit.
- **Validation / checkpoint selection:** we select the checkpoint by its score on a
  held-out sample of two IDNet countries never seen in training, using the exact
  competition metric (`src/freuid_metric.py`) — **not** the in-domain FREUID
  validation fold, which we found during development does not reliably predict
  generalization (see the report for details).

## Repository layout

```
src/                  Training + inference code
  train.py            Main training script (see "Training" below for the exact command)
  model.py, lora.py   Model architecture
  data.py             Image loading / letterbox / normalization
  augment.py          Augmentation + annotation-driven synthetic attack generation
  freuid_metric.py     Exact competition metric (AuDET, APCER@1%BPCER, harmonic mean)
  infer.py            Batch inference over a folder -> Kaggle-format submission CSV
  make_splits.py      Regenerates splits/folds.csv (stratified 5-fold, seed 42)
annotations/
  type_fields.json    Manual per-document-type face/text-field bounding boxes
splits/
  folds.csv           Stratified 5-fold split of the FREUID training set (fold 0 used)
weights/
  cv4_fold0.pt        Frozen model weights for the submitted checkpoint
Dockerfile             Reproducibility container (see "Docker / reproduction" below)
docker/
  prepare_submission.py  Container entrypoint (loads weights, runs inference)
  requirements.txt        Pinned inference dependencies
report/                 Technical report (LaTeX source + compiled PDF)
```

## Data

- **FREUID Challenge dataset:** obtain from the official Kaggle competition page and
  place under `the-freuid-challenge-dataset/` at the repo root (`train/`,
  `train_labels.csv`, ...) matching [`src/data.py`](src/data.py)'s expected layout.
- **IDNet:** obtain from the dataset's official release (CC BY 4.0) and build a flat
  index CSV (`id,path,label,type,source`) at `external/idnet_cropped_index.csv`;
  `type` values are the per-country codes used in `--idnet_countries` /
  `--heldout_idnet` below (e.g. `ESP_scanned`, `EST_scanned`, ...).

Neither dataset is redistributed in this repository (size + license terms); only the
code and our own field/face annotations are included.

## Training

```bash
python src/train.py \
    --fold 0 --epochs 3 --bs 6 --accum 4 --eval_bs 8 \
    --res 448x728 --sbi 0.25 --attacks full \
    --idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned \
    --heldout_idnet EST_scanned,SVK_scanned \
    --lim_idn 56000 --idn_val_n 4000 --select_on idnet \
    --workers 16 --tag cv4
```

This trains on ~111k images/epoch (≈55k FREUID fold-0 training rows + 56k balanced
IDNet rows), 3 epochs, and saves the checkpoint with the best exact-FREUID-metric
score on the 4,000-document IDNet held-out set to `checkpoints/cv4_fold0.pt`.
Training used a single NVIDIA RTX A4500 (20GB); ~12–14h/fold at this configuration.

The submitted checkpoint (`weights/cv4_fold0.pt`) is epoch **\<FILL IN: 1 or 2>** of
this run — see the report for the exact selection rationale.

## Local inference (outside Docker)

```bash
python src/infer.py --ckpt weights/cv4_fold0.pt --out submission.csv --eval_bs 32
```

## Docker / reproduction

Build and run exactly as organizers will:

```bash
docker build -t freuid-repro:local .

docker run --rm \
  --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

`docker/prepare_submission.py` discovers every image directly under `/data/`
(`.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff`), runs the same DINOv2-L+LoRA model and
preprocessing used in training/validation, and writes `/submissions/submission.csv`
with columns `id,label` (`id` = filename stem, `label` = fraud score, higher = more
confident fraudulent). No network access is required or used at inference time; all
weights are baked into the image at `/models/cv4_fold0.pt`.

## Licenses

- This repository's code: **MIT** (see [`LICENSE`](LICENSE)).
- DINOv2 backbone weights/architecture (via `timm`): **Apache-2.0**.
- IDNet training data: **CC BY 4.0** (not redistributed here; see "Data" above).
- We deliberately avoided any non-OSI-licensed code or weights (e.g. TruFor,
  ConvNeXt V2, DINOv3) anywhere in the pipeline used for this submission.
