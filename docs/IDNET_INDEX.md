# Building `external/idnet_cropped_index.csv`

`cv5_ep2` (the winning pick), the external validation protocol, and the IDNet entries in
`annotations/type_fields.json` all read one file: `external/idnet_cropped_index.csv`. Neither
that CSV nor the images it points to are in this repository. IDNet is not ours to
redistribute, and our local copy of the data and index was deleted after the competition.

This document explains how to rebuild it. The three scripts it uses are **verbatim copies**
of the ones that built the original index. They were added to `src/` after the competition
and have not been edited:

| Script | Role |
|---|---|
| [`src/external_data.py`](../src/external_data.py) | Download + extract the IDNet archives, write the raw index `external/idnet_index.csv`; also holds the card-cropping functions |
| [`src/precrop_idnet.py`](../src/precrop_idnet.py) | Crop every raw image once to a tight card and write `external/idnet_cropped_index.csv` |
| [`src/build_cropped_index.py`](../src/build_cropped_index.py) | Optional: rebuild `idnet_cropped_index.csv` from the cropped folder alone |

---

## Contents

- [Pipeline at a glance](#pipeline-at-a-glance)
- [What was selected (and what was not)](#what-was-selected-and-what-was-not)
- [Step 0 — Setup](#step-0--setup)
- [Step 1 — Download, extract, raw index](#step-1--download-extract-raw-index)
- [Step 2 — Crop to FREUID-style cards](#step-2--crop-to-freuid-style-cards)
- [Step 3 (optional) — Rebuild the index from the crops](#step-3-optional--rebuild-the-index-from-the-crops)
- [Step 4 — Sanity checks](#step-4--sanity-checks)
- [How training and evaluation use the index](#how-training-and-evaluation-use-the-index)
- [Can the exact 80,000 training rows be reproduced?](#can-the-exact-80000-training-rows-be-reproduced)
- [Known quirks (kept as-is)](#known-quirks-kept-as-is)
- [Source and license](#source-and-license)

---

## Pipeline at a glance

```
Hugging Face dataset  cactuslab/IDNet-2025      (10 x <COUNTRY>_scanned.tar.gz)
        |
        |  python src/external_data.py --download <10 country archives>
        |      download -> extract -> index every genuine/fraud image
        v
external/idnet/<COUNTRY>_scanned/scanned/<data folder>/*.jpg
external/idnet_index.csv                           (raw, uncropped)
        |
        |  python src/precrop_idnet.py --workers 16
        |      undo scan rotation -> tight card crop -> width <= 1100 px -> JPEG q95
        v
external/idnet_cropped/<same relative path>.jpg
external/idnet_cropped_index.csv                   <- what train.py / eval_external.py read
        |
        |  (all subsampling happens here, never in the index)
        v
src/train.py  (cv5: 40k genuine + 40k fraud for training, 2k + 2k disjoint for validation)
src/eval_external.py  (full EST_scanned + SVK_scanned pool)
```

## What was selected (and what was not)

- **Source:** the Hugging Face copy [`cactuslab/IDNet-2025`](https://huggingface.co/datasets/cactuslab/IDNet-2025).
  It ships one plain archive and one `_scanned` archive for each of 10 European countries.
- **Note on Zenodo:** the technical report's IDNet reference cites the dataset's official
  Zenodo release. That citation is about the dataset and its license. The files used for this
  submission were **not** downloaded from Zenodo, and the Zenodo archives are packaged
  differently: one `.zip` per country (e.g. `EST.zip`) with different sizes, and no separate
  `_scanned` archives. To match `cv5`, use the Hugging Face copy.
- **Used:** the `_scanned` archive for all 10 countries: `ALB_scanned`, `AZE_scanned`,
  `ESP_scanned`, `EST_scanned`, `FIN_scanned`, `GRC_scanned`, `LVA_scanned`, `RUS_scanned`,
  `SRB_scanned`, `SVK_scanned`. The scanned variants add scanner background, shadows and
  local sharpening, which is closer to FREUID's print-capture look than the clean renders.
- **Not used:** the plain template archives (`ALB.tar.gz`, …), `models.tar.gz`, and the
  US-state IDNet documents.
- **Included from each archive:** every image under `positive/` (genuine) and under the two
  fraud folders, `fraud5_inpaint_and_rewrite/` and `fraud6_crop_and_replace/`.
- **No filtering, deduplication or subsampling was applied when building the index.** The same
  rules were applied to all 10 countries. Rows were lost in one way only: if cropping an image
  raised an error, that image was skipped. How many were skipped was not logged.
- **All subsampling happens later, in `src/train.py`.** It balances by **label**, not by
  country. See [How training and evaluation use the index](#how-training-and-evaluation-use-the-index).

## Step 0 — Setup

Run everything from the repository root. The scripts locate `external/` through `ROOT` in
`src/data.py`, which is the repository root.

```bash
pip install huggingface_hub pandas opencv-python-headless numpy
```

`huggingface_hub` is only needed for the download and is not part of the inference
requirements in `docker/requirements.txt`.

**Disk space.** The 10 `_scanned` archives total about **60 GB**. Sizes on the Hugging Face
release at the time of writing:

| archive | size | | archive | size |
|---|---|---|---|---|
| `ALB_scanned.tar.gz` | 4.2 GB | | `GRC_scanned.tar.gz` | 7.0 GB |
| `AZE_scanned.tar.gz` | 6.2 GB | | `LVA_scanned.tar.gz` | 7.9 GB |
| `ESP_scanned.tar.gz` | 5.1 GB | | `RUS_scanned.tar.gz` | 6.0 GB |
| `EST_scanned.tar.gz` | 3.0 GB | | `SRB_scanned.tar.gz` | 10.3 GB |
| `FIN_scanned.tar.gz` | 5.9 GB | | `SVK_scanned.tar.gz` | 4.4 GB |

The images inside are already JPEG-compressed, so the extracted folders take roughly as much
space again. Budget about **120 GB** for archives plus extracted images, plus room for the crops
in `external/idnet_cropped/`. The crops are smaller: width capped at 1100 px, JPEG quality 95.
Once the cropped index exists, the archives and raw extractions can be deleted;
`src/build_cropped_index.py` exists so the index can still be rebuilt without them.

## Step 1 — Download, extract, raw index

```bash
python src/external_data.py --download ALB_scanned,AZE_scanned,ESP_scanned,EST_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned,SVK_scanned
```

This one command runs three stages, all in [`src/external_data.py`](../src/external_data.py):

1. **Download** (`download()`). Each `<name>.tar.gz` is fetched with `hf_hub_download` into
   `external/idnet/_archives/`, with up to 6 attempts per file; retries force a fresh download.
   Downloading in several batches is fine, because the index is rebuilt from a full rescan
   of `external/idnet/` every time.
2. **Extract** (`extract()`). Each archive is unpacked into a folder named after the archive,
   `external/idnet/<COUNTRY>_scanned/`. Folders that already exist and are non-empty are
   skipped. The archives have a top-level `scanned/` folder, so the images end up at:

   ```
   external/idnet/
     _archives/                       *.tar.gz (no images; ignored by the indexer)
     EST_scanned/
       scanned/
         positive/                         genuine (.jpg)
         positive_info/                    one JSON per image
         fraud5_inpaint_and_rewrite/       fraud (.jpg)
         fraud5_inpaint_and_rewrite_info/  one JSON per image
         fraud6_crop_and_replace/          fraud (.jpg)
         fraud6_crop_and_replace_info/     one JSON per image
     ESP_scanned/
       scanned/ ...
     ...
   ```
   Each image has a JSON file with the same name in the matching `_info` folder. It records the
   settings used to simulate the scan: brightness, contrast, noise, shadow, the `rotate` angle,
   card corner positions, and so on. The indexer skips these files because they are not images;
   the cropper in Step 2 reads `rotate` from them.

   For reference, `EST_scanned` contains 5,979 `positive`, 5,979 `fraud5_inpaint_and_rewrite`
   and 5,978 `fraud6_crop_and_replace` images (17,936 in total), each with its JSON. Every
   country has two fraud folders for its one genuine folder, so the raw pool is roughly
   **2:1 fraud to genuine**. `EST_scanned` is almost exactly that.

3. **Index** (`index()`). Every file under `external/idnet/` is scanned recursively. Only
   image extensions are kept (`.jpg .jpeg .png .bmp .tif .tiff`), and each image gets a label
   from its path, lower-cased:

   | Path contains | `label` | Meaning |
   |---|---|---|
   | `positive` (checked first) | `0` | genuine |
   | `fraud` | `1` | fraud (`fraud5_inpaint_and_rewrite`, `fraud6_crop_and_replace`) |
   | neither | skipped | anything else (e.g. metadata) |

   `type` is the first folder under `external/idnet/`, e.g. `EST_scanned`. The script writes
   `external/idnet_index.csv` and prints the row count for each `(type, label)`.

## Step 2 — Crop to FREUID-style cards

```bash
python src/precrop_idnet.py --workers 16
```

**Why crop.** FREUID images are tightly framed cards. IDNet scanned images show the document
on a larger scanner canvas. Cropping once, ahead of training, makes IDNet look like FREUID and
keeps training fast because nothing is cropped per epoch. No face-based 0/180° auto-orientation
is applied: it was unreliable on stylized ID portraits, so training-time augmentation handles
orientation instead.

**What happens to each row of `idnet_index.csv`**, in `crop_card_meta()` in `src/external_data.py`:

1. **Undo the scan rotation using the metadata.** The script looks for a per-image JSON in a
   sibling folder named `<data folder>_info/<image stem>.json`. In the scanned archives every
   image has one (checked for `EST_scanned`: each image folder and its `_info` folder hold the
   same number of files), so this is the branch that actually ran. It reads the `rotate` angle
   (0 if missing) and rotates the image back by that angle. The canvas is
   enlarged with a white border so no corner is cut off (`_rotate_expand`). Then it crops
   tightly (`_bbox_crop`):
   - convert to grayscale; pixels **darker than 235** count as card;
   - apply a 25×25 morphological close;
   - take the bounding box of the largest contour, plus 2 px of padding.
2. **Fallback, if the JSON is missing or unreadable: crop geometrically** (`crop_document(orient=False)`). The same mask is built
   (threshold 235, 25×25 close), the largest contour is found, and `cv2.minAreaRect` fits a
   rotated rectangle to it. The rectangle is **forced to landscape** (long edge horizontal),
   the image is rotated to straighten the card, and the card is cut out with `getRectSubPix`.
   The **full image** is kept instead when any of these hold:
   - the contour covers less than 2% of the image;
   - the rectangle is smaller than 30×20 px;
   - anything raises an exception.
3. **Limit the size.** Crops wider than **1100 px** are downscaled to 1100 px wide
   (`INTER_AREA`, aspect ratio kept).
4. **Save** to `external/idnet_cropped/<same relative path>` with a `.jpg` extension, at JPEG
   quality 95. Existing outputs are skipped, so an interrupted run can be restarted.
5. **Drop failures.** If an image errors out, it gets no crop and no CSV row.

**Output:** `external/idnet_cropped_index.csv`, one row per successful crop, in the same order
as `idnet_index.csv`:

| column | example |
|---|---|
| `id` | `idnetc/EST_scanned/scanned/fraud6_crop_and_replace/scanned_79_generated.photos_v3_0387878_fake_1_2309.jpg` |
| `path` | absolute path, `<repo>/external/idnet_cropped/EST_scanned/scanned/fraud6_crop_and_replace/scanned_79_generated.photos_v3_0387878_fake_1_2309.jpg` |
| `label` | `1` |
| `type` | `EST_scanned` |
| `source` | `idnet` |

## Step 3 (optional) — Rebuild the index from the crops

```bash
python src/build_cropped_index.py
```

This rebuilds `external/idnet_cropped_index.csv` straight from `external/idnet_cropped/**/*.jpg`,
using the same path-based label rule, `type` taken from the first folder, and the same `id`
format. You never need the raw archives again, and the labels can't get lost.

Steps 2 and 3 produce the **same set of rows**: one per successfully cropped image. The only
possible difference is **row order**, because each follows the file-listing order of a
different folder. We have no record of which of the two wrote the final copy of our CSV.
Row order matters for reproducing the exact training subset;
see [below](#can-the-exact-80000-training-rows-be-reproduced).

## Step 4 — Sanity checks

```python
import pandas as pd
idn = pd.read_csv("external/idnet_cropped_index.csv")
print(idn.groupby(["type", "label"]).size())
print(len(idn[idn.type.isin(["EST_scanned", "SVK_scanned"])]))
```

- **Exactly 10 `type` values**, matching the IDNet keys in `annotations/type_fields.json`
  character for character (`ALB_scanned` … `SVK_scanned`). Training uses `type` to look up
  where synthetic attacks are placed (`--attacks full`), so a renamed folder quietly disables
  attack placement for that country.
- **EST + SVK = 35,874 rows** in our original index. This is the external-benchmark pool
  reported in the technical report and README, which makes it a good end-to-end checksum.
- **Card sizes.** The template boxes in `annotations/type_fields.json` were drawn on crops
  from our original index. Their reference sizes are below; your crops should come out at or
  very near these sizes. The boxes are stored as fractions of width and height, so a
  difference of a few pixels doesn't matter.

  | type | w × h | | type | w × h |
  |---|---|---|---|---|
  | `ALB_scanned` | 1027 × 655 | | `GRC_scanned` | 1100 × 782 |
  | `AZE_scanned` | 1100 × 784 | | `LVA_scanned` | 1100 × 782 |
  | `ESP_scanned` | 1015 × 643 | | `RUS_scanned` | 1100 × 777 |
  | `EST_scanned` | 1020 × 640 | | `SRB_scanned` | 1100 × 760 |
  | `FIN_scanned` | 1016 × 641 | | `SVK_scanned` | 1021 × 651 |

  Widths of exactly 1100 are types whose crops hit the width cap.

## How training and evaluation use the index

**`cv5_ep2` (winning pick).** The recipe, held verbatim, is:

```bash
--idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned,EST_scanned,SVK_scanned \
--heldout_idnet "" --idn_val_from_unused --lim_idn 80000 --idn_val_n 4000
```

These exact values are also stored in the saved checkpoint's `args`. The relevant code is
[`src/train.py`](../src/train.py) lines 247–267:

```python
idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
idn_pool = idn[idn.type.isin(args.idnet_countries.split(","))]          # all 10 countries
idn_tr = idn_pool.groupby("label", group_keys=False).apply(
    lambda g: g.sample(min(len(g), args.lim_idn // 2), random_state=0))  # 40k per label
...
unused = idn_pool[~idn_pool.id.isin(set(idn_tr.id))]                      # rows not picked
idn_val = unused.groupby("label", group_keys=False).apply(
    lambda g: g.sample(min(len(g), args.idn_val_n // 2), random_state=1)) # 2k per label
```

- **Training:** 40,000 genuine and 40,000 fraud rows, drawn uniformly at random from the
  **pooled** 10 countries. Nothing is stratified or capped per country, so each country's
  share of the 80k simply matches its share of the pool within each label. The pool is about
  2:1 fraud to genuine, so the 40k genuine rows are a much larger fraction of all genuine images
  than the 40k fraud rows are of all fraud images.
- **Validation (selects the checkpoint each epoch):** 2,000 genuine and 2,000 fraud rows drawn
  from pool rows that were *not* picked for training. The countries are the same, but no image
  appears in both sets.
- **Synthetic attacks on IDNet genuines:** the 40k genuine rows go through the same attack
  pipeline as FREUID genuines (`--sbi 0.25 --attacks full`). Attacks are placed using the
  `annotations/type_fields.json` entry for that row's `type`, and donor images come from the
  same `type`.

**`cv4` (earlier iteration, for comparison):** 8 countries for training (`--lim_idn 56000`,
28k per label). EST and SVK were held out entirely, and 2k per label from them were used
for validation.

**External benchmark ([`src/eval_external.py`](../src/eval_external.py)):** every
`EST_scanned` and `SVK_scanned` row, both labels, with no sampling (35,874 images). This is
a blind test for FREUID-only models. For `cv5` it is optimistic, because those countries
appear in its training data (different images).

## Can the exact 80,000 training rows be reproduced?

**Not exactly, without the original CSV, and we no longer have it.** `DataFrame.sample(random_state=0)`
chooses rows **by position**, so which 80k rows are picked depends on the order of rows in
the CSV. That order came from Python's `glob`, which lists files in whatever order the
filesystem returns them. Ours was built on Windows (NTFS).

What this means in practice:

- A rebuilt index gives a subset **statistically equivalent** to ours: the same pool, the same
  40k/40k label balance, the same per-country proportions in expectation.
- It gives the **identical** row set only if all of these match ours: the file listing order,
  the IDNet release, and which images failed to crop. Rebuilding on NTFS with the folder layout
  above has the best chance, but we can't verify the result without the original.
- If you want the subset to be deterministic across machines in your own work, sort the
  index (e.g. by `id`) before `train.py` reads it. That is a deliberate change from the
  original pipeline, which did not sort.

## Known quirks (kept as-is)

These scripts are published exactly as they were run. Watch out for these when reusing them:

- **Labels come from the absolute path.** The `positive` / `fraud` test runs on the full,
  lower-cased path. If your checkout path contains either word (e.g. `~/fraud-research/`),
  **every** row gets the same label. Clone to a neutral path, or change the test to use the
  path relative to `external/idnet`.
- **`path` is absolute.** The CSV is tied to the machine that built it. Rebuild it rather than
  copying it between machines.
- **Crop failures are silent.** Skipped images are not logged. Compare the row counts of
  `idnet_index.csv` and `idnet_cropped_index.csv` to see how many were lost.
- `src/external_data.py` also contains face-based 0/180° orientation code (`orient_upright`,
  which needs `src/models/yunet.onnx`). **The index pipeline never calls it**
  (`orient=False`), so that model file is not needed.

## Source and license

IDNet: Guo et al., *IDNet: A Novel Dataset for Identity Document Analysis and Fraud Detection
Research* ([arXiv:2409.10472](https://arxiv.org/abs/2409.10472)). IDNet's official release is on
Zenodo, which the technical report cites; the copy used here was downloaded from the
Hugging Face release
[`cactuslab/IDNet-2025`](https://huggingface.co/datasets/cactuslab/IDNet-2025); check that
dataset card for the current license and attribution terms before redistributing anything
derived from it.
