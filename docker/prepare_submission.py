#!/usr/bin/env python3
"""
FREUID Challenge 2026 — submission entrypoint (Team: Nadhir Hasan).

Model: FreuidModel (DINOv2-L ViT + LoRA r=16 + per-patch MIL head), trained via
../src/train.py. Two finalist checkpoints ship in the image, sharing IDENTICAL
inference-time options — re-aggregated patch head (0.25 * mean(top-5% patch logits)
+ 0.75 * attention branch) plus 4-scale logit-avg TTA (0.85, 0.9, 1.0, 1.1):
  /models/cv3_fold0.pt     PICK 1: FREUID-only recipe (tag "cv3", fold 0, epoch 1);
                           reproduces "sub_cv3_pagg_tta4" (public LB 0.00060).
  /models/cv5_full_ep2.pt  PICK 2: FREUID + IDNet(10 countries) full-data recipe
                           (tag "cv5", epoch 2); reproduces
                           "sub_cv5_full_ep2_pagg_tta4" (public LB 0.00191).
Select the pick via FREUID_MODEL_PATH (that is the only difference between picks).
TTA scales via FREUID_TTA_SCALES (comma list, empty = off); patch aggregation via
FREUID_PATCH_AGG ("topk_frac,w_topk", empty = the training aggregation top-10%/w=0.5).
Container defaults reproduce PICK 1 (see Dockerfile ENV).
See ../README.md and ../report/ for the full method description.

Organizers mount:
  /data/           read-only   test images only (flat directory, no CSV)
  /submissions/    read-write  must contain submission.csv after exit

Image filenames define row ids: ``{id}.jpeg`` (``.jpg`` / ``.png`` / ``.webp`` also
accepted). The document id is the filename stem.

Output schema: ``id,label`` where ``label`` is a real-valued fraud score
(higher = more confident the document is fraudulent) — same semantics as on the Kaggle leaderboard.
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "model_src"))
from model import FreuidModel  # noqa: E402

DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions"))
SUBMISSION_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", OUTPUT_DIR / "submission.csv"))
MODEL_PATH = Path(os.environ.get("FREUID_MODEL_PATH", "/models/cv3_fold0.pt"))
# Multi-scale logit-avg TTA, e.g. "0.85,0.9,1.0,1.1". Empty string = single-scale (no TTA).
TTA_SCALES = [float(s) for s in os.environ.get("FREUID_TTA_SCALES", "").split(",") if s.strip()]
# Patch-head aggregation override "topk_frac,w_topk" (e.g. "0.05,0.25").
# Empty = the aggregation baked into PatchHead.forward (top-10%, w=0.5) used in training.
_pagg = [float(v) for v in os.environ.get("FREUID_PATCH_AGG", "").split(",") if v.strip()]
PATCH_AGG = (_pagg[0], _pagg[1]) if len(_pagg) == 2 else None  # (topk_frac, w_topk) or None

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Must match training exactly (train.py --res 448x728; data.py's letterbox/normalize).
INPUT_H, INPUT_W = 448, 728
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BATCH_SIZE = int(os.environ.get("FREUID_BATCH_SIZE", "32"))
# Decode-ahead threading: CPU decodes future batches while the GPU runs the current one.
NUM_DECODE_THREADS = int(os.environ.get("FREUID_DECODE_THREADS", "8"))
PREFETCH_BATCHES = int(os.environ.get("FREUID_PREFETCH_BATCHES", "3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FREUID submission.csv from images in /data."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory of test images (default: $FREUID_DATA_DIR or /data).")
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH,
                        help="Output CSV path (default: $FREUID_SUBMISSION_PATH).")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="Checkpoint path (default: $FREUID_MODEL_PATH or /models/cv3_fold0.pt).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Unused by the real model; kept for template compatibility.")
    return parser.parse_args()


def discover_images(data_dir: Path) -> list[tuple[str, Path]]:
    """Return (id, path) pairs for every image file directly under ``data_dir``."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs: list[tuple[str, Path]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row_id = path.stem
        if not row_id:
            raise ValueError(f"Cannot derive id from filename: {path.name}")
        pairs.append((row_id, path))

    if not pairs:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Expected flat files like '{{id}}.jpeg'."
        )
    return pairs


def _letterbox(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Aspect-preserving resize + pad -- IDENTICAL to src/data.py's letterbox()."""
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w, 3), 0, np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _to_tensor(img: np.ndarray, out_h: int, out_w: int) -> torch.Tensor:
    img = _letterbox(img, out_h, out_w)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    lora_r = args.get("lora_r", 16)
    if "model_lean" in ck:
        # Lean checkpoint: only the trained parameters (LoRA adapters + head, ~27MB).
        # The frozen DINOv2-L backbone is rebuilt from the timm pretrained weights,
        # which are baked into the image's HF cache at BUILD time (see Dockerfile) --
        # no network access happens at runtime. We verified the full checkpoints'
        # backbone tensors are bit-identical to the timm release (343/343) and that
        # lean-loaded models produce bit-identical logits to the full checkpoints.
        model = FreuidModel(pretrained=True, lora_r=lora_r).to(device).eval()
        missing, unexpected = model.load_state_dict(ck["model_lean"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in lean checkpoint: {unexpected[:3]}")
    else:
        model = FreuidModel(pretrained=False, lora_r=lora_r).to(device).eval()
        model.load_state_dict(ck["model"])
    return model


def _image_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Image-level logit; honors the FREUID_PATCH_AGG re-aggregation if set."""
    if PATCH_AGG is None:
        return model(x)
    topk_frac, w_topk = PATCH_AGG
    head = model.head
    tok = model.backbone.forward_features(x)
    patch = head.norm(tok[:, head.n_prefix:])
    plog = head.scorer(patch).squeeze(-1)                     # B, N per-patch logits
    k = max(1, int(topk_frac * plog.shape[1]))
    topk = plog.topk(k, dim=1).values.mean(1)
    aw = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
    attnm = (plog * aw).sum(1)
    return w_topk * topk + (1.0 - w_topk) * attnm


def predict_labels(image_rows: list[tuple[str, Path]], seed: int,
                    model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """
    Run inference for every test image with our trained FreuidModel (DINOv2-L + LoRA).
    If FREUID_TTA_SCALES is set, averages logits across the scaled input resolutions
    (each scale rounded down to a multiple of 14 for the ViT patch grid).

    Each JPEG is decoded ONCE per image; all TTA scale tensors are produced from that
    single decode (keeps the 6h/A100 inference budget CPU-safe on the 142,818-image
    hidden set: measured ~48ms/img single-scale, ~210ms/img 4-scale TTA on an RTX
    A4500 -> ~1h / ~3-4h respectively on an A100).
    Returns columns: id, label.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = TTA_SCALES or [1.0]
    print(f"[prepare_submission] device={device}  model={model_path}  tta_scales={scales}"
          f"  patch_agg={PATCH_AGG or 'training-default'}", file=sys.stderr)
    model = load_model(model_path, device)

    sizes = [(int(INPUT_H * s) // 14 * 14, int(INPUT_W * s) // 14 * 14) for s in scales]
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    logit_sum = np.zeros(len(image_rows), np.float64)

    def _prep_batch(start: int) -> list[torch.Tensor]:
        """Decode each image once and build every scale tensor (runs on CPU threads,
        overlapped with GPU compute of the previous batch)."""
        rows = image_rows[start:start + BATCH_SIZE]
        imgs = [_load_rgb(p) for _, p in rows]
        return [torch.stack([_to_tensor(im, h, w) for im in imgs]) for (h, w) in sizes]

    starts = list(range(0, len(image_rows), BATCH_SIZE))
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=NUM_DECODE_THREADS) as pool:
        futures = {s: pool.submit(_prep_batch, s) for s in starts[:PREFETCH_BATCHES]}
        for i, start in enumerate(starts):
            if i + PREFETCH_BATCHES < len(starts):
                nxt = starts[i + PREFETCH_BATCHES]
                futures[nxt] = pool.submit(_prep_batch, nxt)
            tensors = futures.pop(start).result()
            n = tensors[0].shape[0]
            for x in tensors:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device, dtype=amp_dtype, enabled=(device == "cuda")):
                    logits = _image_logits(model, x)
                p = torch.sigmoid(logits).float().cpu().numpy()
                logit_sum[start:start + n] += _to_logit(p)
            if i % 20 == 0:
                print(f"[prepare_submission] {start + n}/{len(image_rows)}",
                      file=sys.stderr)
    scores = 1.0 / (1.0 + np.exp(-logit_sum / len(scales)))

    out = pd.DataFrame({"id": [rid for rid, _ in image_rows], "label": scores})
    if not np.isfinite(out["label"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite labels produced.")
    return out


def validate_submission(submission: pd.DataFrame, expected_ids: set[str]) -> None:
    if list(submission.columns) != ["id", "label"]:
        raise ValueError(
            f"submission.csv must have columns ['id', 'label']; got {list(submission.columns)}"
        )

    got = set(submission["id"].astype(str))
    missing = expected_ids - got
    extra = got - expected_ids
    if missing:
        raise ValueError(f"submission.csv missing {len(missing)} id(s), e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"submission.csv has {len(extra)} unexpected id(s), e.g. {sorted(extra)[:3]}")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()

    image_rows = discover_images(data_dir)
    expected_ids = {row_id for row_id, _ in image_rows}
    submission = predict_labels(image_rows, seed=args.seed, model_path=args.model_path)
    validate_submission(submission, expected_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
