#!/usr/bin/env python3
"""
FREUID Challenge 2026 — submission entrypoint (Team: Nadhir Hasan).

Model: FreuidModel (DINOv2-L ViT + LoRA r=16 + per-patch MIL head), trained via
../src/train.py on FREUID + IDNet-mixed data with the annotation-driven attack suite
(tag "cv4", fold 0). See ../README.md and ../report/ for the full method description.

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
MODEL_PATH = Path(os.environ.get("FREUID_MODEL_PATH", "/models/cv4_fold0.pt"))

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Must match training exactly (train.py --res 448x728; data.py's letterbox/normalize).
INPUT_H, INPUT_W = 448, 728
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BATCH_SIZE = int(os.environ.get("FREUID_BATCH_SIZE", "16"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FREUID submission.csv from images in /data."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory of test images (default: $FREUID_DATA_DIR or /data).")
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH,
                        help="Output CSV path (default: $FREUID_SUBMISSION_PATH).")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="Checkpoint path (default: $FREUID_MODEL_PATH or /models/cv4_fold0.pt).")
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


def _load_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _letterbox(img, INPUT_H, INPUT_W)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    model = FreuidModel(pretrained=False, lora_r=args.get("lora_r", 16)).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


def predict_labels(image_rows: list[tuple[str, Path]], seed: int,
                    model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """
    Run inference for every test image with our trained FreuidModel
    (DINOv2-L + LoRA, cv4/fold0). Returns columns: id, label.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prepare_submission] device={device}  model={model_path}", file=sys.stderr)
    model = load_model(model_path, device)

    ids: list[str] = []
    labels: list[float] = []
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    with torch.inference_mode():
        for start in range(0, len(image_rows), BATCH_SIZE):
            batch = image_rows[start:start + BATCH_SIZE]
            x = torch.stack([_load_tensor(p) for _, p in batch]).to(device)
            with torch.autocast(device, dtype=amp_dtype, enabled=(device == "cuda")):
                logits = model(x)
            scores = torch.sigmoid(logits).float().cpu().numpy()
            ids.extend(rid for rid, _ in batch)
            labels.extend(float(s) for s in scores)
            if (start // BATCH_SIZE) % 20 == 0:
                print(f"[prepare_submission] {start + len(batch)}/{len(image_rows)}", file=sys.stderr)

    out = pd.DataFrame({"id": ids, "label": labels})
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
