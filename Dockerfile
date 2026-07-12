# FREUID Challenge 2026 — reproducibility Dockerfile
# Team: Nadhir Hasan
#
# Build (from repo root):
#   docker build -t freuid-repro:local .
# Run (organizer contract):
#   docker run --rm --gpus all --network none \
#     -v /path/to/flat/test/images:/data:ro \
#     -v "$(pwd)/out:/submissions" \
#     freuid-repro:local

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    FREUID_MODEL_PATH=/models/cv3_fold0.pt \
    FREUID_TTA_SCALES=0.85,0.9,1.0,1.1 \
    FREUID_PATCH_AGG=0.05,0.25
# ^ Both selected finalists use IDENTICAL inference-time options (patch re-agg
#   top-5%/w=0.25 + 4-scale logit-avg TTA); they differ only in the weights file.
#   Defaults reproduce PICK 1 ("sub_cv3_pagg_tta4", public LB 0.00060).
#   To reproduce PICK 2 ("sub_cv5_full_ep2_pagg_tta4", public LB 0.00191) run with:
#     -e FREUID_MODEL_PATH=/models/cv5_full_ep2.pt

WORKDIR /app

# System libraries for OpenCV image loading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the frozen DINOv2-L backbone (timm pretrained, Apache-2.0) into the image's
# HF cache at BUILD time. The repo's weight files are LEAN checkpoints (only the
# ~27MB of trained LoRA+head parameters); at runtime the backbone is rebuilt from
# this cache with zero network access. The full training checkpoints' backbone
# tensors were verified bit-identical to this timm release (343/343 tensors), and
# lean-loaded models produce bit-identical logits to the full checkpoints.
ENV HF_HOME=/models/hf_cache
RUN python -c "import timm; timm.create_model('vit_large_patch14_reg4_dinov2', pretrained=True, num_classes=0, dynamic_img_size=True)"

COPY docker/prepare_submission.py .

# Model architecture (DINOv2-L + LoRA) -- only the two files inference actually needs,
# not the full src/ tree (which also has training-only code/deps not required here).
COPY src/model.py src/lora.py /app/model_src/

# Trained weights: both selected finalists (lean; see ../report; frozen at code freeze).
COPY weights/cv3_fold0.pt /models/cv3_fold0.pt
COPY weights/cv5_full_ep2.pt /models/cv5_full_ep2.pt

RUN useradd --create-home --uid 1000 runner && \
    chown -R runner:runner /app /models
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
