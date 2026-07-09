# FREUID Challenge 2026 — reproducibility Dockerfile
# Team: <FILL IN TEAM NAME>
#
# Build (from repo root):
#   docker build -t freuid-repro:local .
# Run (organizer contract):
#   docker run --rm --network none \
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
    FREUID_MODEL_PATH=/models/cv4_fold0.pt

WORKDIR /app

# System libraries for OpenCV image loading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/prepare_submission.py .

# Model architecture (DINOv2-L + LoRA) -- only the two files inference actually needs,
# not the full src/ tree (which also has training-only code/deps not required here).
COPY src/model.py src/lora.py /app/model_src/

# Trained weights (see ../report for training details; frozen at code-freeze commit).
COPY weights/cv4_fold0.pt /models/cv4_fold0.pt

RUN useradd --create-home --uid 1000 runner && \
    chown -R runner:runner /app /models
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
