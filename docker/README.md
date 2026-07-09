# Docker submission — notes

The Dockerfile lives at the **repository root** (not in this folder) so its build
context can reach `src/model.py`, `src/lora.py`, and `weights/cv4_fold0.pt`. This
folder holds only the two files specific to the container entrypoint:

- `prepare_submission.py` — organizer-facing entrypoint. Discovers images under
  `/data/`, runs our trained model, writes `/submissions/submission.csv`.
- `requirements.txt` — pinned inference dependencies (torch, timm,
  opencv-python-headless, numpy, pandas), matching the exact versions used for
  training (see `../report/`).

Build and run from the **repository root**:

```bash
docker build -t freuid-repro:local .

docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

GPU note: `torch` auto-detects CUDA (`torch.cuda.is_available()`) and falls back to
CPU automatically — no separate build variant needed. Add `--gpus all` to the `docker
run` command above to use a GPU if the host has one and `nvidia-container-toolkit`
installed.
