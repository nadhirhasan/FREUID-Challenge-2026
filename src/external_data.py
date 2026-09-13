"""Download + index external ID-document datasets for cross-dataset validation/training.

IDNet-2025 (cactuslab/IDNet-2025, CC-BY-4.0): synthetic European IDs, genuine vs fraud
(inpaint/rewrite, crop/replace), with `_scanned` (print-capture-like) variants -> a
real, leakage-free cross-domain proxy for the FREUID print-capture test.

  python src/external_data.py --download EST_scanned,ESP_scanned   # ~8 GB
  python src/external_data.py --index                              # build external/idnet_index.csv
"""
from __future__ import annotations
import os, sys, argparse, tarfile, glob, json
import numpy as np, cv2
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT


def _json_for(img_path):
    """IDNet stores per-image metadata in a sibling <folder>_info/ as <name>.json."""
    d = os.path.dirname(img_path); folder = os.path.basename(d)
    info = os.path.join(os.path.dirname(d), folder + "_info")
    p = os.path.join(info, os.path.splitext(os.path.basename(img_path))[0] + ".json")
    return p if os.path.exists(p) else None


def _rotate_expand(rgb, ang, border=255):
    """Rotate CCW by `ang`, expanding canvas (white border) so nothing is clipped."""
    h, w = rgb.shape[:2]
    c, s = abs(np.cos(np.radians(ang))), abs(np.sin(np.radians(ang)))
    nW, nH = int(h * s + w * c), int(h * c + w * s)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += (nW - w) / 2; M[1, 2] += (nH - h) / 2
    return cv2.warpAffine(rgb, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(border, border, border))


def _bbox_crop(rgb, white=235, pad=2):
    """Tight crop of the non-white (card) region; assumes card is axis-aligned."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(g, white, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    return rgb[y0:y + h + pad, x0:x + w + pad]


def crop_card_meta(img_path):
    """Best crop using IDNet metadata: undo the known `rotate` -> upright card, then
    tight-crop. Falls back to geometric crop_document if no/invalid metadata."""
    rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    jp = _json_for(img_path)
    if jp is not None:
        try:
            rot = float(json.load(open(jp)).get("rotate", 0.0))
            return _bbox_crop(_rotate_expand(rgb, -rot))  # undo scan's `rotate` -> upright
        except Exception:
            pass
    return crop_document(rgb, orient=False)


def _order_pts(p):
    r = np.zeros((4, 2), np.float32)
    s = p.sum(1); r[0] = p[np.argmin(s)]; r[2] = p[np.argmax(s)]
    d = np.diff(p, axis=1); r[1] = p[np.argmin(d)]; r[3] = p[np.argmax(d)]
    return r  # tl, tr, br, bl


_YUNET = None
def _yunet():
    global _YUNET
    if _YUNET is None:
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yunet.onnx")
        _YUNET = cv2.FaceDetectorYN.create(mp, "", (320, 320), 0.5, 0.3, 5)
    return _YUNET


def _face_score(bgr):
    """Strength of the best detected upright face (area x confidence). 0 if none.
    YuNet detects UPRIGHT faces far better than rotated ones -> usable for 0/180."""
    h, w = bgr.shape[:2]
    sc = 512.0 / max(w, h) if max(w, h) > 512 else 1.0
    img = cv2.resize(bgr, (int(w * sc), int(h * sc))) if sc < 1 else bgr
    det = _yunet(); det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return 0.0
    return float(max(f[2] * f[3] * f[14] for f in faces))


def orient_upright(rgb):
    """Make the ID portrait upright: keep 0 vs 180 by which has the stronger YuNet
    face (DNN, reliable). Card is already landscape. Falls back to 0 on failure."""
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        s0 = _face_score(bgr)
        s180 = _face_score(cv2.rotate(bgr, cv2.ROTATE_180))
        return cv2.rotate(rgb, cv2.ROTATE_180) if s180 > s0 else rgb
    except Exception:
        return rgb


def crop_document(rgb, white=235, min_area_frac=0.02, orient=False):
    """Detect the ID card on a (near-white) scan, deskew + crop to a tight landscape
    card — to match FREUID's tight-cropped framing. orient=True fixes 0/180 via face.
    Falls back to the full image on failure."""
    try:
        out = _crop_document(rgb, white, min_area_frac)
        return orient_upright(out) if orient else out
    except Exception:
        return rgb


def _crop_document(rgb, white=235, min_area_frac=0.02):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, white, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * h * w:
        return rgb
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < rh:                       # FORCE LANDSCAPE: long edge -> horizontal
        rw, rh = rh, rw; ang += 90.0
    if rw < 30 or rh < 20:
        return rgb
    # rotate whole image so the card is axis-aligned, then crop centered (no 90deg flips)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.getRectSubPix(rot, (int(rw), int(rh)), (cx, cy))

REPO = "cactuslab/IDNet-2025"
EXT = os.path.join(ROOT, "external", "idnet")
ARCH = os.path.join(EXT, "_archives")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def download(files):
    from huggingface_hub import hf_hub_download
    os.makedirs(ARCH, exist_ok=True)
    for f in files:
        for attempt in range(6):
            try:
                print(f"downloading {f} (attempt {attempt+1}) ...", flush=True)
                p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=ARCH,
                                    force_download=attempt > 0)
                print(f"  -> {f} OK ({os.path.getsize(p)/1e9:.2f} GB)", flush=True)
                break
            except Exception as e:
                print(f"  {f} attempt {attempt+1} FAILED: {str(e)[:140]}", flush=True)
        else:
            print(f"  *** GAVE UP on {f} (continuing) ***", flush=True)


def extract(files):
    for f in files:
        ap = os.path.join(ARCH, f)
        if not os.path.exists(ap):
            print(f"  missing archive {f}, skip"); continue
        out = os.path.join(EXT, f.replace(".tar.gz", ""))
        if os.path.isdir(out) and os.listdir(out):
            print(f"  {f} already extracted"); continue
        print(f"extracting {f} -> {out}", flush=True)
        os.makedirs(out, exist_ok=True)
        with tarfile.open(ap, "r:gz") as t:
            t.extractall(out)


def index():
    rows = []
    for img in glob.glob(os.path.join(EXT, "**", "*"), recursive=True):
        if not img.lower().endswith(IMG_EXT):
            continue
        low = img.lower()
        if "positive" in low:
            label = 0
        elif "fraud" in low:
            label = 1
        else:
            continue  # skip non genuine/fraud (e.g., meta/model artifacts)
        rel = os.path.relpath(img, EXT)
        country = rel.split(os.sep)[0]  # e.g. EST_scanned
        rows.append({"id": "idnet/" + rel.replace(os.sep, "/"),
                     "path": img, "label": label,
                     "type": country, "source": "idnet"})
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "external", "idnet_index.csv")
    df.to_csv(out, index=False)
    print(f"indexed {len(df)} images -> {out}")
    if len(df):
        print(df.groupby(["type", "label"]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", default="", help="comma list e.g. EST_scanned,ESP_scanned")
    ap.add_argument("--extract", default="", help="comma list (defaults to --download set)")
    ap.add_argument("--index", action="store_true")
    args = ap.parse_args()
    dl = [x.strip() + (".tar.gz" if not x.strip().endswith(".tar.gz") else "")
          for x in args.download.split(",") if x.strip()]
    ex = [x.strip() + (".tar.gz" if not x.strip().endswith(".tar.gz") else "")
          for x in args.extract.split(",") if x.strip()] or dl
    if dl:
        download(dl)
    if ex:
        extract(ex)
    if args.index or dl:
        index()


if __name__ == "__main__":
    main()
