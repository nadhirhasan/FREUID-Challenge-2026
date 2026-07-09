"""Evidence-based augmentation for FREUID (grouped & toggleable).

Defaults follow the literature for FORENSIC / cross-domain generalization:
  - Wang et al. CVPR'20: JPEG + blur + downscale -> generalization to unseen forgeries.
  - SBI CVPR'22: RGBShift/HueSat/BrightnessContrast/Downscale/Sharpen + translate/scale
    create the blending boundary for self-blended overlays.
  - ID-PAD baselines: random JPEG, blur, hue, brightness/contrast.
Weighted toward what WON real leaderboards (DFDC 1st place) over paper novelty:
  moderate jpeg+color, LOW noise/blur, and CUTOUT/DROPOUT (the winner's key lever).
Deliberately EXCLUDED (destroy forensic cues / break ID layout / didn't help winners):
  flips, heavy rotation, large cutout over the manipulated region, elastic, channel
  shuffle, and the aggressive compound print-capture ('pcapture_heavy' + crude 'moire')
  which risks the "synthetic utility gap" (overfitting our own artefact, not real fraud).
  moire/pcapture(_heavy)/orient remain available via `groups=` for ablation only.

Groups (default = degrade,color,noise,geometry,dropout): 'degrade' (jpeg/downscale/blur),
        'color', 'noise', 'geometry', 'dropout'; ablation: 'moire','pcapture','pcapture_heavy','orient'.
"""
from __future__ import annotations
import os, json
import cv2
import numpy as np
import albumentations as A

cv2.setNumThreads(0)


# ---- Manual per-TYPE field/face annotations (tools/annotate_fields.py) ----
def load_field_annotations(path):
    """Load annotations/type_fields.json -> (fields_by_type, faces_by_type).
    fields_by_type[type] = list of RELATIVE [rx,ry,rw,rh] text/data-field boxes.
    faces_by_type[type]  = list of RELATIVE (rx,ry,rw,rh) portrait boxes (main + optional ghost;
                           empty if none annotated). Accepts old single-"face" format too.
    Missing file -> empty dicts (attacks fall back to auto-detection)."""
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        d = json.load(f)
    fields = {t: [list(map(float, b)) for b in v.get("fields", [])] for t, v in d.items()}
    faces = {}
    for t, v in d.items():
        fl = v.get("faces")
        if not fl and v.get("face"):                 # migrate old single-face format
            fl = [v["face"]]
        faces[t] = [tuple(float(x) for x in b) for b in (fl or [])]
    return fields, faces


def _ann_band(fields, W, H, rng, bh_frac):
    """Place an edit band INSIDE a manually-annotated field box (exact text placement).
    Height clamped to the field but no taller than the requested band fraction (keeps text
    edits thin); width 70-100% of the field. Returns (x0, y0, bw, bh) in pixels or None."""
    if not fields:
        return None
    rx, ry, rw, rh = fields[int(rng.integers(len(fields)))]
    x0f, y0f = rx * W, ry * H
    wf, hf = max(1.0, rw * W), max(1.0, rh * H)
    bh = max(6.0, min(hf, rng.uniform(*bh_frac) * H))
    y0 = y0f + rng.uniform(0.0, max(0.0, hf - bh))
    bw = max(8.0, min(wf, wf * rng.uniform(0.7, 1.0)))
    x0 = x0f + rng.uniform(0.0, max(0.0, wf - bw))
    x0 = float(np.clip(x0, 0, W - 8)); y0 = float(np.clip(y0, 0, H - 6))
    return int(x0), int(y0), int(min(bw, W - x0)), int(min(bh, H - y0))

# COMPETITION-WEIGHTED defaults (mirror what actually won leaderboards, kept LIGHT):
#   DFDC 1st place (selimsef): ImageCompression p0.5, color(BC/FancyPCA/HueSat) p0.7,
#   ShiftScaleRotate p0.5, GaussNoise p0.1, GaussianBlur p0.05, ToGray p0.2, and
#   CUTOUT/DROPOUT (dropping parts) credited as a KEY generalization lever. His note:
#   "other complex things did not work so well on the public leaderboard".
#   ID-PAD research (FakeIDet2 + PAD review): DINOv2/foundation models need MINIMAL aug;
#   heavy SYNTHETIC print-capture -> "synthetic utility gap" (model overfits OUR artefact,
#   not the real attack). Crude spatial moire "distorts semantic information".
# => Keep the winner's realistic per-transform degradations (jpeg/downscale/blur/brightness),
#    ENABLE dropout (the winning lever, synergises with our patch-MIL head), lower noise,
#    and REMOVE the aggressive compound 'pcapture_heavy' (crude moire/over-degrade) entirely.
# NOTE: 'orient' (180 deg) and flips excluded — our crops are upright; flipping only makes
#    upside-down text that never occurs at test time. 'moire'/'pcapture(_heavy)' are
#    ablation-only now (available via `groups=` but OFF by default).
DEFAULT_GROUPS = {"degrade", "color", "noise", "geometry", "dropout"}


def build_transform(groups=DEFAULT_GROUPS):
    if not groups or groups == {"none"}:
        return None
    g = set(groups)
    aug = []
    if "degrade" in g:  # DFDC winner: ImageCompression p0.5 (jpeg is the core capture cue)
        aug += [
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, p=1.0),
                A.Downscale(scale_min=0.65, scale_max=0.9, interpolation=cv2.INTER_AREA, p=1.0),
            ], p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.10),  # DFDC blur p0.05 -> keep low
        ]
    if "color" in g:    # DFDC winner: one-of color at p0.7; keep moderate but a touch lighter
        aug += [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.35),
            A.RGBShift(10, 10, 10, p=0.20),
            A.ToGray(p=0.10),  # DFDC uses 0.2; keep lower to preserve chroma forensic cues
        ]
    if "noise" in g:    # DFDC winner: GaussNoise p0.1 (low). ISONoise dropped (can mask sensor cues)
        aug += [A.GaussNoise(var_limit=(5, 30), p=0.12)]
    if "geometry" in g:  # rotate fragility +0.65, perspective +0.46 (recapture/print)
        # fit_output=True keeps the ENTIRE card in frame (no sections warped off-screen);
        # scale<=1.0 (shrink only) so nothing is pushed out by enlargement.
        aug += [
            A.Perspective(scale=(0.02, 0.05), pad_mode=cv2.BORDER_REPLICATE,
                          fit_output=True, p=0.3),
            A.Affine(rotate=(-4, 4), shear=(-2, 2), scale=(0.90, 1.0),
                     mode=cv2.BORDER_REPLICATE, fit_output=True, p=0.3),
        ]
    if "moire" in g:      # moire fragility +0.37; print/screen recapture is in the test
        aug += [A.Lambda(image=_moire, p=0.25)]
    if "pcapture" in g:   # compound print-capture (lighter variant)
        aug += [A.Lambda(image=_pcapture, p=0.35)]
    if "pcapture_heavy" in g:  # softened print-capture (LB-realistic but keeps text legible)
        aug += [A.Lambda(image=_pcapture_heavy, p=0.30)]
    if "orient" in g:     # orientation invariance (OFF by default; crops are upright)
        aug += [A.Lambda(image=_rot180, p=0.45)]
    if "dropout" in g:    # DFDC winner's KEY lever ("dropping parts", GridMask/Severstal).
        # Small, UNBIASED holes (never always over portrait/MRZ) so we don't erase the whole
        # tampered region. Forces many patches to carry forgery signal (synergises w/ patch-MIL).
        aug += [A.CoarseDropout(max_holes=3, min_holes=1,
                                max_height=0.10, max_width=0.10,
                                min_height=0.03, min_width=0.03,
                                fill_value=0, p=0.35)]
    return A.Compose(aug)


def _moire(img, **kw):
    """Cheap moiré/interference overlay to approximate screen/print recapture."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    f = np.random.uniform(0.15, 0.5)
    ang = np.random.uniform(0, np.pi)
    patt = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * f)
    amp = np.random.uniform(4, 12)
    out = img.astype(np.float32) + (patt[..., None] * amp)
    return np.clip(out, 0, 255).astype(np.uint8)


def _glare(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
    g = 1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / (max(h, w) ** 2)
    g = 1 + np.random.uniform(0.1, 0.35) * g[..., None]
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def _pcapture(img, **kw):
    """Compound print-capture: downscale + moire + glare + brightness + JPEG
    (the kind of stacked degradation that matched the LB difficulty)."""
    h, w = img.shape[:2]
    s = np.random.uniform(0.5, 0.8)
    img = cv2.resize(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                                interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    img = _moire(img); img = _glare(img)
    a = 1 + np.random.uniform(-0.12, 0.15); b = np.random.uniform(-10, 15)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(35, 70))])
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB) if ok else img


def _pcapture_heavy(img, **kw):
    """Softened compound print-capture (derived from calibrate_cv.print_capture_HEAVY, which
    matched the public LB, but toned down so document TEXT stays legible — over-degrading
    erases the very cues the model must learn). Moderate downscale + small in-frame
    perspective + gentle moire + glare + brightness + occasional blur + mid-quality JPEG."""
    h, w = img.shape[:2]
    s = np.random.uniform(0.50, 0.72)                       # moderate downscale (was 0.35-0.55)
    img = cv2.resize(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                                interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    d = 0.03 * w                                            # small perspective (was 0.07 -> pushed card off-frame)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:h, 0:w]                             # gentle moire (amp 4-9, was 10-18)
    f = np.random.uniform(0.2, 0.5); ang = np.random.uniform(0, np.pi)
    patt = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * f)
    img = np.clip(img.astype(np.float32) + patt[..., None] * np.random.uniform(4, 9), 0, 255).astype(np.uint8)
    img = _glare(img)                                       # glare
    a = 1 + np.random.uniform(-0.10, 0.12); b = np.random.uniform(-8, 12)  # milder brightness
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if np.random.random() < 0.4:                            # blur only sometimes (was always)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(35, 65))])  # mid-q JPEG (was 20-45)
    return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB) if ok else img


def _rot180(img, **kw):
    return cv2.rotate(img, cv2.ROTATE_180)


# ---- SBI-style source transforms (validated set) for self-blend ----
def _sbi_source(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = patch
    if rng.random() < 0.8:  # brightness/contrast (gentle)
        a = 1 + rng.uniform(-0.05, 0.05); b = rng.uniform(-5, 5)
        p = np.clip(p.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:  # RGBShift (gentle)
        sh = rng.integers(-5, 6, 3)
        p = np.clip(p.astype(np.int16) + sh, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # HueSat (gentle)
        hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-5, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.integers(-10, 11), 0, 255)
        p = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.4:  # Downscale (mild resampling)
        s = rng.uniform(0.75, 0.95); hh, ww = p.shape[:2]
        p = cv2.resize(cv2.resize(p, (max(1, int(ww * s)), max(1, int(hh * s))),
                                  interpolation=cv2.INTER_AREA), (ww, hh),
                       interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.25:  # Sharpen (occasional)
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        p = np.clip(cv2.filter2D(p, -1, k), 0, 255).astype(np.uint8)
    return p


# ---- Portrait localization (face position varies by template: LEFT / RIGHT / centre) ----
try:
    _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if _FACE_CASCADE.empty():
        _FACE_CASCADE = None
except Exception:
    _FACE_CASCADE = None


def detect_face_box(img: np.ndarray):
    """Locate the main portrait as a RELATIVE (rx, ry, rw, rh) box in [0,1], wherever it sits on
    the card (left/right/centre). Expanded from the tight face to head+shoulders so a swap covers
    the whole photo. Returns None if no face is found. Detect ONCE per type (layout is fixed)."""
    if _FACE_CASCADE is None:
        return None
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = min(1.0, 480.0 / max(H, W))
    gs = cv2.resize(g, (max(1, int(W * s)), max(1, int(H * s)))) if s < 1.0 else g
    faces = _FACE_CASCADE.detectMultiScale(gs, scaleFactor=1.1, minNeighbors=4,
                                           minSize=(int(0.08 * gs.shape[1]), int(0.10 * gs.shape[0])))
    if len(faces) == 0:
        return None
    x, y, w, h = (float(v) / s for v in max(faces, key=lambda b: b[2] * b[3]))
    cx, cy = x + w / 2, y + h / 2                        # expand tight face -> portrait (mostly down)
    pw, ph = w * 1.6, h * 2.1; px, py = cx - pw / 2, cy - ph * 0.42
    rx = float(np.clip(px / W, 0, 1)); ry = float(np.clip(py / H, 0, 1))
    return rx, ry, float(np.clip(pw / W, 0.05, 1 - rx)), float(np.clip(ph / H, 0.05, 1 - ry))


_MIN_FACE_AREA = 0.004   # drop degenerate/eye-sized boxes (<0.4% of card = accidental micro-drag)


def _choose_face_subset(faces, rng):
    """Pick WHICH portraits to tamper this image. The MAIN (largest) face is ALWAYS included, so a
    swap always covers the full face -- never a lone tiny-ghost 'eye-sized' swap. The ghost(s) ride
    along: mostly BOTH main+ghost (higher weight, so the model flags a swap even when main==ghost,
    not just main-vs-ghost mismatch), sometimes main-only. Degenerate boxes < _MIN_FACE_AREA are
    ignored entirely. Returns a list of boxes (main first)."""
    faces = [f for f in (faces or []) if f is not None and f[2] * f[3] >= _MIN_FACE_AREA]
    if len(faces) <= 1:
        return faces
    main = max(faces, key=lambda b: b[2] * b[3])
    others = [f for f in faces if f is not main]
    if rng.random() < 0.65:                    # MAIN + all ghosts ("both", higher weight)
        return [main] + others
    if rng.random() < 0.55 or len(others) == 1:  # MAIN only
        return [main]
    return [main, others[int(rng.integers(len(others)))]]   # MAIN + one random ghost


def _blend_region(out, full, src_img, x0, y0, x1, y1, rng, alpha):
    """SBI self-blend of one region (in place on out/full float arrays)."""
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    patch = _sbi_source(src_img[y0:y1, x0:x1], rng)
    dx = int(rng.uniform(-0.03, 0.03) * (x1 - x0)); dy = int(rng.uniform(-0.03, 0.03) * (y1 - y0))
    sc = rng.uniform(0.97, 1.03); ph, pw = patch.shape[:2]
    patch = cv2.warpAffine(patch, np.float32([[sc, 0, dx], [0, sc, dy]]), (pw, ph),
                           borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.30:                               # seam-free variant: Poisson self-blend
        r = _poisson_paste(np.clip(out, 0, 255).astype(np.uint8), patch, x0, y0)
        if r is not None:
            res, m = r
            out[:] = res.astype(np.float32)
            full[:] = np.maximum(full, m)
            return
    region = np.zeros((ph, pw), np.float32)               # geometry of the tampered area (peak ~1)
    fb = max(3, int(0.12 * min(ph, pw)))
    region[fb:ph - fb, fb:pw - fb] = 1.0
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=fb / 1.5)
    blend = region * alpha; b3 = blend[..., None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - b3) + patch.astype(np.float32) * b3
    full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], region)   # supervision = region geometry


def self_blend(img: np.ndarray, rng: np.random.Generator, box=None, faces=None):
    """SBI-style self-blended overlay attack + mask. `faces` (list of portrait boxes) -> blend a
    RANDOM SUBSET (main/ghost/both, weighted to both). `box` -> single portrait. Else a generic
    region. Blend strength raised so the seam is visible-but-plausible (was too subtle)."""
    H, W = img.shape[:2]
    if faces:
        sel = _choose_face_subset(faces, rng)
    elif box is not None and rng.random() < 0.8:
        sel = [box]
    else:
        sel = [None]                                       # generic region (no face prior)
    # largest portrait = MAIN; smaller ones = ghost/secondary (small + already faint -> need a
    # STRONGER blend to be detectable at all, so alpha scales UP as the box gets smaller).
    max_area = max((b[2] * b[3] for b in sel if b is not None), default=0.0)
    out = img.copy().astype(np.float32)
    full = np.zeros((H, W), np.float32)
    for b in sel:
        if b is not None:                                  # portrait region (any position)
            rx, ry, rw, rh = b
            x0 = int(np.clip(rx + rng.uniform(-0.02, 0.02), 0, 1) * W)
            y0 = int(np.clip(ry + rng.uniform(-0.02, 0.02), 0, 1) * H)
            bw = int(np.clip(rw * rng.uniform(0.92, 1.12), 0.05, 1) * W)
            bh = int(np.clip(rh * rng.uniform(0.92, 1.12), 0.05, 1) * H)
            is_ghost = max_area > 0 and (rw * rh) < 0.60 * max_area
            alpha = rng.uniform(0.55, 0.85) if is_ghost else rng.uniform(0.35, 0.62)
        else:
            x0 = int(rng.uniform(0.0, 0.6) * W); y0 = int(rng.uniform(0.0, 0.5) * H)
            bw = int(rng.uniform(0.15, 0.45) * W); bh = int(rng.uniform(0.15, 0.45) * H)
            alpha = rng.uniform(0.35, 0.62)
        _blend_region(out, full, img, x0, y0, min(W, x0 + bw), min(H, y0 + bh), rng, alpha=alpha)
    if full.max() <= 1e-6:
        return img, np.zeros((H, W), np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), full


# ---- ID-specific COMPOSITE synthetic attacks (what ID-card competition winners manufacture) ----
# Rationale: PAD-ID Card 2024/2025 + DeepID winners generate "composite attacks by swapping the
# face OR text area between two ID cards" and text-field inpaint/rewrite. Richer blend boundaries
# than face-only self-blend -> better generalisation to UNSEEN attack types (the hard FREUID case).
def _field_strip(box, W):
    """Horizontal [x_lo, x_hi] of the text/field area = the side OPPOSITE the portrait.
    Face on the LEFT -> fields to its RIGHT; face on the RIGHT -> fields to its LEFT."""
    if box is None:
        return int(0.03 * W), int(0.97 * W)
    rx, rw = box[0], box[2]
    if rx + rw / 2.0 < 0.5:                       # face on left half
        return int((rx + rw + 0.02) * W), int(0.98 * W)
    return int(0.02 * W), int((rx - 0.02) * W)    # face on right half


def _text_band(img, rng, bw_frac, bh_frac, y_lo=0.08, y_hi=0.92, box=None, fields=None):
    """Place a SMALL edit band on a real text field. If manual `fields` (annotated boxes for this
    type) are given, sample INSIDE one -> exact placement (used for sparse-text templates like
    RUS_scanned where auto-detection lands on blank paper). Otherwise fall back to auto-detection:
    band on the field side (opposite the portrait), CENTERED on a real text line found by joining
    characters into horizontal runs (morph close) and thresholding weak rows. Returns (x0,y0,bw,bh)
    or None."""
    H, W = img.shape[:2]
    if fields:
        return _ann_band(fields, W, H, rng, bh_frac)
    sx_lo, sx_hi = _field_strip(box, W)
    if sx_hi - sx_lo < int(0.10 * W):                            # no room on the field side
        return None
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edge = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))       # vertical strokes -> text
    thr = np.percentile(edge[:, sx_lo:sx_hi], 80)                # adaptive: keep the strongest strokes
    b = (edge > max(thr, 1e-3)).astype(np.uint8)
    b[:, :sx_lo] = 0; b[:, sx_hi:] = 0                           # field side only
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,                     # join characters into text-line runs
                         cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))
    rowc = b.sum(axis=1).astype(np.float32)                      # text-pixel count per row -> line peaks
    rowc = np.convolve(rowc, np.ones(5, np.float32) / 5, mode="same")   # smooth over ~a line height
    ys = np.arange(H); lo, hi = int(y_lo * H), int(y_hi * H)
    rowc = np.where((ys >= lo) & (ys < hi), rowc, 0.0)
    if rowc.max() <= 1e-6:
        return None
    rowc[rowc < 0.40 * rowc.max()] = 0.0                        # only CLEAR text lines are eligible
    p = rowc ** 2; sp = p.sum()
    if sp <= 0:
        return None
    cy = int(rng.choice(ys, p=p / sp))                          # center on a real text line
    bh = int(np.clip(rng.uniform(*bh_frac) * H, 6, H - 4))
    y0 = int(np.clip(cy - bh // 2, 0, H - bh))
    strip_w = sx_hi - sx_lo
    present = (b[y0:y0 + bh].sum(axis=0) > 0).astype(np.int8)    # columns containing text in this line
    present[:sx_lo] = 0; present[sx_hi:] = 0
    idx = np.where(present)[0]
    if len(idx) == 0:                                            # (shouldn't happen: row had text)
        bw = int(np.clip(rng.uniform(*bw_frac) * W, 8, strip_w))
        x0 = int(np.clip((sx_lo + sx_hi) // 2 - bw // 2, sx_lo, max(sx_lo, sx_hi - bw)))
        return x0, y0, min(bw, W - x0), min(bh, H - y0)
    runs = []; a = prev = int(idx[0])                            # contiguous runs = fields/words
    for k in idx[1:]:
        if k == prev + 1: prev = int(k)
        else: runs.append((a, prev)); a = prev = int(k)
    runs.append((a, prev))
    runs = [(a, bb) for a, bb in runs if bb - a + 1 >= 10] or runs
    lens = np.array([bb - a + 1 for a, bb in runs], np.float32)
    a, bb = runs[int(rng.choice(len(runs), p=lens / lens.sum()))]   # pick a run (weighted by length)
    run_len = bb - a + 1
    bw = int(np.clip(rng.uniform(*bw_frac) * W, 8, min(strip_w, run_len)))
    x0 = a if run_len <= bw else a + int(rng.integers(0, run_len - bw + 1))   # band INSIDE the run
    return x0, y0, min(bw, W - x0), min(bh, H - y0)


def _poisson_paste(img, patch, x0, y0):
    """Poisson (seamless) clone `patch` into `img` at (x0,y0) — NO alpha seam. Mimics the
    DeepID private set's hardest attack class (Poisson-blended edits beat all 26 teams);
    breaks the feathered-seam monoculture so the seg head can't just learn soft boundaries.
    Returns (out_uint8, full_mask) or None on cv2 failure (caller falls back to feather)."""
    H, W = img.shape[:2]; ph, pw = patch.shape[:2]
    if ph < 8 or pw < 8 or x0 < 1 or y0 < 1 or x0 + pw > W - 1 or y0 + ph > H - 1:
        return None
    try:
        out = cv2.seamlessClone(np.ascontiguousarray(patch), np.ascontiguousarray(img),
                                np.full((ph, pw), 255, np.uint8),
                                (x0 + pw // 2, y0 + ph // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        return None
    fb = max(2, int(0.08 * min(ph, pw)))
    region = np.zeros((ph, pw), np.float32)
    region[fb:ph - fb, fb:pw - fb] = 1.0
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=max(1.0, fb / 1.5))
    full = np.zeros((H, W), np.float32)
    full[y0:y0 + ph, x0:x0 + pw] = region
    return out, full


def _feather_paste(img, patch, x0, y0, rng, alpha=(0.5, 0.9), fb_frac=0.12, poisson_p=0.45):
    """Blend `patch` into `img` at (x0,y0): with prob `poisson_p` a seam-free Poisson clone,
    else a feathered (soft-boundary) alpha paste. Returns (composited_uint8, full_size_mask)."""
    if poisson_p > 0 and rng.random() < poisson_p:
        r = _poisson_paste(img, patch, x0, y0)
        if r is not None:
            return r
    H, W = img.shape[:2]; ph, pw = patch.shape[:2]
    x1, y1 = x0 + pw, y0 + ph
    mask = np.zeros((ph, pw), np.float32); fb = max(2, int(fb_frac * min(ph, pw)))
    if ph - 2 * fb > 0 and pw - 2 * fb > 0:
        mask[fb:ph - fb, fb:pw - fb] = 1.0
    else:
        mask[:] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, fb / 1.5)) * rng.uniform(*alpha)
    out = img.copy().astype(np.float32); m3 = mask[..., None]
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m3) + patch.astype(np.float32) * m3
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = mask
    return np.clip(out, 0, 255).astype(np.uint8), full


def _swap_one(img, donor, x0, y0, bw, bh, rng):
    """Paste the same-relative region from the donor card at (x0,y0,bw,bh). Returns (out, mask)."""
    H, W = img.shape[:2]
    bw = min(bw, W - x0); bh = min(bh, H - y0)
    if bw < 8 or bh < 8:
        return img, np.zeros((H, W), np.float32)
    dh, dw = donor.shape[:2]
    dx0 = int(x0 / W * dw); dy0 = int(y0 / H * dh)
    dpatch = donor[dy0:min(dh, dy0 + int(bh / H * dh) + 1), dx0:min(dw, dx0 + int(bw / W * dw) + 1)]
    if dpatch.size == 0:
        return img, np.zeros((H, W), np.float32)
    dpatch = _sbi_source(cv2.resize(dpatch, (bw, bh), interpolation=cv2.INTER_LINEAR), rng)
    return _feather_paste(img, dpatch, x0, y0, rng, alpha=(0.55, 0.90), fb_frac=0.12)


def region_swap(img: np.ndarray, donor: np.ndarray, rng: np.random.Generator,
                box=None, fields=None, faces=None):
    """Cross-card composite: paste region(s) from ANOTHER genuine card of the SAME template
    (crop-and-replace). `faces` (list) -> PORTRAIT swap of a random subset (main/ghost/both,
    weighted to both). `box` -> single portrait. `fields` -> DATA-field branch. Same-relative
    donor crop keeps face->face / field->field (no chip-on-face). Returns (fake_uint8, mask)."""
    H, W = img.shape[:2]
    portraits = faces if faces else ([box] if box is not None else None)
    if portraits and rng.random() < 0.7:                    # PORTRAIT swap (subset of faces)
        sel = _choose_face_subset(portraits, rng)
        out = img; full = np.zeros((H, W), np.float32)
        for b in sel:
            rx, ry, rw, rh = b
            out, m = _swap_one(out, donor, int(rx * W), int(ry * H), int(rw * W), int(rh * H), rng)
            full = np.maximum(full, m)
        if full.max() <= 1e-6:
            return img, np.zeros((H, W), np.float32)
        return out, full
    # DATA-FIELD: a small field on the text side (annotated field if provided)
    band = _text_band(img, rng, (0.15, 0.33), (0.04, 0.07), y_lo=0.12, y_hi=0.92, box=box, fields=fields)
    if band is None:
        return img, np.zeros((H, W), np.float32)             # no field room -> skip (don't fake a blank)
    x0, y0, bw, bh = band
    return _swap_one(img, donor, x0, y0, bw, bh, rng)


def text_field_edit(img: np.ndarray, rng: np.random.Generator, box=None, fields=None):
    """Simulate inpaint/rewrite of a TEXT FIELD (inpaint_and_rewrite family): land on a REAL text
    line (never blank space, never the portrait `box`), copy a neighbouring same-line band (shifted)
    with a photometric shift and blend it over the field -> a local tamper boundary where a value
    would be edited. `fields` (manual annotations) place the edit exactly on a real field when given.
    Returns (fake_uint8, mask)."""
    H, W = img.shape[:2]
    band = _text_band(img, rng, (0.12, 0.28), (0.03, 0.055), y_lo=0.10, y_hi=0.92, box=box, fields=fields)
    if band is None:
        return img, np.zeros((H, W), np.float32)
    x0, y0, bw, bh = band
    if bw < 8 or bh < 8:
        return img, np.zeros((H, W), np.float32)
    sx = int(np.clip(x0 + rng.uniform(-0.18, 0.18) * W, 0, W - bw))   # SAME line, shifted -> edited chars
    sy = int(np.clip(y0 + rng.uniform(-0.03, 0.03) * H, 0, H - bh))
    patch = _sbi_source(img[sy:sy + bh, sx:sx + bw].copy(), rng)
    return _feather_paste(img, patch, x0, y0, rng, alpha=(0.60, 0.95), fb_frac=0.15)


_FONT_PATHS = None


def _font_paths():
    global _FONT_PATHS
    if _FONT_PATHS is None:
        cand = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\cour.ttf",
                r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\calibri.ttf"]
        _FONT_PATHS = [p for p in cand if os.path.exists(p)]
    return _FONT_PATHS


def erase_retype(img: np.ndarray, rng: np.random.Generator, box=None, fields=None):
    """Erase-and-retype a text field — mechanically what real document-fraud tools do:
    inpaint the ink strokes (cv2.inpaint, seam-free), then render a NEW value with a system
    font at matched size/color. Forensic cue = font/kerning/inpaint-texture mismatch, NOT an
    alpha seam (complements the blend-based attacks). Returns (fake_uint8, mask)."""
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]
    band = _text_band(img, rng, (0.12, 0.30), (0.035, 0.06), y_lo=0.10, y_hi=0.92,
                      box=box, fields=fields)
    if band is None:
        return img, np.zeros((H, W), np.float32)
    x0, y0, bw, bh = band
    if bw < 14 or bh < 9:
        return img, np.zeros((H, W), np.float32)
    roi = img[y0:y0 + bh, x0:x0 + bw]
    g = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    dark = g < np.percentile(g, 35)                       # ink strokes (dark text on light card)
    if dark.mean() < 0.01:                                # blank band -> nothing to rewrite
        return img, np.zeros((H, W), np.float32)
    ink = roi.reshape(-1, 3)[dark.reshape(-1)]
    color = tuple(int(v) for v in np.percentile(ink, 30, axis=0))
    strokes = cv2.dilate(dark.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(roi, strokes, 3, cv2.INPAINT_TELEA)
    n = int(rng.integers(4, 13))
    ab = "0123456789" if rng.random() < 0.5 else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    txt = "".join(ab[int(rng.integers(len(ab)))] for _ in range(n))
    if ab[0] == "0" and n >= 8 and rng.random() < 0.4:
        txt = f"{txt[:2]}/{txt[2:4]}/{txt[4:8]}"          # date-like value
    pil = Image.fromarray(clean); dr = ImageDraw.Draw(pil)
    size = max(8, int(bh * rng.uniform(0.55, 0.80)))
    fonts = _font_paths()
    try:
        font = (ImageFont.truetype(fonts[int(rng.integers(len(fonts)))], size)
                if fonts else ImageFont.load_default())
        while dr.textlength(txt, font=font) > bw - 4 and len(txt) > 3:
            txt = txt[:-1]
        tw = dr.textlength(txt, font=font)
    except Exception:
        font = ImageFont.load_default(); tw = min(bw - 4, 6 * len(txt))
    tx = int(rng.uniform(1, max(2.0, bw - tw - 2)))
    ty = max(0, (bh - size) // 2 + int(rng.uniform(-2, 2)))
    dr.text((tx, ty), txt, fill=color, font=font)
    out = img.copy(); out[y0:y0 + bh, x0:x0 + bw] = np.asarray(pil)
    fb = max(2, int(0.10 * bh))
    region = np.zeros((bh, bw), np.float32)
    region[fb:bh - fb, fb:bw - fb] = 1.0
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=max(1.0, fb / 1.5))
    full = np.zeros((H, W), np.float32); full[y0:y0 + bh, x0:x0 + bw] = region
    return out, full


def synth_fake(img, rng, donor=None):
    """Dispatch a random synthetic ID attack. If `donor` (another genuine card) is given, cross-card
    region_swap is available; else falls back to self-blend / text-field edit. Returns (fake, mask)."""
    r = rng.random()
    if donor is not None and r < 0.34:
        return region_swap(img, donor, rng)
    if r < 0.67:
        return text_field_edit(img, rng)
    return self_blend(img, rng)
