"""Train DINOv2-L + LoRA fraud detector. Validation regimes:
  --holdout TYPE   leave-one-doc-type-out (north-star generalization)
  --fold K         stratified fold K as val (in-domain)
Reports exact FREUID each epoch; saves best checkpoint + val OOF.

Example:
  python src/train.py --holdout MOZAMBIQUE/DL --epochs 4 --bs 16 --accum 2 \
      --res 322x518 --aug strong --sbi 0.25 --tag v1
"""
from __future__ import annotations
import os, sys, time, argparse, math, json
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT, DATA, train_path, letterbox, to_tensor_norm, load_rgb
from augment import (build_transform, self_blend, region_swap, text_field_edit, erase_retype,
                     load_field_annotations, DEFAULT_GROUPS)
from model import FreuidModel
from freuid_metric import freuid_score

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
CKPT_DIR = os.path.join(ROOT, "checkpoints"); os.makedirs(CKPT_DIR, exist_ok=True)
OOF_DIR = os.path.join(ROOT, "oof"); os.makedirs(OOF_DIR, exist_ok=True)


def parse_groups(s):
    if s in ("core", "", None): return DEFAULT_GROUPS
    if s == "none": return {"none"}
    return set(x.strip() for x in s.split(",") if x.strip())


def load_image(source, idv, path):
    """Dispatch FREUID (id -> train_path) vs IDNet (absolute path already in the index)."""
    return load_rgb(train_path(idv)) if source == "freuid" else load_rgb(path)


class TrainDS(Dataset):
    """attacks='self_blend' (default, cv1/cv2 recipe): untargeted generic self-blend only.
    attacks='full': the annotation-driven suite (region_swap/text_field_edit/erase_retype,
    placed via annotations/type_fields.json) that cue2 used -- same fraud-generation realism,
    layered onto the SAME otherwise-unchanged FREUID-only recipe (one lever at a time)."""
    def __init__(self, ids, labels, types, H, W, groups=DEFAULT_GROUPS, sbi=0.0, attacks="self_blend",
                 sources=None, paths=None):
        self.ids = ids; self.y = np.asarray(labels, np.float32); self.types = np.asarray(types)
        self.src = np.asarray(sources) if sources is not None else np.full(len(ids), "freuid")
        self.path = np.asarray(paths) if paths is not None else np.full(len(ids), "")
        self.H, self.W = H, W; self.tf = build_transform(groups); self.sbi = sbi
        self.attacks = attacks
        self.gen_idx = np.where(self.y == 0)[0]
        self.type_fields, self.type_faces = {}, {}
        self.type_pool = {}
        if attacks == "full":
            self.type_fields, self.type_faces = load_field_annotations(
                os.path.join(ROOT, "annotations", "type_fields.json"))
            for idx in self.gen_idx:
                self.type_pool.setdefault(self.types[idx], []).append(int(idx))

    def __len__(self): return len(self.ids)

    def _attack_full(self, img, rng, typ):
        faces = self.type_faces.get(typ) or []
        box = faces[0] if faces else None
        fields = self.type_fields.get(typ)
        r = rng.random(); fake = None; m = None
        if r < 0.28:
            pool = self.type_pool.get(typ)
            if pool and len(pool) > 1:
                j = int(pool[rng.integers(len(pool))])
                try:
                    donor = load_image(self.src[j], self.ids[j], self.path[j])
                    fake, m = region_swap(img, donor, rng, box=box, fields=fields, faces=faces)
                except Exception:
                    fake = m = None
        elif r < 0.53:
            fake, m = text_field_edit(img, rng, box=box, fields=fields)
        elif r < 0.76:
            try:
                fake, m = erase_retype(img, rng, box=box, fields=fields)
            except Exception:
                fake = m = None
        if m is None or float(m.max()) <= 1e-6:      # attack couldn't apply -> guaranteed self-blend
            fake, m = self_blend(img, rng, box=box, faces=faces)
        return fake

    def __getitem__(self, i):
        rng = np.random.default_rng()
        # augment on full-res then letterbox (GPU-bound anyway; full-res aug is more
        # accurate -- resize-first measured worse: held-out Mauritius 0.0038 -> 0.0136).
        img = load_image(self.src[i], self.ids[i], self.path[i]); y = float(self.y[i])
        if y == 0.0 and self.sbi > 0 and rng.random() < self.sbi:
            img = (self._attack_full(img, rng, self.types[i]) if self.attacks == "full"
                  else self_blend(img, rng)[0])
            y = 1.0
        if self.tf is not None:
            img = self.tf(image=img)["image"]
        img = letterbox(img, self.H, self.W)
        return to_tensor_norm(img), y


class ValDS(Dataset):
    def __init__(self, ids, labels, H, W):
        self.ids = ids; self.y = np.asarray(labels, np.float32); self.H, self.W = H, W

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img = letterbox(load_rgb(train_path(self.ids[i])), self.H, self.W)
        return to_tensor_norm(img), float(self.y[i])


def corrupt_fixed(img, seed):
    """Deterministic moderate test-like corruption (the high-impact ones from the
    probe: resample + brightness + JPEG) -> a public-LB proxy for validation."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]; s = 0.7
    img = cv2.resize(cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    a = 1 + rng.uniform(-0.10, 0.15); b = rng.uniform(-8, 12)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    q = int(rng.integers(45, 75))
    _, e = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(e, 1), cv2.COLOR_BGR2RGB)


class IDNetEvalDS(Dataset):
    """Fixed HELDOUT-IDNet sample (unseen countries, clean/uncorrupted) -- the trustworthy
    generalization gauge (zero leakage risk, unlike in-domain FREUID val which we proved
    can rank checkpoints backwards). Selection criterion for the FREUID+IDNet mixed recipe."""
    def __init__(self, df, H, W):
        self.paths = df.path.values; self.y = df.label.values.astype(np.float32)
        self.H, self.W = H, W

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        img = letterbox(load_rgb(self.paths[i]), self.H, self.W)
        return to_tensor_norm(img), float(self.y[i])


class HardValDS(Dataset):
    """Val set with deterministic test-like corruption (public-LB proxy)."""
    def __init__(self, ids, labels, H, W):
        self.ids = ids; self.y = np.asarray(labels, np.float32); self.H, self.W = H, W

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img = corrupt_fixed(load_rgb(train_path(self.ids[i])), i)
        return to_tensor_norm(letterbox(img, self.H, self.W)), float(self.y[i])


def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets == 1, p, 1 - p)
    at = torch.where(targets == 1, alpha, 1 - alpha)
    return (at * (1 - pt).pow(gamma) * ce).mean()


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); ps, ys = [], []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(model(x)).float().cpu())
        ys.append(y)
    p = torch.cat(ps).numpy(); y = torch.cat(ys).numpy()
    f, a, ap = freuid_score(y, p)
    return f, a, ap, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=str, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--res", type=str, default="322x518")
    ap.add_argument("--aug", type=str, default="core",
                    help="'core', 'none', or comma list: degrade,color,noise,geometry,moire,dropout")
    ap.add_argument("--sbi", type=float, default=0.25)
    ap.add_argument("--attacks", type=str, default="self_blend", choices=["self_blend", "full"],
                    help="self_blend=cv1/cv2 recipe; full=cue2-style annotation-driven suite")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_lora", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", type=str, default="v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=250, help="periodic 'last' ckpt every N opt-steps")
    ap.add_argument("--hardval", type=int, default=1, help="also eval corrupted val (public-LB proxy) & select by it")
    ap.add_argument("--idnet_countries", type=str,
                    default="ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,LVA_scanned,RUS_scanned,SRB_scanned",
                    help="IDNet countries mixed into TRAINING (comma list, '' to disable IDNet mixing entirely)")
    ap.add_argument("--heldout_idnet", type=str, default="EST_scanned,SVK_scanned",
                    help="IDNet countries used ONLY for the generalization eval, never trained on")
    ap.add_argument("--lim_idn", type=int, default=56000, help="cap on IDNet training rows (balanced by label)")
    ap.add_argument("--idn_val_n", type=int, default=4000, help="HELDOUT-IDNet eval sample size")
    ap.add_argument("--select_on", type=str, default="idnet", choices=["idnet", "hard", "clean"],
                    help="checkpoint-selection criterion. idnet = the ONLY leak-free signal we have "
                         "(in-domain FREUID val proved unreliable for ranking checkpoints, even the "
                         "'hard'/corrupted proxy -- see ROADMAP 2026-07-08 entries)")
    args = ap.parse_args()
    H, W = (int(v) for v in args.res.lower().split("x"))

    df = pd.read_csv(os.path.join(ROOT, "splits", "folds.csv"))
    if args.holdout:
        val_mask = df.type == args.holdout; vname = "hold_" + args.holdout.replace("/", "_")
    elif args.fold is not None:
        val_mask = df.strat_fold == args.fold; vname = f"fold{args.fold}"
    else:
        raise SystemExit("specify --holdout or --fold")
    tr, va = df[~val_mask].copy(), df[val_mask].copy()
    if args.limit:
        tr = tr.sample(args.limit, random_state=0); va = va.sample(min(args.limit, len(va)), random_state=0)
    tr["source"] = "freuid"; tr["path"] = ""

    idn_val = None
    if args.idnet_countries:
        idn = pd.read_csv(os.path.join(ROOT, "external", "idnet_cropped_index.csv"))
        idn_tr = idn[idn.type.isin(args.idnet_countries.split(","))]
        if args.lim_idn:
            idn_tr = idn_tr.groupby("label", group_keys=False).apply(
                lambda g: g.sample(min(len(g), args.lim_idn // 2), random_state=0))
        cols = ["id", "label", "type", "source", "path"]
        tr = pd.concat([tr[cols], idn_tr[cols]], ignore_index=True)

        idn_ho = idn[idn.type.isin(args.heldout_idnet.split(","))]
        idn_val = idn_ho.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.idn_val_n // 2), random_state=1))
    groups = parse_groups(args.aug)
    print(f"[{vname}] train={len(tr)} (freuid {(tr.source=='freuid').sum()}, idnet {(tr.source=='idnet').sum()}) "
          f"val={len(va)}  HELDOUT-IDNet={0 if idn_val is None else len(idn_val)}  "
          f"res={H}x{W} aug_groups={sorted(groups)} sbi={args.sbi} attacks={args.attacks} "
          f"select_on={args.select_on}")

    tds = TrainDS(tr.id.tolist(), tr.label.values, tr.type.tolist(), H, W, groups, args.sbi, args.attacks,
                 sources=tr.source.tolist(), paths=tr.path.tolist())
    vds = ValDS(va.id.tolist(), va.label.values, H, W)
    tdl = DataLoader(tds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    vdl = DataLoader(vds, batch_size=args.eval_bs, shuffle=False, num_workers=8, pin_memory=True)
    hvdl = (DataLoader(HardValDS(va.id.tolist(), va.label.values, H, W),
                       batch_size=args.eval_bs, shuffle=False, num_workers=8, pin_memory=True)
            if args.hardval else None)
    idl = (DataLoader(IDNetEvalDS(idn_val, H, W), batch_size=args.eval_bs, shuffle=False,
                      num_workers=8, pin_memory=True)
           if idn_val is not None else None)

    model = FreuidModel(lora_r=args.lora_r).cuda()
    nt = model.trainable_params()
    print(f"trainable {nt/1e6:.2f}M  lora_modules={model.n_lora}")
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    opt = torch.optim.AdamW([
        {"params": head_p, "lr": args.lr_head, "weight_decay": args.wd},
        {"params": lora_p, "lr": args.lr_lora, "weight_decay": 0.0},
    ])
    steps = max(1, len(tdl) // args.accum) * args.epochs
    warm = max(1, int(0.05 * steps))
    def lr_at(s): return s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler()
    last_path = os.path.join(CKPT_DIR, f"{args.tag}_{vname}_last.pt")
    best_path = os.path.join(CKPT_DIR, f"{args.tag}_{vname}.pt")
    best = {"freuid": 1e9}; gstep = 0; start_ep = 0

    def save_state(ep):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "ep": ep, "gstep": gstep, "best": best, "args": vars(args)}, last_path)

    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep = ck["ep"] + 1; gstep = ck["gstep"]; best = ck["best"]
        print(f"resumed {last_path} -> start ep{start_ep} (best={best.get('freuid'):.4f})", flush=True)

    for ep in range(start_ep, args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(tdl):
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = focal_bce(model(x), y) / args.accum
            scaler.scale(loss).backward(); run += loss.item() * args.accum
            if (it + 1) % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); gstep += 1
                if args.save_every and gstep % args.save_every == 0:
                    save_state(ep - 1)  # mid-epoch crash -> resume redoes this epoch (no data skipped)
            if (it + 1) % (args.accum * 25) == 0:
                ips = (it + 1) * args.bs / (time.time() - t0)
                print(f"  ep{ep} it{it+1}/{len(tdl)} loss={run/(it+1):.4f} lr={sched.get_last_lr()[0]:.2e} {ips:.1f}img/s", flush=True)
        train_t = time.time() - t0; tput = len(tds) / train_t
        torch.cuda.empty_cache()
        f, a, apc, p = evaluate(model, vdl)
        hf = ha = hap = None
        if args.hardval:
            hf, ha, hap, _ = evaluate(model, hvdl)
        idf = ida = idapc = None
        if idl is not None:
            idf, ida, idapc, _ = evaluate(model, idl)
        torch.cuda.empty_cache()
        # SELECT ON HELDOUT-IDNet: the in-domain FREUID val (clean AND corrupted "hard" proxy)
        # proved unreliable for RANKING checkpoints (2026-07-08: fold2's best-ever local score
        # scored WORSE on the real LB than fold1's mediocre one). IDNet has zero leakage risk.
        sel_map = {"idnet": idf, "hard": hf, "clean": f}
        sel = sel_map[args.select_on] if sel_map[args.select_on] is not None else f
        msg = f"== ep{ep} clean FREUID={f:.4f}(AUC={1-a:.4f},APCER@1%={apc:.4f})"
        if args.hardval:
            msg += f" | HARD FREUID={hf:.4f}(AUC={1-ha:.4f},APCER@1%={hap:.4f})"
        if idl is not None:
            msg += f" | HELDOUT-IDNet FREUID={idf:.4f}(AUC={1-ida:.4f},APCER@1%={idapc:.4f})"
        msg += f" | train={train_t:.0f}s({tput:.1f}img/s)"
        print(msg, flush=True)
        save_state(ep)
        if sel < best["freuid"]:
            best = {"freuid": float(sel), "clean": float(f), "hard": (float(hf) if hf is not None else None),
                    "idnet": (float(idf) if idf is not None else None),
                    "audet": float(a), "apcer": float(apc), "epoch": ep}
            torch.save({"model": model.state_dict(), "args": vars(args), "val": best}, best_path)
            np.save(os.path.join(OOF_DIR, f"valpred_{args.tag}_{vname}.npy"), p)
            np.save(os.path.join(OOF_DIR, f"valid_{args.tag}_{vname}.npy"), va.id.values)
    print(f"BEST {vname}: sel(FREUID)={best['freuid']:.4f} clean={best.get('clean'):.4f} "
          f"hard={best.get('hard')} (ep{best.get('epoch')})")
    with open(os.path.join(CKPT_DIR, f"{args.tag}_{vname}.json"), "w") as fjson:
        json.dump(best, fjson, indent=2)


if __name__ == "__main__":
    main()
