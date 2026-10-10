#!/usr/bin/env python3
"""Kit 2 training stages: extra seeds for the original ten architectures and
eight new architectures at three seeds, all with the run-1 protocol.

  seeds     SPECS (the ten run-1 ImageNet-pretrained architectures) at seeds
            1 and 2 -> ckpt_seeds/, thresholds_seeds.json, scored as groups
            pretrained_s1 and pretrained_s2. Restores run 1 first; retrains
            seed 0 into ckpt_pretrained/ if it is missing, and recreates
            ckpt_kvasir/ with kit_run.stage_kvasir if that is missing
            (unless RETRAIN_MISSING=0).
  modern2   SPECS2 (eight new architectures) at seeds 0, 1, 2 ->
            ckpt_modern2/, thresholds_modern2.json, scored as groups
            modern2_s0, modern2_s1, modern2_s2.

Both stages train through the single function train_model(), which is a line
by line copy of the training part of kit_run.stage_pretrained. The ONLY
intended protocol difference is the seed: torch.manual_seed(seed) before the
model is built and np.random.default_rng(seed) for batch order and
augmentation (run 1 used 0 for both; seed 0 here reproduces run 1).

Additions that do not change what is trained or how it is scored:
  * checkpoints and JSON files are written atomically (tmp file + rename), so
    a run killed mid-write never leaves a truncated file that a resumed run
    would treat as finished;
  * a trained model whose weights contain NaN/Inf is treated as a failure
    and not saved (it is logged; run 1 would have saved and scored it);
  * the mean training loss is logged every 10 epochs and stored in the report;
  * the pretrained-weight source (HF repo / timm tag) and library versions are
    stored in the report, so a change of timm default tag is visible;
  * building the model is attempted twice (re-seeded before each attempt) to
    survive a transient download error;
  * files that must not be reused are moved (never deleted) to OUT/stale/ (logged):
    score files of a group are moved when they
    are unreadable, or when they lack a model that now exists, or contain a
    model that was (re)trained in this invocation, so that a resumed run
    never keeps scores from an incomplete or outdated model set.

Each model is trained in its own try/except; failures are logged with a
traceback and the stage continues. A group is scored with whatever models
exist. A stage that had any failure raises at the end, so no DONE_ file is
written for it and rerunning it retries only what is missing.

    OUT=... [PREV_OUT=...] [DRY=1] python k2_train.py seeds modern2
"""
import gc
import glob
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import k2_common as C                                             # noqa: E402
import kit_run as K                                               # noqa: E402
from woundbench.postproc import postprocess_prob                  # noqa: E402

OUT, DRY, DEV, log = C.OUT, C.DRY, C.DEV, C.log
BATCH = 16
LOG_EVERY = 10


# ------------------------------------------------------------ small helpers

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _aside(path, why):
    """Move a file out of the way (never delete), logged."""
    rel = os.path.relpath(path, OUT)
    dst = os.path.join(OUT, "stale", rel + f".{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(path, dst)
    log(f"  moved {rel} to {os.path.relpath(dst, OUT)} ({why})")


def _load_json(fname):
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except Exception:
        _aside(p, "unreadable JSON")
        return {}


def _save_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _json_set(fname, key, value):
    d = _load_json(fname)
    d[key] = value
    _save_json_atomic(os.path.join(OUT, fname), d)


def _disk():
    import shutil
    return f"{shutil.disk_usage(OUT).free / 1e9:.1f} GB free in OUT"


def _versions():
    import segmentation_models_pytorch as smp
    v = {"torch": torch.__version__, "smp": smp.__version__}
    try:
        import timm
        v["timm"] = timm.__version__
    except Exception:
        pass
    return v


def _weights_source(m, enc, weights):
    """Where the encoder weights came from (for the report only)."""
    if weights is None:
        return None
    try:
        if enc.startswith("tu-"):
            cfg = m.encoder.model.pretrained_cfg
            return {"hf_hub_id": cfg.get("hf_hub_id"), "tag": cfg.get("tag")}
        from segmentation_models_pytorch.encoders import encoders
        return dict(encoders[enc]["pretrained_settings"][weights])
    except Exception as e:
        return {"unknown": repr(e)[:200]}


# --------------------------------------------------------------- training data

_DATA = {}


def train_data():
    """Exactly kit_run.stage_pretrained's tensors: images through pt_input
    (letterbox 256, INTER_AREA, ImageNet normalisation), masks letterboxed as
    float32 with INTER_NEAREST, channel axis added. Built once per process."""
    if "d" not in _DATA:
        (Xtr, Ytr), (Xsw, Ysw), _ = K.fuseg_split()
        XT = torch.stack([K.pt_input(x)[0] for x in Xtr])
        YT = torch.stack([torch.from_numpy(K.letterbox(y.astype(np.float32), K.PT_SIZE,
                          cv2.INTER_NEAREST)[0])[None] for y in Ytr])
        _DATA["d"] = (XT, YT, Xsw, Ysw)
        log(f"  training data: {len(XT)} FUSeg images, {len(Xsw)} sweep images")
    return _DATA["d"]


def protocol():
    return {"epochs": 2 if DRY else int(os.environ.get("PT_EPOCHS", "40")),
            "weights": None if DRY else "imagenet"}


# ------------------------------------------------------------ the one trainer

def train_model(name, arch, enc, kw, seed, ck_dir, thr_file):
    """Train (or resume) one model with the run-1 protocol and sweep its
    threshold. Mirrors kit_run.stage_pretrained step by step; the only
    deliberate difference is `seed` in torch.manual_seed and default_rng.
    Returns True if the checkpoint or threshold was (re)made now. Raises on
    failure (the caller logs and continues)."""
    import segmentation_models_pytorch as smp
    from woundbench.learned import dice_bce_loss
    XT, YT, Xsw, Ysw = train_data()
    P = protocol()
    epochs, weights = P["epochs"], P["weights"]
    os.makedirs(ck_dir, exist_ok=True)
    path = os.path.join(ck_dir, name + ".pt")
    thr = _load_json(thr_file)
    if os.path.exists(path) and name in thr:
        return False

    m = None
    for attempt in (1, 2):
        try:
            torch.manual_seed(seed)                    # run 1: torch.manual_seed(0)
            m = getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3,
                                   classes=1, **kw).to(DEV)
            break
        except Exception as e:
            if attempt == 2:
                raise
            log(f"  {name}: building the model failed ({repr(e)[:300]}); retrying in 20s")
            time.sleep(20)

    seconds, losses, nonfinite = None, [], 0
    if os.path.exists(path):
        try:
            m.load_state_dict(torch.load(path, map_location=DEV))
            log(f"  {name}: checkpoint exists, threshold missing; sweeping only")
        except Exception as e:
            log(f"  {name}: existing checkpoint unreadable ({repr(e)[:200]})")
            _aside(path, "unreadable checkpoint")
            # rebuild from the same seed so the retrained model follows the protocol
            del m
            _free()
            torch.manual_seed(seed)
            m = getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3,
                                   classes=1, **kw).to(DEV)
    if not os.path.exists(path):
        t0 = time.time()
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        rng = np.random.default_rng(seed)              # run 1: default_rng(0)
        scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
        m.train()
        for ep in range(epochs):
            order = rng.permutation(len(XT))
            ep_loss = []
            for i in range(0, len(order), BATCH):
                idx = order[i:i + BATCH]
                if len(idx) < 2:
                    continue
                xb, yb = XT[idx].to(DEV), YT[idx].to(DEV)
                if rng.random() < 0.5:
                    xb, yb = xb.flip(3), yb.flip(3)
                if rng.random() < 0.5:
                    xb, yb = xb.flip(2), yb.flip(2)
                k = int(rng.integers(4))
                xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (2, 3))
                opt.zero_grad()
                with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                    loss = dice_bce_loss(m(xb).float(), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                lv = float(loss.detach())
                if np.isfinite(lv):
                    ep_loss.append(lv)
                else:
                    nonfinite += 1
            sched.step()
            losses.append(float(np.mean(ep_loss)) if ep_loss else None)
            if (ep + 1) % LOG_EVERY == 0 or ep + 1 == epochs:
                log(f"  {name}: epoch {ep + 1}/{epochs} mean loss {losses[-1]}"
                    f"{f', {nonfinite} non-finite losses so far' if nonfinite else ''}")
        seconds = time.time() - t0
        sd = m.state_dict()
        bad = [k for k, v in sd.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            raise RuntimeError(f"{name}: non-finite weights after training "
                               f"({len(bad)} tensors, e.g. {bad[:3]}); not saved")
        tmp = path + ".tmp"
        try:
            torch.save(sd, tmp)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):          # e.g. disk full: do not leave a partial file
                os.remove(tmp)
            raise
        del opt, sched, scaler, sd
        log(f"  {name}: trained {epochs} epochs in {seconds:.0f}s ({_disk()})")
    m.eval()
    t, d = K.sweep(lambda x, m=m: K.pt_prob(m, x), Xsw, Ysw)
    _json_set(thr_file, name, t)
    rep = _load_json("train_report_k2.json").get(name, {})
    rep.update({"sweep_dice": d, "threshold": t, "seed": seed, "arch": arch,
                "encoder": enc, "kwargs": kw, "group_dir": os.path.basename(ck_dir),
                "epochs": epochs, "encoder_weights": weights, "n_train": len(XT),
                "n_sweep": len(Xsw), "dry": DRY, "device": str(DEV),
                "weights_source": _weights_source(m, enc, weights),
                "versions": _versions()})
    if seconds is not None:
        rep.update({"seconds": seconds, "epoch_loss": losses,
                    "nonfinite_loss_steps": nonfinite})
    else:
        rep.setdefault("seconds", None)
    _json_set("train_report_k2.json", name, rep)
    log(f"  {name}: threshold {t:.2f}, held-out Dice {d:.3f}")
    del m
    return True


def train_all(jobs, ck_dir, thr_file):
    """jobs: list of (name, arch, enc, kwargs, seed). Each in its own
    try/except. Returns (fresh names, failed names)."""
    fresh, failed = set(), []
    for name, arch, enc, kw, seed in jobs:
        try:
            if train_model(name, arch, enc, kw, seed, ck_dir, thr_file):
                fresh.add(name)
        except Exception:
            failed.append(name)
            log(f"  {name}: FAILED\n{traceback.format_exc()}")
        _free()                     # after the except block: traceback frames released
    return fresh, failed


# ------------------------------------------------------------------ scoring

def _invalidate(group, names, fresh, sets):
    """Move aside score files that a resumed run must not keep."""
    od = os.path.join(OUT, "scores", group)
    for _, sname, _ in sets:
        p = os.path.join(od, sname + ".json")
        if not os.path.exists(p):
            continue
        try:
            old = json.load(open(p))["names"]
        except Exception:
            _aside(p, "unreadable score file")
            continue
        new = set(names) - set(old)
        redone = set(old) & fresh
        if new or redone:
            _aside(p, f"models added {sorted(new)} or retrained {sorted(redone)}")
        elif set(old) != set(names):
            log(f"  WARNING {group}/{sname}: kept existing scores for {old}; "
                f"now available {names}")


def score_family(family, expected, fresh):
    """Score group `family` with every model of it that loads. Returns the
    expected names that are missing and the list of sets whose scoring failed."""
    fam = C.load_family(family)
    names = [n for n in expected if n in fam]
    missing = [n for n in expected if n not in fam]
    extra = [n for n in fam if n not in expected]
    assert not extra, f"unexpected models in {family}: {extra}"
    assert all(fam[n][2] == "pt" for n in names), f"{family}: not all 'pt' models"
    if missing:
        log(f"  {family}: scoring {len(names)} of {len(expected)} models; missing {missing}")
    if not names:
        log(f"  {family}: no model available, group not scored")
        return missing, []
    sets = C.ref_sets() + C.synth_sets()
    _invalidate(family, names, fresh, sets)

    def predict(im, gt):
        return {n: postprocess_prob(K.pt_prob(fam[n][0], im), fam[n][1]) for n in names}
    bad_sets = []
    try:
        for s in sets:                     # one set at a time: a failing set
            try:                           # does not block the others
                C.score_sets(family, names, predict, [s])
            except Exception:
                bad_sets.append(s[1])
                log(f"  {family}/{s[1]} FAILED:\n{traceback.format_exc()}")
    finally:
        fam.clear()
        _free()
    return missing, bad_sets


def score_all(families, fresh):
    problems = []
    for family, expected in families:
        try:
            miss, bad_sets = score_family(family, expected, fresh)
            if miss:
                problems.append(f"{family} missing {miss}")
            if bad_sets:
                problems.append(f"{family} sets failed {bad_sets}")
        except Exception:
            problems.append(f"{family} scoring failed")
            log(f"  {family}: scoring FAILED\n{traceback.format_exc()}")
        _free()
    return problems


# ------------------------------------------------------------------- stages

def _seed0_missing():
    thr = _load_json("thresholds_pretrained.json")
    return [b for b, _, _ in C.SPECS
            if not os.path.exists(os.path.join(OUT, "ckpt_pretrained", C.name_pre(b, 0) + ".pt"))
            or C.name_pre(b, 0) not in thr]


def _kvasir_missing():
    d = os.path.join(OUT, "ckpt_kvasir")
    return not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.pt"))


def _recreate_kvasir():
    """kit_run.stage_kvasir only sweeps a threshold when the name is absent
    from thresholds_kvasir.json; drop entries whose checkpoint is missing so
    retrained networks never inherit a threshold from other weights, and move
    old kvasir scores aside since they belong to other weights."""
    thr = _load_json("thresholds_kvasir.json")
    ck = os.path.join(OUT, "ckpt_kvasir")
    stale = [n for n in thr if not os.path.exists(os.path.join(ck, n + ".pt"))]
    if stale:
        _aside(os.path.join(OUT, "thresholds_kvasir.json"),
               f"thresholds without checkpoints: {stale}")
        _save_json_atomic(os.path.join(OUT, "thresholds_kvasir.json"),
                          {k: v for k, v in thr.items() if k not in stale})
    for p in sorted(glob.glob(os.path.join(OUT, "scores", "kvasir", "*.json"))):
        _aside(p, "kvasir networks are being retrained")
    K.stage_kvasir()


def stage_seeds():
    log(f"  protocol: {protocol()}, batch {BATCH}, seeds {C.SEEDS_NEW}, versions {_versions()}")
    C.restore_prev()
    problems = []

    missing0 = _seed0_missing()
    kv_missing = _kvasir_missing()
    retrain_kv = kv_missing and os.environ.get("RETRAIN_MISSING") != "0"
    if kv_missing:
        log("  ckpt_kvasir missing after restore; "
            + ("will recreate it with kit_run.stage_kvasir at the end of this stage"
               if retrain_kv else "RETRAIN_MISSING=0, not recreated"))
        if not retrain_kv:
            problems.append("ckpt_kvasir missing (RETRAIN_MISSING=0)")

    if missing0:
        log(f"  seed-0 pretrained models missing after restore: {missing0}; retraining "
            "them into ckpt_pretrained/ with the same protocol (group 'pretrained' "
            "is not rescored)")
        jobs = [(C.name_pre(b, 0), a, e, {}, 0) for b, a, e in C.SPECS if b in missing0]
        fresh0, failed0 = train_all(jobs, os.path.join(OUT, "ckpt_pretrained"),
                                    "thresholds_pretrained.json")
        if failed0:
            problems.append(f"seed-0 retrain failed: {failed0}")
        if fresh0 and os.path.isdir(os.path.join(OUT, "scores", "pretrained")):
            log(f"  WARNING: OUT/scores/pretrained exists but seed-0 models {sorted(fresh0)} "
                "were retrained now; those scores came from other weights")

    jobs = [(C.name_pre(b, s), a, e, {}, s) for s in C.SEEDS_NEW for b, a, e in C.SPECS]
    fresh, failed = train_all(jobs, os.path.join(OUT, "ckpt_seeds"), "thresholds_seeds.json")
    if failed:
        problems.append(f"training failed: {failed}")
    problems += score_all([(f"pretrained_s{s}", [C.name_pre(b, s) for b, _, _ in C.SPECS])
                           for s in C.SEEDS_NEW], fresh)

    if retrain_kv:
        log("  recreating ckpt_kvasir with kit_run.stage_kvasir (ckpt_kvasir was missing)")
        try:
            _recreate_kvasir()
            log("  ckpt_kvasir recreated")
        except Exception:
            problems.append("kit_run.stage_kvasir failed")
            log(f"  kit_run.stage_kvasir FAILED\n{traceback.format_exc()}")
        _free()

    if problems:
        raise RuntimeError("stage seeds incomplete: " + "; ".join(problems))


def stage_modern2():
    log(f"  protocol: {protocol()}, batch {BATCH}, seeds {C.SEEDS2}, versions {_versions()}")
    jobs = [(C.name_m2(b, s), a, e, kw, s) for s in C.SEEDS2 for b, a, e, kw in C.SPECS2]
    fresh, failed = train_all(jobs, os.path.join(OUT, "ckpt_modern2"), "thresholds_modern2.json")
    problems = [f"training failed: {failed}"] if failed else []
    problems += score_all([(f"modern2_s{s}", [C.name_m2(b, s) for b, _, _, _ in C.SPECS2])
                           for s in C.SEEDS2], fresh)
    if problems:
        raise RuntimeError("stage modern2 incomplete: " + "; ".join(problems))


STAGES = {"seeds": stage_seeds, "modern2": stage_modern2}

if __name__ == "__main__":
    for st in sys.argv[1:] or ["seeds", "modern2"]:
        log(f"stage {st} on {DEV}{' (DRY)' if DRY else ''}")
        t0 = time.time()
        try:
            STAGES[st]()
            log(f"stage {st} done in {time.time() - t0:.0f}s")
            open(os.path.join(OUT, f"DONE_{st}"), "w").write(str(time.time() - t0))
        except Exception:
            log(f"stage {st} FAILED:\n{traceback.format_exc()}")
        _free()
