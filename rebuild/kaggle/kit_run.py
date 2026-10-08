#!/usr/bin/env python3
"""GPU run for the benchmark-validity study: four new method groups.

  pretrained   the ten architectures of the main pool with ImageNet encoders,
               256 px letterboxed (aspect ratio kept), trained on FUSeg
  nnunet       nnU-Net v2, 2D configuration, trained on the same FUSeg images
  sam          SAM ViT-B and MedSAM ViT-B, no training, prompted with one box
               per ground-truth component (an oracle prompt, reported as such)
  kvasir       the 14 trained methods of the main pool, same protocol, but
               trained on Kvasir-SEG instead of FUSeg (the mirror-image test)

Every group is scored on the 12 clinical references and the 21 synthetic sets
in this kit with the study's own Dice, NSD and post-processing code, and
writes OUT/scores/<group>/<set>.json in the same layout as perimage32.

Each stage is resumable and independent:  python kit_run.py STAGE
DRY=1 runs every stage end to end on a few images with tiny settings and no
downloads, to check the code before spending GPU time.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
OUT = os.environ.get("OUT", "/kaggle/working/out")
DRY = os.environ.get("DRY") == "1"
os.makedirs(OUT, exist_ok=True)

from woundbench.metrics import dice, nsd                          # noqa: E402
from woundbench.postproc import postprocess_prob                  # noqa: E402

import torch                                                      # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAN = json.load(open(os.path.join(KIT, "manifest.json")))
GRID = np.round(np.arange(0.05, 0.91, 0.05), 2)

SPECS = [
    ("unet_r18",        "Unet",           "resnet18"),
    ("unetpp_r18",      "UnetPlusPlus",   "resnet18"),
    ("deeplabv3p_r18",  "DeepLabV3Plus",  "resnet18"),
    ("fpn_r18",         "FPN",            "resnet18"),
    ("pspnet_r18",      "PSPNet",         "resnet18"),
    ("linknet_r18",     "Linknet",        "resnet18"),
    ("manet_r18",       "MAnet",          "resnet18"),
    ("segformer_mitb0", "Segformer",      "mit_b0"),
    ("unet_mbv2",       "Unet",           "mobilenet_v2"),
    ("unet_effb0",      "Unet",           "efficientnet-b0"),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ data

def read_dir(d, idx=None):
    files = sorted(glob.glob(os.path.join(d, "*_img.png")))
    if idx is not None:
        files = [files[i] for i in idx if i < len(files)]
    X, Y = [], []
    for f in files:
        X.append(cv2.imread(f))
        Y.append((cv2.imread(f.replace("_img.png", "_mask.png"),
                             cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8))
    return X, Y


def synth_root():
    """The synthetic sets, regenerated once into OUT/synth and checked against
    the fingerprints of the images the local pool was scored on."""
    if os.path.isdir(os.path.join(KIT, "synth")):
        return os.path.join(KIT, "synth")
    root = os.path.join(OUT, "synth")
    check = os.path.join(OUT, "synth_check.json")
    if os.path.exists(check):
        return root
    from kit_synth import fingerprint, synth_sets
    res = {}
    for name, f in synth_sets():
        if DRY and name != "genB_amp11_s2024":
            continue
        imgs, masks = f()
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        want = MAN["synth"][name]["fingerprint"]
        same = 0
        for k, (im, m) in enumerate(zip(imgs, masks)):
            same += fingerprint(im, m) == want[k]
            cv2.imwrite(os.path.join(d, f"{k:04d}_img.png"), im)
            cv2.imwrite(os.path.join(d, f"{k:04d}_mask.png"),
                        (np.asarray(m) > 0).astype(np.uint8) * 255)
        res[name] = {"n": len(imgs), "identical": same}
        if same != len(imgs):
            log(f"  WARNING {name}: {len(imgs) - same} of {len(imgs)} images "
                "differ from the locally scored ones")
    json.dump(res, open(check, "w"), indent=1)
    log(f"  synthetic sets regenerated: "
        f"{sum(r['identical'] for r in res.values())} of "
        f"{sum(r['n'] for r in res.values())} identical to the local images")
    return root


def eval_sets():
    """(kind, name, directory) for every set to score."""
    out = [("refs", n, os.path.join(KIT, "refs", n)) for n in MAN["refs"]]
    sr = synth_root()
    out += [("synth", n, os.path.join(sr, n)) for n in MAN["synth"]]
    if DRY:
        out = out[:2] + [s for s in out if s[0] == "synth"][:1]
    return out


def load_eval(d):
    X, Y = read_dir(d)
    if DRY:
        X, Y = X[:3], Y[:3]
    return X, Y


def fuseg_split():
    dup = set(MAN["fuseg_train"]["eval_duplicates"])
    tr = [i for i in range(0, 393) if i not in dup]
    sw = [i for i in range(400, 550) if i not in dup]
    if DRY:
        tr, sw = tr[:6], sw[:3]
    d = os.path.join(KIT, "train", "fuseg")
    return read_dir(d, tr), read_dir(d, sw), tr


def kvasir_split():
    d = os.path.join(KIT, "train", "kvasir")
    px, pxs, net, nets = (list(range(0, 250)), list(range(250, 400)),
                          list(range(0, 393)), list(range(400, 550)))
    if DRY:
        px, pxs, net, nets = px[:4], pxs[:3], net[:6], nets[:3]
    return {k: read_dir(d, v) for k, v in
            (("px", px), ("px_sweep", pxs), ("net", net), ("net_sweep", nets))}


# --------------------------------------------------------------- scoring

def score_group(group, names, predict):
    """predict(bgr, gt) -> {name: uint8 mask}. gt is only used by the
    box-prompted SAM group. Resumable per set."""
    od = os.path.join(OUT, "scores", group)
    os.makedirs(od, exist_ok=True)
    for kind, name, d in eval_sets():
        path = os.path.join(od, name + ".json")
        if os.path.exists(path):
            continue
        X, Y = load_eval(d)
        D, N = [], []
        t0 = time.time()
        for im, gt in zip(X, Y):
            pred = predict(im, gt)
            D.append([dice(pred[n], gt) for n in names])
            N.append([nsd(pred[n], gt) for n in names])
        json.dump({"group": group, "set": name, "kind": kind, "names": names,
                   "n": len(X), "dice": D, "nsd": N,
                   "gt_empty": [bool(g.sum() == 0) for g in Y],
                   "gt_fraction": [float(g.mean()) for g in Y]},
                  open(path, "w"))
        md = np.mean(D, axis=0)
        log(f"  {group}/{name}: {len(X)} images, {time.time() - t0:.0f}s, "
            f"mean Dice {md.min():.3f} to {md.max():.3f}")


def sweep(prob_fn, X, Y):
    probs = [prob_fn(x) for x in X]
    best, best_d = 0.5, -1.0
    for t in GRID:
        d = float(np.mean([dice(postprocess_prob(p, float(t)), y)
                           for p, y in zip(probs, Y)]))
        if d > best_d:
            best, best_d = float(t), d
    return best, best_d


def save_json(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)


def load_json(name, default):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else default


# ------------------------------------------------- stage: pretrained smp

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PT_SIZE = 256


def letterbox(img, size, interp):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size) + img.shape[2:], img.dtype)
    canvas[:nh, :nw] = r
    return canvas, (h, w, nh, nw)


def unletterbox(p, meta):
    h, w, nh, nw = meta
    return cv2.resize(p[:nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)


def pt_input(bgr):
    x, meta = letterbox(bgr, PT_SIZE, cv2.INTER_AREA)
    x = (x[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), meta


def stage_pretrained():
    import segmentation_models_pytorch as smp
    from woundbench.learned import dice_bce_loss
    (Xtr, Ytr), (Xsw, Ysw), _ = fuseg_split()
    epochs = 2 if DRY else int(os.environ.get("PT_EPOCHS", "40"))
    weights = None if DRY else "imagenet"
    ck = os.path.join(OUT, "ckpt_pretrained")
    os.makedirs(ck, exist_ok=True)
    thr = load_json("thresholds_pretrained.json", {})
    report = load_json("train_report_pretrained.json", {})

    XT = torch.stack([pt_input(x)[0] for x in Xtr])
    YT = torch.stack([torch.from_numpy(letterbox(y.astype(np.float32), PT_SIZE,
                      cv2.INTER_NEAREST)[0])[None] for y in Ytr])
    models = {}
    for base, arch, enc in SPECS:
        name = base + "_in"
        path = os.path.join(ck, name + ".pt")
        torch.manual_seed(0)
        m = getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3,
                               classes=1).to(DEV)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEV))
        else:
            t0 = time.time()
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            rng = np.random.default_rng(0)
            scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
            m.train()
            for ep in range(epochs):
                order = rng.permutation(len(XT))
                for i in range(0, len(order), 16):
                    idx = order[i:i + 16]
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
                sched.step()
            torch.save(m.state_dict(), path)
            log(f"  {name}: trained {epochs} epochs in {time.time() - t0:.0f}s")
        m.eval()
        models[name] = m
        if name not in thr:
            t, d = sweep(lambda x, m=m: pt_prob(m, x), Xsw, Ysw)
            thr[name], report[name] = t, {"sweep_dice": d}
            save_json("thresholds_pretrained.json", thr)
            save_json("train_report_pretrained.json", report)
            log(f"  {name}: threshold {t:.2f}, held-out Dice {d:.3f}")

    names = [s[0] + "_in" for s in SPECS]

    def predict(im, gt):
        return {n: postprocess_prob(pt_prob(models[n], im), thr[n]) for n in names}
    score_group("pretrained", names, predict)


@torch.no_grad()
def pt_prob(m, bgr):
    x, meta = pt_input(bgr)
    p = torch.sigmoid(m(x[None].to(DEV)))[0, 0].float().cpu().numpy()
    return unletterbox(p, meta)


# ------------------------------------------------------------ stage: nnU-Net

NN_ID, NN_NAME = 501, "Dataset501_FUSeg"
TRAINER = "nnUNetTrainer_WB"
TRAINER_SRC = '''
import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_WB(nnUNetTrainer):
    """Default nnU-Net trainer with the epoch count set from the environment."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("NNUNET_EPOCHS", "100"))
        it = os.environ.get("NNUNET_ITERS")
        if it:
            self.num_iterations_per_epoch = int(it)
            self.num_val_iterations_per_epoch = max(1, int(it) // 5)
'''


def stage_nnunet():
    import nnunetv2
    base = os.path.join(OUT, "nnunet")
    env = dict(os.environ,
               nnUNet_raw=os.path.join(base, "raw"),
               nnUNet_preprocessed=os.path.join(base, "preprocessed"),
               nnUNet_results=os.path.join(base, "results"),
               NNUNET_EPOCHS="1" if DRY else os.environ.get("NNUNET_EPOCHS", "100"))
    if DRY:
        env["NNUNET_ITERS"] = "2"
        env["nnUNet_n_proc_DA"] = "1"
    tdir = os.path.join(os.path.dirname(nnunetv2.__file__), "training",
                        "nnUNetTrainer", "variants", "wb")
    os.makedirs(tdir, exist_ok=True)
    open(os.path.join(tdir, "__init__.py"), "w").close()
    open(os.path.join(tdir, "nnUNetTrainer_WB.py"), "w").write(TRAINER_SRC)

    raw = os.path.join(env["nnUNet_raw"], NN_NAME)
    dev = ["-device", "cuda" if DEV.type == "cuda" else "cpu"]
    model_dir = os.path.join(env["nnUNet_results"], NN_NAME,
                             f"{TRAINER}__nnUNetPlans__2d", "fold_all")
    if not os.path.exists(os.path.join(model_dir, "checkpoint_final.pth")):
        _, _, tr = fuseg_split()
        os.makedirs(os.path.join(raw, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(raw, "labelsTr"), exist_ok=True)
        src = os.path.join(KIT, "train", "fuseg")
        for i in tr:
            shutil.copy(os.path.join(src, f"{i:04d}_img.png"),
                        os.path.join(raw, "imagesTr", f"fuseg_{i:04d}_0000.png"))
            m = cv2.imread(os.path.join(src, f"{i:04d}_mask.png"),
                           cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(raw, "labelsTr", f"fuseg_{i:04d}.png"),
                        (m > 0).astype(np.uint8))
        json.dump({"channel_names": {"0": "R", "1": "G", "2": "B"},
                   "labels": {"background": 0, "wound": 1},
                   "numTraining": len(tr), "file_ending": ".png"},
                  open(os.path.join(raw, "dataset.json"), "w"), indent=1)
        run(["nnUNetv2_plan_and_preprocess", "-d", str(NN_ID), "-c", "2d",
             "--verify_dataset_integrity", "-npfp", "1" if DRY else "4",
             "-np", "1" if DRY else "4"], env)
        run(["nnUNetv2_train", str(NN_ID), "2d", "all", "-tr", TRAINER] + dev, env)

    # predict every evaluation image once, with probabilities
    pin = os.path.join(base, "pred_in")
    pout = os.path.join(base, "pred_out")
    os.makedirs(pin, exist_ok=True)
    keys = []
    for kind, name, d in eval_sets():
        files = sorted(glob.glob(os.path.join(d, "*_img.png")))
        if DRY:
            files = files[:3]
        for k, f in enumerate(files):
            key = f"{name}__{k:04d}"
            keys.append(key)
            dst = os.path.join(pin, key + "_0000.png")
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    if not all(os.path.exists(os.path.join(pout, k + ".npz")) for k in keys):
        run(["nnUNetv2_predict", "-i", pin, "-o", pout, "-d", str(NN_ID),
             "-c", "2d", "-f", "all", "-tr", TRAINER, "--save_probabilities"]
            + dev + (["-npp", "1", "-nps", "1"] if DRY else []), env)

    counter = {}

    def predict(im, gt):
        # sets are scored in eval_sets() order and images in file order, so a
        # running index per set recovers the key
        name = predict.current
        k = counter.get(name, 0)
        counter[name] = k + 1
        z = np.load(os.path.join(pout, f"{name}__{k:04d}.npz"))["probabilities"]
        p = np.asarray(z)[1].squeeze().astype(np.float32)
        if p.shape != im.shape[:2]:
            p = cv2.resize(p, (im.shape[1], im.shape[0]))
        return {"nnunet_2d": postprocess_prob(p, 0.5)}

    od = os.path.join(OUT, "scores", "nnunet")
    for kind, name, d in eval_sets():
        if os.path.exists(os.path.join(od, name + ".json")):
            continue
        predict.current = name
        counter[name] = 0
        one_set_score("nnunet", ["nnunet_2d"], predict, kind, name, d)
    # the probability files are several GB; the scores are what is kept
    shutil.rmtree(pin, ignore_errors=True)
    shutil.rmtree(pout, ignore_errors=True)


def one_set_score(group, names, predict, kind, name, d):
    od = os.path.join(OUT, "scores", group)
    os.makedirs(od, exist_ok=True)
    X, Y = load_eval(d)
    D, N = [], []
    for im, gt in zip(X, Y):
        pred = predict(im, gt)
        D.append([dice(pred[n], gt) for n in names])
        N.append([nsd(pred[n], gt) for n in names])
    json.dump({"group": group, "set": name, "kind": kind, "names": names,
               "n": len(X), "dice": D, "nsd": N,
               "gt_empty": [bool(g.sum() == 0) for g in Y],
               "gt_fraction": [float(g.mean()) for g in Y]},
              open(os.path.join(od, name + ".json"), "w"))
    log(f"  {group}/{name}: {len(X)} images, mean Dice "
        f"{np.mean(D, axis=0).round(3).tolist()}")


def run(cmd, env):
    log("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    with open(os.path.join(OUT, "nnunet_stdout.txt"), "a") as f:
        f.write(" ".join(cmd) + "\n" + p.stdout[-20000:] + "\n" + p.stderr[-20000:])
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}): {p.stderr[-3000:]}")


# ---------------------------------------------------------- stage: SAM

SAM_MODELS = [("sam_vitb_box", "facebook/sam-vit-base"),
              ("medsam_vitb_box", "wanglab/medsam-vit-base")]
MAX_BOXES = 10


def gt_boxes(gt):
    n, _, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    comps = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    boxes = []
    for i in comps[:MAX_BOXES]:
        x, y, w, h = st[i, :4]
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def stage_sam():
    from transformers import SamConfig, SamModel, SamProcessor
    loaded = {}
    for name, repo in SAM_MODELS:
        try:
            if DRY:
                cfg = SamConfig()
                cfg.vision_config.num_hidden_layers = 2
                model = SamModel(cfg)
                from transformers import SamImageProcessor
                proc = SamProcessor(image_processor=SamImageProcessor())
            else:
                model = SamModel.from_pretrained(repo)
                proc = SamProcessor.from_pretrained(repo)
            loaded[name] = (model.to(DEV).eval(), proc)
            log(f"  loaded {name} from {repo}")
        except Exception as e:
            log(f"  could not load {name} ({repo}): {e}")
    names = list(loaded)
    if not names:
        raise RuntimeError("no SAM model could be loaded")

    @torch.no_grad()
    def prob(model, proc, bgr, boxes):
        h, w = bgr.shape[:2]
        if not boxes:
            return np.zeros((h, w), np.float32)
        rgb = bgr[:, :, ::-1].copy()
        if proc is None:                       # DRY: no processor download
            x = cv2.resize(rgb, (1024, 1024)).astype(np.float32) / 255.0
            pv = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEV)
            s = 1024 / np.array([w, h, w, h], np.float32)
            ib = torch.tensor([[list(np.array(b) * s) for b in boxes]],
                              dtype=torch.float32, device=DEV)
            o = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            lo = o.pred_masks[0, :, 0]
            lo = torch.nn.functional.interpolate(lo[None], (h, w),
                                                 mode="bilinear")[0]
        else:
            inp = proc(rgb, input_boxes=[boxes], return_tensors="pt").to(DEV)
            o = model(**inp, multimask_output=False)
            lo = proc.image_processor.post_process_masks(
                o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                inp["reshaped_input_sizes"].cpu(), binarize=False)[0][:, 0]
        return torch.sigmoid(lo.float()).amax(0).cpu().numpy()

    def predict(im, gt):
        b = gt_boxes(gt)
        return {n: postprocess_prob(prob(*loaded[n], im, b), 0.5) for n in names}
    score_group("sam", names, predict)


# ----------------------------------------------- stage: Kvasir-trained pool

KV_LEARNED = ["logreg_all", "gbm_all", "logreg_colour", "unet"]


def stage_kvasir():
    import pickle
    import segmentation_models_pytorch as smp
    from woundbench.learned import dice_bce_loss
    from woundbench.methods import MethodRegistry
    S = kvasir_split()
    pk = os.path.join(OUT, "kvasir_learned.pkl")
    if os.path.exists(pk):
        reg = pickle.load(open(pk, "rb"))
    else:
        reg = MethodRegistry()
        t0 = time.time()
        reg.fit_pixel_classifiers(*S["px"], seed=0, log=log)
        reg.fit_unet(*S["net"], seed=0, log=log, epochs=1 if DRY else 30)
        probs = [reg.probability("logreg_all", x) for x in S["px_sweep"][0]]
        best, bd = 0.9, -1.0
        for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
            d = float(np.mean([dice(postprocess_prob(p, float(t)), y)
                               for p, y in zip(probs, S["px_sweep"][1])]))
            if d > bd:
                best, bd = float(t), d
        reg.learned_threshold = best
        pickle.dump(reg, open(pk, "wb"))
        log(f"  kvasir pixel classifiers and small U-Net fitted in "
            f"{time.time() - t0:.0f}s; pixel threshold {best:.2f} (Dice {bd:.3f})")

    # the ten architectures: the main pool's protocol (128 px square resize,
    # random init, Dice+BCE, Adam 1e-3, batch 8, horizontal flips, 30 epochs),
    # on the GPU
    X = np.stack([cv2.resize(i, (128, 128)) for i in S["net"][0]]).astype(np.float32)
    Y = np.stack([cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST)
                  for m in S["net"][1]]).astype(np.float32)
    XT = torch.from_numpy(X).permute(0, 3, 1, 2) / 255.0
    YT = torch.from_numpy(Y)[:, None]
    ck = os.path.join(OUT, "ckpt_kvasir")
    os.makedirs(ck, exist_ok=True)
    thr = load_json("thresholds_kvasir.json", {})
    models = {}
    epochs = 1 if DRY else 30
    for base, arch, enc in SPECS:
        name = base + "_kv"
        path = os.path.join(ck, name + ".pt")
        torch.manual_seed(0)
        m = getattr(smp, arch)(enc, encoder_weights=None, in_channels=3,
                               classes=1).to(DEV)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEV))
        else:
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            rng = np.random.default_rng(0)
            m.train()
            for ep in range(epochs):
                order = rng.permutation(len(XT))
                for i in range(0, len(order), 8):
                    idx = order[i:i + 8]
                    if len(idx) < 2:
                        continue
                    xb, yb = XT[idx].to(DEV), YT[idx].to(DEV)
                    if rng.random() < 0.5:
                        xb, yb = xb.flip(3), yb.flip(3)
                    opt.zero_grad()
                    loss = dice_bce_loss(m(xb), yb)
                    loss.backward()
                    opt.step()
            torch.save(m.state_dict(), path)
        m.eval()
        models[name] = m
        if name not in thr:
            t, d = sweep(lambda x, m=m: sq_prob(m, x), *S["net_sweep"])
            thr[name] = t
            save_json("thresholds_kvasir.json", thr)
            log(f"  {name}: threshold {t:.2f}, held-out Dice {d:.3f}")

    names = [n + "_kv" for n in KV_LEARNED] + [s[0] + "_kv" for s in SPECS]

    def predict(im, gt):
        out = {n + "_kv": reg.run(n, im) for n in KV_LEARNED}
        for n, m in models.items():
            out[n] = postprocess_prob(sq_prob(m, im), thr[n])
        return out
    score_group("kvasir", names, predict)


@torch.no_grad()
def sq_prob(m, bgr, size=128):
    h, w = bgr.shape[:2]
    x = cv2.resize(bgr, (size, size)).astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(DEV)
    p = torch.sigmoid(m(t))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ main

def stage_pack():
    """Zip what comes back: scores, thresholds, training reports and logs."""
    import zipfile
    dst = os.path.join(os.path.dirname(OUT.rstrip("/")), "wb_results.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(OUT):
            dirs[:] = [d for d in dirs if not d.startswith(("ckpt", "nnunet", "synth"))]
            for f in files:
                if f.endswith((".json", ".txt")) or f.startswith("DONE_"):
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, OUT))
    log(f"  wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


STAGES = {"pretrained": stage_pretrained, "nnunet": stage_nnunet,
          "sam": stage_sam, "kvasir": stage_kvasir, "pack": stage_pack}

if __name__ == "__main__":
    for st in sys.argv[1:] or ["sam", "pretrained", "kvasir", "nnunet", "pack"]:
        log(f"stage {st} on {DEV}{' (DRY)' if DRY else ''}")
        t0 = time.time()
        try:
            STAGES[st]()
            log(f"stage {st} done in {time.time() - t0:.0f}s")
            open(os.path.join(OUT, f"DONE_{st}"), "w").write(str(time.time() - t0))
        except Exception:
            log(f"stage {st} FAILED:\n{traceback.format_exc()}")
