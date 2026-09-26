"""Modern segmentation architectures, trained under the project's own protocol.

The original pool is 21 classical methods plus one 117,681-parameter U-Net. That
is not the population of methods anyone chooses between in 2026, and it is the
single biggest objection to the whole study: classical methods are separated
almost entirely by gross colour and intensity behaviour, which is exactly what
the generators manipulate, so rank agreement over that pool risks being a
statement about colour statistics rather than about benchmark validity.

This module adds ten architectures from segmentation_models_pytorch, spanning
two axes of modern variation:

  seven decoders on one encoder (resnet18): Unet, UnetPlusPlus, DeepLabV3Plus,
  FPN, PSPNet, Linknet, MAnet

  three encoders under one decoder (Unet): mobilenet_v2, efficientnet-b0, and
  a transformer (Segformer on mit_b0, which brings its own decoder)

Every one is randomly initialised. No pretrained weights are used anywhere, so
nothing enters the pool carrying knowledge of an external dataset, and all ten
see exactly the same training images under exactly the same protocol as each
other: 393 FUSeg training images at 128 px, Dice+BCE, Adam, horizontal flips,
30 epochs. That protocol is copied from learned.train_unet so the new models are
comparable both to each other and to the existing one, with a single deliberate
difference: learning rate 1e-3 rather than 3e-3, because 3e-3 with Adam is not
stable for 14 to 22 million parameter models.

Training is checkpointed per epoch so a run can be interrupted and resumed.
"""

import os

import cv2
import numpy as np
import torch

from .learned import dice_bce_loss

SIZE = 128
EPOCHS = 30
LR = 1e-3
BATCH = 8

# (method name, smp architecture, encoder)
DEEP_SPECS = [
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
DEEP_NAMES = [s[0] for s in DEEP_SPECS]
DEEP_FAMILY = {
    "unet_r18": "deep encoder-decoder",
    "unetpp_r18": "deep encoder-decoder",
    "linknet_r18": "deep encoder-decoder",
    "manet_r18": "deep attention",
    "deeplabv3p_r18": "deep atrous",
    "pspnet_r18": "deep pyramid",
    "fpn_r18": "deep pyramid",
    "segformer_mitb0": "deep transformer",
    "unet_mbv2": "deep encoder-decoder",
    "unet_effb0": "deep encoder-decoder",
}


def build(arch, encoder):
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(encoder, encoder_weights=None,
                              in_channels=3, classes=1)


def _tensors(images, masks, size=SIZE):
    X = np.stack([cv2.resize(i, (size, size)) for i in images]).astype(np.float32)
    Y = np.stack([cv2.resize((m > 0).astype(np.uint8), (size, size),
                             interpolation=cv2.INTER_NEAREST)
                  for m in masks]).astype(np.float32)
    return (torch.from_numpy(X).permute(0, 3, 1, 2) / 255.0,
            torch.from_numpy(Y).unsqueeze(1))


def train_one(name, arch, encoder, images, masks, ckpt_dir,
              epochs=EPOCHS, lr=LR, seed=0, log=print, time_budget=None):
    """Train one architecture, resuming from its per-epoch checkpoint.

    Returns (model, epochs_done). If time_budget is set and is exceeded, the
    function returns early with the checkpoint saved, so the caller can resume.
    """
    import time
    t0 = time.time()
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name + ".pt")

    torch.set_num_threads(int(os.environ.get("WB_THREADS", "2")))
    torch.manual_seed(seed)
    model = build(arch, encoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = blob["epoch"]
        if start >= epochs:
            model.eval()
            return model, start

    X, Y = _tensors(images, masks)
    rng = np.random.default_rng(seed)
    # advance the augmentation stream to where it was, so a resumed run is not
    # a different experiment from an uninterrupted one
    for _ in range(start * ((len(X) + BATCH - 1) // BATCH)):
        rng.random()
    n = len(X)
    model.train()
    for ep in range(start, epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            if len(idx) < 2:
                # 393 images at batch 8 leaves a final batch of one, and
                # DeepLabV3+'s ASPP pooling branch applies BatchNorm to a 1x1
                # map, which is undefined for a single sample. One image per
                # epoch is dropped, for every architecture, so the protocol
                # stays identical across the pool.
                continue
            xb, yb = X[idx], Y[idx]
            if rng.random() < 0.5:
                xb, yb = torch.flip(xb, [3]), torch.flip(yb, [3])
            opt.zero_grad()
            loss = dice_bce_loss(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "arch": arch, "encoder": encoder},
                   path)
        if log:
            log(f"    {name} epoch {ep + 1}/{epochs}  loss {tot / n:.4f}  "
                f"({time.time() - t0:.0f}s)")
        if time_budget and time.time() - t0 > time_budget:
            model.eval()
            return model, ep + 1
    model.eval()
    return model, epochs


class DeepZoo:
    """Loads the trained architectures and produces probability fields."""

    def __init__(self, ckpt_dir, threshold=0.5):
        self.ckpt_dir = ckpt_dir
        self.threshold = threshold
        self.models = {}

    def load(self, names=None, require_complete=True):
        names = names or DEEP_NAMES
        spec = {s[0]: s for s in DEEP_SPECS}
        for n in names:
            path = os.path.join(self.ckpt_dir, n + ".pt")
            if not os.path.exists(path):
                continue
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if require_complete and blob.get("epoch", 0) < EPOCHS:
                continue
            _, arch, enc = spec[n]
            m = build(arch, enc)
            m.load_state_dict(blob["model"])
            m.eval()
            self.models[n] = m
        return self

    @property
    def ready(self):
        return sorted(self.models)

    @torch.no_grad()
    def probability(self, name, bgr, size=SIZE):
        h, w = bgr.shape[:2]
        x = cv2.resize(bgr, (size, size)).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        p = torch.sigmoid(self.models[name](t))[0, 0].numpy()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
