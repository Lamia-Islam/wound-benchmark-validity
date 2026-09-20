"""The 3 learned per-pixel classifiers (manuscript 3.5) and the U-Net (3.6).

3.5: logistic regression over all features, gradient boosting over all
features, logistic regression over the nine colour features only. Trained on
264,292 pixels sampled from 250 FUSeg training images, balanced between lesion
and non-lesion, in approximately one second on a single CPU core.

3.6: three-level U-Net with 16, 32 and 64 channels (117,681 parameters),
trained from random initialisation on 393 FUSeg training images at 128 x 128
for 30 epochs with combined Dice and binary cross-entropy loss, horizontal flip
augmentation, and Adam at learning rate 3e-3. Training took 470 s on a single
CPU core.

The 117,681 parameter count is the one exact number available to verify the
architecture rebuild against. `build_unet().count_parameters()` must return it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as feat

N_TRAIN_PIXELS = 264_292     # from the manuscript
N_TRAIN_IMAGES_PIXEL = 250   # FUSeg training images used for the pixel classifiers
N_TRAIN_IMAGES_UNET = 393    # FUSeg training images used for the U-Net
UNET_SIZE = 128
UNET_EPOCHS = 30
UNET_LR = 3e-3
TARGET_UNET_PARAMS = 117_681


# --------------------------------------------------------------------------
# pixel classifiers
# --------------------------------------------------------------------------

def sample_pixels(images, masks, n_total=N_TRAIN_PIXELS, seed=0):
    """Balanced lesion / non-lesion pixel sample across a list of images."""
    rng = np.random.default_rng(seed)
    per_img = max(1, n_total // max(1, len(images)))
    half = per_img // 2
    X, y = [], []
    for img, m in zip(images, masks):
        f = feat.extract(img).reshape(-1, len(feat.FEATURE_NAMES))
        lab = (m.reshape(-1) > 0)
        pos, neg = np.flatnonzero(lab), np.flatnonzero(~lab)
        if len(pos) == 0 or len(neg) == 0:
            continue
        pi = rng.choice(pos, size=min(half, len(pos)), replace=False)
        ni = rng.choice(neg, size=min(half, len(neg)), replace=False)
        idx = np.concatenate([pi, ni])
        X.append(f[idx])
        y.append(lab[idx].astype(np.uint8))
    if not X:
        raise RuntimeError("no usable training pixels")
    X = np.concatenate(X)
    y = np.concatenate(y)
    if len(X) > n_total:
        keep = rng.choice(len(X), size=n_total, replace=False)
        X, y = X[keep], y[keep]
    return X, y


def fit_logreg(X, y, colour_only=False, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if colour_only:
        X = X[:, :len(feat.COLOUR_FEATURES)]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(X, y)
    return clf


def fit_gbm(X, y, seed=0):
    """REBUILD CHOICE: HistGradientBoosting defaults. The manuscript records no
    hyperparameters for the gradient boosting variant."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X, y)
    return clf


def predict_prob(clf, bgr, colour_only=False):
    """Per-pixel lesion probability field for one image."""
    f = feat.extract(bgr)
    h, w, c = f.shape
    flat = f.reshape(-1, c)
    if colour_only:
        flat = flat[:, :len(feat.COLOUR_FEATURES)]
    p = clf.predict_proba(flat)[:, 1]
    return p.reshape(h, w).astype(np.float32)


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------

class _Block(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU.

    NOTE: convolution bias is ON. With bias=False (the usual choice alongside
    BatchNorm) the model totals 117,361 parameters; the manuscript reports
    117,681. The difference is exactly 320, which is the sum of the ten
    convolution bias vectors (16+16 + 32+32 + 64+64 + 32+32 + 16+16). Keeping
    bias=True reproduces the reported count exactly, so the original used
    biased convolutions.
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=True),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=True),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3(nn.Module):
    """Three-level U-Net, 16 / 32 / 64 channels.

    Target parameter count 117,681 (manuscript 3.6). If the assertion in
    build_unet() fails, this architecture does not match the original.
    """

    def __init__(self, ch=(16, 32, 64), in_ch=3):
        super().__init__()
        c1, c2, c3 = ch
        self.enc1 = _Block(in_ch, c1)
        self.enc2 = _Block(c1, c2)
        self.bottleneck = _Block(c2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _Block(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _Block(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_unet(strict: bool = False) -> UNet3:
    m = UNet3()
    n = count_parameters(m)
    if strict and n != TARGET_UNET_PARAMS:
        raise AssertionError(
            f"U-Net has {n:,} parameters, manuscript says {TARGET_UNET_PARAMS:,}")
    return m


def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum() + eps
    den = p.sum() + target.sum() + eps
    return bce + (1 - num / den)


def train_unet(images, masks, epochs=UNET_EPOCHS, lr=UNET_LR, seed=0,
               size=UNET_SIZE, log=print):
    """Train from random initialisation on CPU. Horizontal flip augmentation."""
    import cv2
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.stack([cv2.resize(i, (size, size)) for i in images]).astype(np.float32)
    Y = np.stack([cv2.resize((m > 0).astype(np.uint8), (size, size),
                             interpolation=cv2.INTER_NEAREST)
                  for m in masks]).astype(np.float32)
    X = torch.from_numpy(X).permute(0, 3, 1, 2) / 255.0
    Y = torch.from_numpy(Y).unsqueeze(1)

    model = build_unet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, bs = len(X), 8
    model.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            xb, yb = X[idx], Y[idx]
            if rng.random() < 0.5:                 # horizontal flip
                xb, yb = torch.flip(xb, [3]), torch.flip(yb, [3])
            opt.zero_grad()
            loss = dice_bce_loss(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if log and (ep + 1) % 10 == 0:
            log(f"  unet epoch {ep + 1}/{epochs}  loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def unet_prob(model, bgr, size=UNET_SIZE):
    """Lesion probability field at the image's native resolution."""
    import cv2
    h, w = bgr.shape[:2]
    x = cv2.resize(bgr, (size, size)).astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    p = torch.sigmoid(model(t))[0, 0].numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
