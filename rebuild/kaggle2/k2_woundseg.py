"""Stage woundseg: a fifth wound reference from the WoundSeg dataset of WSNet
(Oota et al., WACV 2023), Hugging Face dataset subbareddyoota/wseg_dataset,
file wseg_dataset.zip, licence CC BY-NC 4.0.

The zip layout is not documented, so the stage discovers it:
  1. download (env WOUNDSEG_ZIP overrides with a local file), extract to
     OUT/woundseg/raw, write OUT/woundseg/tree.txt
  2. classify every image file as image or mask from its path (a parent
     folder named exactly mask(s)/label(s)/gt(s)/sample(s)/image(s) decides
     first), pair them by a cleaned key, check every pair: unreadable files
     dropped; a mask whose aspect ratio differs from its image's is dropped,
     one with the same aspect ratio but another size is resized
     (INTER_NEAREST), and the stage stops if more than 5% of masks differ in
     size at all; masks binarised; exact duplicate images removed; then a
     content check (image edges must follow the mask boundary at least 1.3x
     better than with the next pair's mask) stops the stage on mispairing
  3. duplicate check against the study's wound data (SIFT + FLANN ratio test
     + RANSAC partial affine), one reference index built once
  4. select 150 pairs evenly spaced over the duplicate-free candidates sorted
     by key, resample so the long side is at most 256
  5. write OUT/woundseg/refs/NNNN_{img,mask}.png and OUT/woundseg/meta.json
Resumable: returns at once when meta.json records 150 selected pairs. The
duplicate check is cached in OUT/woundseg/candidates.json.
"""
import collections
import glob
import hashlib
import json
import os
import re
import shutil
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from k2_common import DRY, KIT, OUT, log
import k2_common as C

WDIR = os.path.join(OUT, "woundseg")
RAW = os.path.join(WDIR, "raw")
REFS = os.path.join(WDIR, "refs")
META = os.path.join(WDIR, "meta.json")
CANDS = os.path.join(WDIR, "candidates.json")
TREE = os.path.join(WDIR, "tree.txt")

REPO = "subbareddyoota/wseg_dataset"
FNAME = "wseg_dataset.zip"
LICENCE = "CC BY-NC 4.0"
CITATION = "Oota et al., WSNet: Towards an Effective Method for Wound Image Segmentation, WACV 2023"
N_SELECT = 150
MIN_PAIRS = 100
LONG_SIDE = 256
MAX_RESIZE_FRAC = 0.05  # more masks than this with a size other than the image's: stop
CONTENT_N = 300         # pairs used by the content (mispairing) check
CONTENT_MIN = 1.3       # required mean(true) / mean(shifted) boundary-gradient ratio

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
# words are matched only where they are not glued to other letters
# ("mask12", "12_mask", "masks" match; "wseg" or "length" do not)
MASK_RE = re.compile(r"(?<![a-z])(masks?|labels?|gts?|groundtruths?|ground_truths?|"
                     r"annotations?|annots?)(?![a-z])")
IMAGE_RE = re.compile(r"(?<![a-z])(samples?|images?|imgs?)(?![a-z])")
CLASS_DIR_RE = re.compile(r"(masks?|samples?|images?|labels?|gts?)", re.IGNORECASE)

# duplicate check settings
SIFT_N = 800
SIFT_SIDE = 384
RATIO = 0.75
RANSAC_THR = 5.0
MIN_INLIERS = 25
TOP_REFS = 3
KNN = 6                 # neighbours fetched per descriptor (copy-aware ratio test)
ALIAS = 0.5             # a neighbour this close to the best one is a copy of it
# Supplementary rule for low-texture references: many AZH images give fewer than
# 25 SIFT keypoints, so even an exact copy cannot reach 25 inliers. Such a pair
# also counts as a duplicate when inliers >= LT_MIN and inliers cover at least
# LT_FRAC of the smaller keypoint count. Recorded separately (dup_reason).
# It also needs >= LT_VOTES ratio-test votes for that reference (a Kvasir test
# image once reached 14 of 27 possible inliers with a single vote).
LT_MIN = 8
LT_FRAC = 0.5
LT_VOTES = 3
REF_SETS = [("medetec", "refs/medetec"), ("co2wounds", "refs/co2wounds"),
            ("fuseg", "refs/fuseg"), ("azh", "refs/azh"),
            ("azh_train_slice", "refs/azh_train_slice"), ("train_fuseg", "train/fuseg")]
WORKERS = max(1, min(8, os.cpu_count() or 4))
CACHE_V = 3             # bump when the duplicate rule changes


# ------------------------------------------------------------------ helpers

def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _natural(s):
    """Sort key: digit runs compare as numbers ("2" < "10")."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t)
            for t in re.split(r"(\d+)", s) if t != ""]


def _read_color(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        try:
            from PIL import Image
            im = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return im


def _read_mask_raw(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        try:
            from PIL import Image
            m = np.asarray(Image.open(path))
        except Exception:
            return None
    if m.ndim == 3:
        m = m[..., :3].max(axis=2)       # coloured masks: any lit channel counts
    return m


def _binarise(m):
    mx = int(m.max()) if m.size else 0
    if mx <= 1:
        return (m > 0).astype(np.uint8), "0/1"
    if m.dtype != np.uint8:
        m = (m.astype(np.float64) * (255.0 / (65535 if mx > 255 else 255)))
    return (m > 127).astype(np.uint8), "0/255"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------ 1. download

def _get_zip():
    env = os.environ.get("WOUNDSEG_ZIP")
    if env:
        if not os.path.isfile(env):
            raise RuntimeError(f"WOUNDSEG_ZIP={env} is not a file")
        log(f"  using local zip {env}")
        return env
    if DRY:
        return None
    from huggingface_hub import hf_hub_download
    last = None
    for attempt in range(1, 4):
        try:
            p = hf_hub_download(repo_id=REPO, filename=FNAME, repo_type="dataset")
            log(f"  downloaded {REPO}/{FNAME} -> {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
            return p
        except Exception as e:                       # network, gated repo, renamed file
            last = e
            log(f"  download attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}")
            if attempt < 3:
                time.sleep(10 * attempt)
    raise RuntimeError(f"could not download {REPO}/{FNAME} after 3 attempts "
                       f"({type(last).__name__}: {str(last)[:300]}); if the repo is gated, "
                       "set HF_TOKEN, or attach the zip and set WOUNDSEG_ZIP")


def _extract(zp):
    marker = os.path.join(RAW, ".extracted")
    sig = {"zip": os.path.basename(zp), "bytes": os.path.getsize(zp), "sha256": _sha256(zp)}
    if os.path.exists(marker) and _load(marker) == sig:
        log("  raw folder already extracted from this zip")
        return sig
    if not zipfile.is_zipfile(zp):
        with open(zp, "rb") as f:
            head = f.read(64)
        raise RuntimeError(f"{zp} is not a zip ({sig['bytes']} bytes, starts {head[:32]!r})")
    if os.path.isdir(RAW):
        shutil.rmtree(RAW)
    os.makedirs(RAW)
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        if bad:
            log(f"  WARNING zip member {bad} fails its CRC check")
        z.extractall(RAW)                        # zipfile strips absolute and .. paths
        log(f"  extracted {len(z.namelist())} members to {RAW}")
    # one level of nested zips (some uploads wrap the data twice)
    for inner in glob.glob(os.path.join(RAW, "**", "*.zip"), recursive=True):
        try:
            d = os.path.splitext(inner)[0]
            with zipfile.ZipFile(inner) as z:
                z.extractall(d)
            log(f"  extracted nested zip {os.path.relpath(inner, RAW)}")
        except Exception as e:
            log(f"  WARNING nested zip {inner} not extracted: {e}")
    _save(marker, sig)
    return sig


def _skip_path(rel):
    parts = rel.split(os.sep)
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def _write_tree():
    lines, ext = [], collections.Counter()
    for root, dirs, files in os.walk(RAW):
        dirs.sort()
        rel = os.path.relpath(root, RAW)
        if rel != "." and _skip_path(rel):
            continue
        files = sorted(f for f in files if not f.startswith("."))
        for f in files:
            ext[os.path.splitext(f)[1].lower() or "(none)"] += 1
        lines.append(f"{rel}/  {len(files)} files" + (f": {', '.join(files[:5])}" if files else ""))
    lines.append("")
    lines.append("extensions: " + ", ".join(f"{k} {v}" for k, v in ext.most_common()))
    with open(TREE, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  tree.txt: {len(lines) - 2} directories; extensions " +
        ", ".join(f"{k} {v}" for k, v in ext.most_common(6)))
    return dict(ext)


# --------------------------------------------------------------- 2. pairing

def _classify(rel):
    """'mask', 'image' or None. A parent folder whose whole name is a class
    word (mask, masks, label, labels, gt, gts -> mask; sample, samples, image,
    images -> image), nearest first, decides. Otherwise the file name decides,
    then the deepest folder that contains a class word."""
    parts = rel.lower().split(os.sep)
    for p in parts[-2::-1]:
        f = CLASS_DIR_RE.fullmatch(p)
        if f:
            return "image" if f.group(1).startswith(("sample", "image")) else "mask"
    stem = os.path.splitext(parts[-1])[0]
    for p in [stem] + parts[-2::-1]:
        if MASK_RE.search(p):
            return "mask"
        if IMAGE_RE.search(p):
            return "image"
    return None


def _clean(s, cls):
    """Pairing key from a file stem: both mask-like and image-like words are
    removed, so IMG_12.jpg and IMG_12_mask.png both give "12"."""
    s = IMAGE_RE.sub("", MASK_RE.sub("", s.lower()))
    s = re.sub(r"[_\-\s.]+", "_", s).strip("_")
    return s


def _keys(rel, cls):
    parts = rel.split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    dirs = [p.lower() for p in parts[:-1] if _classify(p + ".x") is None]
    k_stem = _clean(stem, cls)
    return {"path": "/".join(dirs + [k_stem]), "stem": k_stem, "name": stem.lower()}


def _pair_by(files, field):
    by = {"image": collections.defaultdict(list), "mask": collections.defaultdict(list)}
    for rel, cls, keys in files:
        by[cls][keys[field]].append(rel)
    pairs, ambiguous = [], 0
    for k in by["image"].keys() & by["mask"].keys():
        im, mk = by["image"][k], by["mask"][k]
        if len(im) == 1 and len(mk) == 1:
            pairs.append((k, im[0], mk[0]))
        else:
            ambiguous += 1
    return pairs, ambiguous, by


def _discover():
    files, unclassified = [], []
    for root, dirs, fs in os.walk(RAW):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), RAW)
            if _skip_path(rel) or os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            cls = _classify(rel)
            if cls is None:
                unclassified.append(rel)
            else:
                files.append((rel, cls, _keys(rel, cls)))
    files.sort()
    n = collections.Counter(c for _, c, _ in files)
    log(f"  image files: {n['image']} classed image, {n['mask']} classed mask, "
        f"{len(unclassified)} unclassified")
    tried = {}
    for field, label in [("path", "cleaned path key"), ("stem", "cleaned file-name key"),
                         ("name", "identical file names")]:
        pairs, amb, by = _pair_by(files, field)
        tried[label] = {"pairs": len(pairs), "ambiguous_keys": amb}
        log(f"  pairing by {label}: {len(pairs)} pairs, {amb} ambiguous keys")
        if len(pairs) >= MIN_PAIRS:
            return pairs, {"strategy": label, "tried": tried, "n_image_files": n["image"],
                           "n_mask_files": n["mask"], "n_unclassified": len(unclassified)}
    # diagnosis, then a clean stop
    diag = ["", "PAIRING FAILED (fewer than %d pairs with every strategy)" % MIN_PAIRS,
            f"classed image: {n['image']}, classed mask: {n['mask']}, unclassified: {len(unclassified)}",
            "strategies: " + json.dumps(tried)]
    for cls in ("image", "mask"):
        ex = [(r, k) for r, c, k in files if c == cls][:10]
        diag.append(f"first {cls} files and keys: " +
                    "; ".join(f"{r} -> {k['path']} | {k['stem']}" for r, k in ex))
    diag.append("first unclassified files: " + "; ".join(sorted(unclassified)[:10]))
    with open(TREE, "a") as f:
        f.write("\n".join(diag) + "\n")
    for d in diag[1:]:
        log("  " + d[:1000])
    raise RuntimeError(f"woundseg: could not pair images with masks (best "
                       f"{max(v['pairs'] for v in tried.values())} pairs); see {TREE}")


def _check_pair(raw, key, irel, mrel):
    r = {"key": key, "image": irel, "mask": mrel}
    im = _read_color(os.path.join(raw, irel))
    m = _read_mask_raw(os.path.join(raw, mrel))
    if im is None or m is None or im.size == 0 or m.size == 0:
        r["unreadable"] = "image" if im is None or im.size == 0 else "mask"
        return r
    h, w = im.shape[:2]
    r.update(h=h, w=w, mask_h=int(m.shape[0]), mask_w=int(m.shape[1]))
    r["mask_unique"] = int(min(len(np.unique(m)), 999))
    r["md5"] = hashlib.md5(im.tobytes()).hexdigest()
    r["image_binary_like"] = bool(len(np.unique(im[::4, ::4])) <= 2)
    if m.shape[:2] != (h, w):
        r["mask_size_differs"] = True
        if abs(m.shape[1] / m.shape[0] - w / h) > 0.02 * (w / h):
            r["aspect_mismatch"] = True          # dropped, never resized
            return r
        r["mask_resized"] = True
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    b, scale = _binarise(m)
    r["mask_scale"] = scale
    r["fraction"] = float(b.mean())
    return r


def _check_all(pairs, swapped=False):
    with ThreadPoolExecutor(WORKERS) as ex:
        recs = list(ex.map(lambda p: _check_pair(RAW, *p), pairs))
    recs.sort(key=lambda r: _natural(r["key"]))
    counts = {"pairs": len(recs)}
    unread = [r for r in recs if "unreadable" in r]
    ok = [r for r in recs if "unreadable" not in r]
    counts["unreadable"] = len(unread)
    for r in unread[:10]:
        log(f"  unreadable {r['unreadable']}: {r['image']} | {r['mask']}")
    # are the classes swapped? (images that look binary, masks that do not)
    if ok:
        img_bin = np.mean([r["image_binary_like"] for r in ok])
        msk_bin = np.mean([r["mask_unique"] <= 2 for r in ok])
        if img_bin > 0.5 and msk_bin < 0.5 and not swapped:
            log(f"  classes look swapped ({img_bin:.0%} of images binary, "
                f"{msk_bin:.0%} of masks binary); swapping roles")
            return _check_all([(r["key"], r["mask"], r["image"]) for r in recs], True)
    n_diff = sum(bool(r.get("mask_size_differs")) for r in ok)
    counts["mask_size_differs"] = n_diff
    counts["mask_resized"] = sum(bool(r.get("mask_resized")) for r in ok)
    counts["aspect_mismatch_dropped"] = sum(bool(r.get("aspect_mismatch")) for r in ok)
    for r in [r for r in ok if r.get("aspect_mismatch")][:5]:
        log(f"  aspect mismatch (dropped): {r['image']} {r['w']}x{r['h']} | "
            f"{r['mask']} {r['mask_w']}x{r['mask_h']}")
    if ok and n_diff > MAX_RESIZE_FRAC * len(ok):
        raise RuntimeError(f"woundseg: {n_diff} of {len(ok)} masks differ in size from "
                           f"their image (> {MAX_RESIZE_FRAC:.0%}); pairing is suspect")
    ok = [r for r in ok if not r.get("aspect_mismatch")]
    counts["mask_values_0_1"] = sum(r["mask_scale"] == "0/1" for r in ok)
    counts["mask_not_two_valued"] = sum(r["mask_unique"] > 2 for r in ok)
    counts["image_binary_like"] = sum(r["image_binary_like"] for r in ok)
    seen, uniq = set(), []
    for r in ok:
        if r["md5"] in seen:
            continue
        seen.add(r["md5"])
        uniq.append(r)
    counts["exact_duplicate_images_dropped"] = len(ok) - len(uniq)
    counts["after_checks"] = len(uniq)
    counts["classes_swapped"] = swapped
    return uniq, counts


def _edge_ratio(gray_grad, mask_bin):
    """Mean gradient magnitude on the mask boundary / mean over the image."""
    bd = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if not bd.any():
        return None
    return float(gray_grad[bd].mean() / (gray_grad.mean() + 1e-6))


def _content_check(cands):
    """Mispairing check. For up to CONTENT_N pairs evenly spaced over the key
    order, image edges should line up with the mask boundary far better than
    with the mask of the next pair in key order. Raises when
    mean(true) / mean(shifted) < CONTENT_MIN."""
    n = len(cands)
    sel = np.unique(np.round(np.linspace(0, n - 1, min(CONTENT_N, n))).astype(int))

    def mask_for(c, h, w):
        m = _read_mask_raw(os.path.join(RAW, c["mask"]))
        if m is None:
            return None
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return _binarise(m)[0]

    def one(i):
        im = _read_color(os.path.join(RAW, cands[i]["image"]))
        if im is None:
            return None, None
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        h, w = g.shape
        out = []
        for j in (i, (i + 1) % n, (i + n // 2) % n):
            m = mask_for(cands[j], h, w)
            out.append(None if m is None else _edge_ratio(grad, m))
        return out

    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(one, sel))
    ok = [r for r in res if r and r[0] is not None and r[1] is not None]
    far = [r[2] for r in res if r and r[2] is not None]
    info = {"pairs_used": len(ok), "rule": f"mean(true)/mean(shifted by one) >= {CONTENT_MIN}"}
    if len(ok) < 20:
        info["verdict"] = "failed"
        _save(os.path.join(WDIR, "content_check.json"), info)
        raise RuntimeError(f"woundseg: content check had only {len(ok)} pairs with a mask "
                           "boundary (masks mostly empty or unreadable)")
    t = float(np.mean([r[0] for r in ok]))
    s1 = float(np.mean([r[1] for r in ok]))
    info.update(true_mean=round(t, 3), shifted_mean=round(s1, 3), ratio=round(t / s1, 3),
                far_shift_mean=round(float(np.mean(far)), 3) if far else None,
                far_shift_ratio=round(t / float(np.mean(far)), 3) if far else None)
    msg = (f"content check: boundary/whole gradient {t:.2f} true vs {s1:.2f} shifted by "
           f"one (ratio {t / s1:.2f}; vs half-list shift {info['far_shift_ratio']}), "
           f"{len(ok)} pairs")
    log("  " + msg)
    info["verdict"] = "pass" if t / s1 >= CONTENT_MIN else "failed"
    _save(os.path.join(WDIR, "content_check.json"), info)
    if t / s1 < CONTENT_MIN:
        raise RuntimeError(f"woundseg: images and masks look mispaired; {msg}; "
                           f"need ratio >= {CONTENT_MIN}")
    return info


# ------------------------------------------------------ 3. duplicate check

def _sift(path):
    """(points Nx2 float32, descriptors Nx128 float32) or (None, None)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        im = _read_color(path)
        if im is None:
            return None, None
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    s = SIFT_SIDE / max(g.shape[:2])
    if s < 1:
        g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)),
                       interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(SIFT_N).detectAndCompute(g, None)
    if des is None or len(kp) == 0:
        return None, None
    return np.float32([k.pt for k in kp]), des.astype(np.float32)


def _ref_index(pool):
    names, paths = [], []
    missing = []
    for sname, sub in REF_SETS:
        fs = sorted(glob.glob(os.path.join(KIT, sub, "*_img.png")))
        if not fs:
            missing.append(sub)
        for f in fs:
            names.append(f"{sname}/{os.path.basename(f)}")
            paths.append(f)
    if missing:
        log(f"  WARNING reference folders missing or empty: {missing}")
    t0 = time.time()
    feats = list(pool.map(_sift, paths))
    keep = [i for i, (p, d) in enumerate(feats) if d is not None]
    if len(keep) < len(feats):
        log(f"  {len(feats) - len(keep)} reference images gave no SIFT features")
    pts = [feats[i][0] for i in keep]
    des = [feats[i][1] for i in keep]
    names = [names[i] for i in keep]
    owner = np.concatenate([np.full(len(d), j, np.int32) for j, d in enumerate(des)])
    D = np.ascontiguousarray(np.vstack(des))
    index = cv2.flann_Index(D, {"algorithm": 1, "trees": 4})
    log(f"  reference index: {len(names)} images, {len(D)} descriptors, "
        f"{time.time() - t0:.0f}s")
    return {"names": names, "pts": pts, "des": des, "owner": owner, "D": D,
            "index": index, "missing": missing}


def _votes(R, q):
    """Ratio-test votes per reference image for one candidate's descriptors.
    The study's references contain exact and near copies of the same photo,
    so a neighbour that is itself almost identical to the best neighbour
    (and comes from another reference image) is treated as a copy of it and
    skipped before the ratio test; the vote goes to every copy."""
    idx, d2 = R["index"].knnSearch(q, KNN, params={"checks": 64})
    d = np.sqrt(np.maximum(d2, 0))
    owner, D = R["owner"], R["D"]
    o = owner[idx]                                         # (Q, K)
    best = D[idx[:, 0]]
    votes = collections.Counter()
    for row in range(len(q)):
        copies = [o[row, 0]]
        second = None
        for j in range(1, KNN):
            # "<=" with a small tolerance so bit-identical copies (all distances 0) count
            if o[row, j] != o[row, 0] and \
                    np.linalg.norm(D[idx[row, j]] - best[row]) <= ALIAS * d[row, j] + 1e-3:
                copies.append(o[row, j])
                continue
            second = d[row, j]
            break
        if second is None or d[row, 0] < RATIO * second:
            for c in set(copies):
                votes[int(c)] += 1
    return votes


def _verify(cp, cd, rp, rd):
    """Inliers of a RANSAC partial affine fit between candidate and reference."""
    if len(cd) < 2 or len(rd) < 2:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(cd, rd, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < RATIO * b.distance]
    if len(good) < 3:
        return 0
    src = cp[[g.queryIdx for g in good]].reshape(-1, 1, 2)
    dst = rp[[g.trainIdx for g in good]].reshape(-1, 1, 2)
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_THR)
    return int(inl.sum()) if inl is not None else 0


def _dupcheck(paths):
    """For each candidate image path: best reference, inliers, votes."""
    t0 = time.time()
    out = [None] * len(paths)
    with ThreadPoolExecutor(WORKERS) as pool:
        R = _ref_index(pool)
        t1 = time.time()

        def one(i):
            try:
                cp, cd = _sift(paths[i])
                if cd is None:
                    return {"best": None, "inliers": 0, "votes": 0, "dup": False,
                            "dup_reason": None, "no_features": True}
                v = _votes(R, cd)
                top = [j for j, _ in v.most_common(TOP_REFS)]
                res = []
                for j in top:
                    inl = _verify(cp, cd, R["pts"][j], R["des"][j])
                    kmin = min(len(cd), len(R["des"][j]))
                    reason = ("inliers" if inl >= MIN_INLIERS else
                              "low_texture" if inl >= LT_MIN and inl >= LT_FRAC * kmin
                              and v[j] >= LT_VOTES else None)
                    res.append((inl, j, reason, kmin))
                if not res:
                    return {"best": None, "inliers": 0, "votes": 0, "dup": False,
                            "dup_reason": None, "kp": len(cd)}
                hit = [r for r in res if r[2]]
                inl, j, reason, kmin = max(hit or res)
                return {"best": R["names"][j], "inliers": inl, "votes": int(v[j]),
                        "kp": len(cd), "kp_ref": len(R["des"][j]),
                        "dup": reason is not None, "dup_reason": reason,
                        "top": [[R["names"][jj], int(v[jj]), ii] for ii, jj, _, _ in res]}
            except Exception as e:                   # one bad image must not stop the stage
                return {"best": None, "inliers": 0, "votes": 0, "dup": False,
                        "dup_reason": None, "error": f"{type(e).__name__}: {e}"[:300]}

        chunk = 200
        for s in range(0, len(paths), chunk):
            idx = range(s, min(s + chunk, len(paths)))
            for i, r in zip(idx, pool.map(one, idx)):
                out[i] = r
            el = time.time() - t1
            done = idx[-1] + 1
            log(f"  duplicate check {done}/{len(paths)}: {el:.0f}s, "
                f"ETA {el / done * (len(paths) - done):.0f}s")
    t_query = time.time() - t1
    timing = {"index_s": round(t1 - t0, 1), "query_s": round(t_query, 1),
              "s_per_1000": round(t_query / max(1, len(paths)) * 1000, 1),
              "workers": WORKERS, "ref_images": len(R["names"]),
              "ref_descriptors": int(len(R["D"])), "ref_missing": R["missing"]}
    log(f"  duplicate check done: index {timing['index_s']}s, {len(paths)} candidates "
        f"in {timing['query_s']}s ({timing['s_per_1000']}s per 1000, {WORKERS} threads)")
    return out, timing


# ---------------------------------------------------------- 4/5. select, write

def _select_and_write(cands, counts, info):
    free = sorted([c for c in cands if not c["dup"]], key=lambda c: _natural(c["key"]))
    n = len(free)
    if n < MIN_PAIRS:
        raise RuntimeError(f"woundseg: only {n} duplicate-free pairs (need {MIN_PAIRS})")
    sel = np.unique(np.round(np.linspace(0, n - 1, N_SELECT)).astype(int)).tolist()
    if len(sel) < N_SELECT:
        log(f"  WARNING only {len(sel)} pairs selected (fewer than {N_SELECT} candidates)")
    tmp = REFS + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    rows, fracs, sizes = [], [], collections.Counter()
    for k, i in enumerate(sel):
        c = free[i]
        im = _read_color(os.path.join(RAW, c["image"]))
        m = _read_mask_raw(os.path.join(RAW, c["mask"]))
        if im is None or m is None:
            raise RuntimeError(f"woundseg: selected pair became unreadable: {c['image']}")
        h, w = im.shape[:2]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        b, _ = _binarise(m)
        s = LONG_SIDE / max(h, w)
        if s < 1:
            wh = (round(w * s), round(h * s))
            im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, wh, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(tmp, f"{k:04d}_img.png"), im)
        cv2.imwrite(os.path.join(tmp, f"{k:04d}_mask.png"), (b > 0).astype(np.uint8) * 255)
        fracs.append(float((b > 0).mean()))
        sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
        rows.append({"index": k, "candidate_rank": i, "key": c["key"], "image": c["image"],
                     "mask": c["mask"], "source_size": [w, h],
                     "written_size": [int(im.shape[1]), int(im.shape[0])],
                     "lesion_fraction": round(fracs[-1], 5),
                     "best_ref": c["best"], "inliers": c["inliers"]})
    if os.path.isdir(REFS):
        shutil.rmtree(REFS)
    os.replace(tmp, REFS)
    counts["duplicate_free"] = n
    counts["selected"] = len(sel)
    allfr = [c["fraction"] for c in cands]
    srcsz = collections.Counter(f"{c['w']}x{c['h']}" for c in cands)
    ws, hs = [c["w"] for c in cands], [c["h"] for c in cands]
    meta = dict(info)
    meta.update({
        "counts": counts,
        "selection_rule": (f"duplicate-free candidates sorted by pairing key (natural "
                           f"sort, digit runs numeric); indices = unique(round(linspace("
                           f"0, n-1, {N_SELECT}))) with n = {n}; resampled so the long "
                           f"side is at most {LONG_SIDE} (INTER_AREA image, INTER_NEAREST "
                           "mask); mask binarised at > 127 (> 0 when its maximum is 1), "
                           "written 0/255"),
        "selected": rows,
        "n_selected": len(rows),
        "lesion_fraction": {
            "selected_median": float(np.median(fracs)),
            "candidates_median": float(np.median(allfr)),
            "selected_empty": int(sum(f == 0 for f in fracs)),
            "candidates_empty": int(sum(f == 0 for f in allfr))},
        "image_sizes": {
            "candidates_width": [min(ws), int(np.median(ws)), max(ws)],
            "candidates_height": [min(hs), int(np.median(hs)), max(hs)],
            "candidates_top": srcsz.most_common(20),
            "candidates_distinct": len(srcsz),
            "selected_written": sizes.most_common()},
        "note": "masks include peri-wound skin (WSNet paper); not directly comparable "
                "with lesion-only references",
    })
    _save(META, meta)
    log(f"  wrote {len(rows)} pairs to {REFS}; median lesion fraction "
        f"{meta['lesion_fraction']['selected_median']:.3f}, "
        f"{meta['lesion_fraction']['selected_empty']} empty masks")
    return meta


# --------------------------------------------------------------------- stage

def stage_woundseg():
    """Build OUT/woundseg/refs. Returns the refs folder (or None when DRY and
    no local zip is given). Raises RuntimeError with a clear message on any
    failure; the details are in log.txt, tree.txt and error.txt."""
    os.makedirs(WDIR, exist_ok=True)
    meta = _load(META)
    if meta and meta.get("n_selected") == N_SELECT and C.woundseg_dir():
        log(f"  woundseg already built ({N_SELECT} pairs in {REFS})")
        return C.woundseg_dir()
    t0 = time.time()
    try:
        zp = _get_zip()
        if zp is None:
            log("  DRY: no WOUNDSEG_ZIP given and downloads are off; stage skipped")
            return None
        sig = _extract(zp)
        ext = _write_tree()
        pairs, pinfo = _discover()
        cands, counts = _check_all(pairs)
        log("  pair checks: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        if len(cands) < MIN_PAIRS:
            raise RuntimeError(f"woundseg: only {len(cands)} usable pairs after checks")
        content = _content_check(cands)

        cache = _load(CANDS)
        if cache and cache.get("version") == CACHE_V and \
                cache.get("zip_sha256") == sig["sha256"] and \
                [c["key"] for c in cache["candidates"]] == [c["key"] for c in cands]:
            log("  duplicate check loaded from candidates.json")
            cands, timing = cache["candidates"], cache["timing"]
        else:
            res, timing = _dupcheck([os.path.join(RAW, c["image"]) for c in cands])
            for c, r in zip(cands, res):
                c.update(r)
            nerr = sum("error" in c for c in cands)
            if nerr:
                log(f"  WARNING duplicate check failed on {nerr} candidates (kept, "
                    f"first: {next(c['error'] for c in cands if 'error' in c)})")
            _save(CANDS, {"version": CACHE_V, "zip_sha256": sig["sha256"], "timing": timing,
                          "rule": f"duplicate if RANSAC inliers >= {MIN_INLIERS} against "
                                  f"any of the top {TOP_REFS} references by vote, or "
                                  f"inliers >= {LT_MIN} and >= {LT_FRAC} x the smaller "
                                  f"keypoint count with >= {LT_VOTES} votes (low-texture rule)",
                          "candidates": cands})
        dups = [c for c in cands if c["dup"]]
        by_set = collections.Counter(c["best"].split("/")[0] for c in dups)
        inl = np.array([c["inliers"] for c in cands])
        dup_stats = {
            "rule": (f"SIFT({SIFT_N}) on grayscale, max side {SIFT_SIDE}; FLANN kd-tree "
                     f"ratio test {RATIO} (copies of the same feature in duplicate "
                     f"references skipped); top {TOP_REFS} references by vote verified "
                     f"with pairwise ratio test + estimateAffinePartial2D RANSAC "
                     f"(threshold {RANSAC_THR}); duplicate if inliers >= {MIN_INLIERS}; "
                     f"supplementary low-texture rule: inliers >= {LT_MIN} and >= "
                     f"{LT_FRAC} x min(candidate keypoints, reference keypoints) and >= "
                     f"{LT_VOTES} ratio-test votes for that reference"),
            "duplicates_by_rule": dict(collections.Counter(c["dup_reason"] for c in dups)),
            "dupcheck_errors": sum("error" in c for c in cands),
            "no_features": sum(bool(c.get("no_features")) for c in cands),
            "reference_sets": [s for s, _ in REF_SETS],
            "n_candidates": len(cands), "n_duplicates": len(dups),
            "duplicates_by_set": dict(by_set),
            "duplicates": [[c["key"], c["best"], c["inliers"], c["dup_reason"]] for c in dups],
            "inlier_histogram": {"0-9": int((inl < 10).sum()),
                                 "10-24": int(((inl >= 10) & (inl < 25)).sum()),
                                 "25-49": int(((inl >= 25) & (inl < 50)).sum()),
                                 "50+": int((inl >= 50).sum())},
            "timing": timing}
        log(f"  duplicates: {len(dups)} of {len(cands)} candidates "
            f"({dict(by_set) or 'none'}); near misses (10 to 24 inliers): "
            f"{dup_stats['inlier_histogram']['10-24']}")
        counts["duplicates_of_study_data"] = len(dups)
        info = {"source": {"repo_id": REPO, "filename": FNAME, "repo_type": "dataset",
                           "zip": sig, "licence": LICENCE, "citation": CITATION,
                           "paper_reported_pairs": 2686},
                "licence": LICENCE, "extensions": ext, "pairing": pinfo,
                "content_check": content,
                "duplicate_check": dup_stats}
        meta = _select_and_write(cands, counts, info)
        meta["seconds"] = round(time.time() - t0, 1)
        _save(META, meta)
        if os.path.exists(os.path.join(WDIR, "error.txt")):
            os.remove(os.path.join(WDIR, "error.txt"))
        d = C.woundseg_dir()
        if d is None:
            raise RuntimeError("woundseg: refs written but k2_common.woundseg_dir() is None")
        log(f"  woundseg done in {time.time() - t0:.0f}s")
        return d
    except Exception as e:
        with open(os.path.join(WDIR, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
        log(f"  woundseg FAILED: {type(e).__name__}: {e}")
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"woundseg failed: {type(e).__name__}: {e}") from e


if __name__ == "__main__":
    stage_woundseg()
