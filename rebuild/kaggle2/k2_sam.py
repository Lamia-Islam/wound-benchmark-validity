"""Stage sam2: SAM ViT-B and MedSAM ViT-B re-scored with corrected MedSAM
preprocessing, and with boxes from nnU-Net instead of the ground truth.

Run 1 fed MedSAM through the Hugging Face SamProcessor (rescale to [0,1],
ImageNet mean/std, long side resized to 1024 and padded). The official MedSAM
inference script (bowang-lab/MedSAM, MedSAM_Inference.py) does something else,
and MedSAM was fine-tuned with that preprocessing:

    img_1024 = transform.resize(img_3c, (1024, 1024), order=3,
                                preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    image_embedding = medsam_model.image_encoder(img_1024_tensor)
    ...
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear",
                                 align_corners=False)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

medsam_preprocess, medsam_box_1024 and medsam_prob below reproduce those lines
with the transformers SamModel port of the weights ("wanglab/medsam-vit-base").
SamModel.forward does not normalise pixel_values (the vision encoder starts
with the patch-embedding convolution), and its prompt encoder divides box
coordinates by config.image_size (1024) after adding 0.5, exactly as the
official segment_anything prompt encoder does, so boxes are passed in the
1024 x 1024 input frame. The only step after the official sigmoid map is the
study's shared post-processing (postprocess_prob at 0.5, as in run 1), which
every method receives.

Groups written to OUT/scores/<group>/<set>.json (k2_common.score_sets layout):

  sam_v2_oracle   medsam_fixed_gtbox
      ORACLE PROMPTS: one box per ground-truth connected component (the up to
      MAX_BOXES largest, kit_run.gt_boxes), union of the per-box masks. The
      boxes come from the ground truth being scored, so these numbers are an
      upper-bound style reference, not a deployable method. Empty ground truth
      gives no box and an empty prediction, as in run 1.

  sam_v2_nnbox    sam_nnbox, medsam_fixed_nnbox
      Boxes from the nnU-Net P0 masks written by the rescore stage
      (k2_rescore.load_nnunet_masks): bounding boxes of the up to MAX_BOXES
      largest connected components of the nnU-Net mask. No component means no
      prompt and an empty prediction. sam_nnbox uses SamProcessor exactly as
      kit_run.stage_sam; medsam_fixed_nnbox uses the official preprocessing.

Every prediction: prob = max over boxes of the per-box sigmoid probability at
native size, then postprocess_prob(prob, 0.5). Sets: k2_common.ref_sets()
(clinical references only, no synthetic sets). Each set is scored in its own
try/except and skipped when its JSON exists, so the stage is resumable.
"""
import glob
import hashlib
import importlib
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import k2_common as C                                             # noqa: E402
import kit_run as K                                               # noqa: E402

OUT, DEV, DRY, log = C.OUT, C.DEV, C.DRY, C.log
MAX_BOXES = K.MAX_BOXES

SAM_REPO = "facebook/sam-vit-base"
MEDSAM_REPO = "wanglab/medsam-vit-base"
G_ORACLE, N_ORACLE = "sam_v2_oracle", ["medsam_fixed_gtbox"]
G_NNBOX, N_NNBOX = "sam_v2_nnbox", ["sam_nnbox", "medsam_fixed_nnbox"]


# ----------------------------------------------- official MedSAM pre/post

def medsam_preprocess(rgb):
    """Official MedSAM_Inference.py image preprocessing. rgb: HxWx3 uint8 in
    RGB order (the official script reads with skimage.io, which gives RGB).
    Returns a (1, 3, 1024, 1024) float32 tensor on the CPU."""
    from skimage import transform
    img_3c = np.asarray(rgb)
    if img_3c.ndim == 2:
        img_3c = np.repeat(img_3c[:, :, None], 3, axis=-1)
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    return torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0)


def medsam_box_1024(boxes, h, w):
    """Official box scaling: box / [W, H, W, H] * 1024 (no rounding).
    boxes: list of [x0, y0, x1, y1] in original pixel coordinates."""
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) / np.array([w, h, w, h]) * 1024


def medsam_upsample(low_res_logits, h, w):
    """Official postprocessing up to the threshold: sigmoid on the 256x256
    logits, then bilinear (align_corners=False) to the original H x W.
    low_res_logits: (B, 1, 256, 256). Returns (B, H, W) float32 numpy."""
    p = torch.sigmoid(low_res_logits.float())
    p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
    return p[:, 0].cpu().numpy()


class EmbCache:
    """MedSAM image embeddings of the current set, keyed by the image bytes,
    so the oracle and nnU-Net-box passes share one (slow) official
    preprocessing and encoder call per image. Kept on the CPU, bounded."""
    def __init__(self, cap=400):
        self.cap, self.d = cap, {}

    @staticmethod
    def key(bgr):
        return hashlib.sha1(np.ascontiguousarray(bgr).tobytes()).hexdigest() + str(bgr.shape)

    def get(self, bgr):
        e = self.d.get(self.key(bgr))
        return None if e is None else e.to(DEV)

    def put(self, bgr, emb):
        if len(self.d) < self.cap:
            self.d[self.key(bgr)] = emb.cpu()

    def clear(self):
        self.d.clear()


@torch.no_grad()
def medsam_prob(model, bgr, boxes, cache=None):
    """Union (max) of the per-box MedSAM probability maps at native size."""
    h, w = bgr.shape[:2]
    if not boxes:
        return np.zeros((h, w), np.float32)
    emb = cache.get(bgr) if cache is not None else None
    if emb is None:
        pv = medsam_preprocess(bgr[:, :, ::-1]).to(DEV)
        emb = model.get_image_embeddings(pixel_values=pv)        # (1, 256, 64, 64)
        if cache is not None:
            cache.put(bgr, emb)
    ib = torch.as_tensor(medsam_box_1024(boxes, h, w)[None], dtype=torch.float,
                         device=DEV)                             # (1, nbox, 4)
    o = model(image_embeddings=emb, input_boxes=ib, multimask_output=False)
    lo = o.pred_masks[0]                                         # (nbox, 1, 256, 256)
    return medsam_upsample(lo, h, w).max(0).astype(np.float32)


# ------------------------------------------- SAM via SamProcessor (run 1)

@torch.no_grad()
def sam_proc_prob(model, proc, bgr, boxes):
    """Identical to the processor branch of kit_run.stage_sam.prob."""
    h, w = bgr.shape[:2]
    if not boxes:
        return np.zeros((h, w), np.float32)
    rgb = bgr[:, :, ::-1].copy()
    inp = proc(rgb, input_boxes=[boxes], return_tensors="pt").to(DEV)
    o = model(**inp, multimask_output=False)
    lo = proc.image_processor.post_process_masks(
        o.pred_masks.cpu(), inp["original_sizes"].cpu(),
        inp["reshaped_input_sizes"].cpu(), binarize=False)[0][:, 0]
    return torch.sigmoid(lo.float()).amax(0).cpu().numpy()


# ----------------------------------------------------------------- boxes

def mask_boxes(mask):
    """Bounding boxes [x0, y0, x0 + w, y0 + h] of the up to MAX_BOXES largest
    8-connected components; same rule as kit_run.gt_boxes."""
    return K.gt_boxes((np.asarray(mask) > 0).astype(np.uint8))


def _n_images(d):
    return len(glob.glob(os.path.join(d, "*_img.png")))


def nnunet_masks(set_name, d):
    """nnU-Net P0 masks for one set from k2_rescore.load_nnunet_masks, or
    (None, reason). The module is imported lazily; it may be missing or
    incomplete."""
    try:
        R = importlib.import_module("k2_rescore")
    except Exception as e:                 # missing or broken module
        return None, f"k2_rescore not importable ({type(e).__name__}: {e})"
    fn = getattr(R, "load_nnunet_masks", None)
    if fn is None:
        return None, "k2_rescore has no load_nnunet_masks"
    try:
        masks = fn(set_name)
    except FileNotFoundError as e:
        return None, f"no nnU-Net masks file ({e})"
    except Exception:
        return None, "load_nnunet_masks failed:\n" + traceback.format_exc()
    if masks is None:
        return None, "load_nnunet_masks returned None"
    masks = list(masks)
    n = _n_images(d)
    # DRY scoring reads only the first 3 images (kit_run.load_eval), so a DRY
    # rescore stage may have written only those
    if len(masks) != n and not (DRY and len(masks) == min(n, 3)):
        return None, f"{len(masks)} nnU-Net masks for {n} images"
    return masks, None


# ----------------------------------------------------------------- models

def _load():
    """{'sam': (model, proc), 'medsam': model}; entries that fail to load are
    left out with a log line."""
    from transformers import SamConfig, SamImageProcessor, SamModel, SamProcessor
    out = {}
    for key, repo in (("sam", SAM_REPO), ("medsam", MEDSAM_REPO)):
        try:
            if DRY:
                torch.manual_seed(0 if key == "sam" else 1)
                cfg = SamConfig()
                cfg.vision_config.num_hidden_layers = 2
                model = SamModel(cfg)
                proc = SamProcessor(image_processor=SamImageProcessor())
            else:
                model = SamModel.from_pretrained(repo)
                proc = SamProcessor.from_pretrained(repo) if key == "sam" else None
            model = model.to(DEV).eval()
            out[key] = (model, proc) if key == "sam" else model
            log(f"  loaded {key} from {repo}{' (DRY: tiny random config)' if DRY else ''}")
        except Exception as e:
            log(f"  could not load {key} ({repo}): {e}")
    return out


# ------------------------------------------------------------------ stage

def _score_one(group, names, predict, s):
    try:
        C.score_sets(group, names, predict, [s])
    except Exception:
        log(f"  {group}/{s[1]} FAILED:\n{traceback.format_exc()}")


def stage_sam2():
    """Score sam_v2_oracle (MedSAM with ORACLE ground-truth boxes, official
    preprocessing) and sam_v2_nnbox (SAM via SamProcessor and MedSAM with
    official preprocessing, both prompted with nnU-Net boxes) on
    k2_common.ref_sets(). The oracle group uses boxes taken from the very
    ground truth it is scored against and must be reported as an oracle."""
    sets = C.ref_sets()
    todo = {g: [s for s in sets if not os.path.exists(
        os.path.join(OUT, "scores", g, s[1] + ".json"))] for g in (G_ORACLE, G_NNBOX)}
    if not any(todo.values()):
        log("  sam2: every set already scored")
        return
    M = _load()
    med = M.get("medsam")
    sam_model, sam_proc = M.get("sam", (None, None))
    # each group scores whichever of its methods loaded; a set JSON written
    # now lists only those names and is not rescored later (resumability)
    n_oracle = N_ORACLE if med is not None else []
    n_nnbox = [n for n, ok in (("sam_nnbox", sam_model is not None),
                               ("medsam_fixed_nnbox", med is not None)) if ok]
    log(f"  {G_ORACLE}: methods {n_oracle or 'none, group skipped'}")
    log(f"  {G_NNBOX}: methods {n_nnbox or 'none, group skipped'}")
    if not n_oracle and not n_nnbox:
        raise RuntimeError("neither SAM nor MedSAM could be loaded; nothing to score")
    cache = EmbCache()

    def predict_oracle(im, gt):
        return {"medsam_fixed_gtbox":
                C.postprocess_prob(medsam_prob(med, im, K.gt_boxes(gt), cache), 0.5)}

    for s in sets:
        kind, name, d = s
        cache.clear()
        # -------- ORACLE: MedSAM, official preprocessing, ground-truth boxes
        if s in todo[G_ORACLE] and n_oracle:
            _score_one(G_ORACLE, n_oracle, predict_oracle, s)
        # -------- nnU-Net boxes: SAM (processor) and MedSAM (official)
        if s not in todo[G_NNBOX] or not n_nnbox:
            continue
        masks, why = nnunet_masks(name, d)
        fake = False
        if masks is None:
            if not DRY:
                log(f"  {G_NNBOX}/{name}: skipped, {why}")
                continue
            log(f"  {G_NNBOX}/{name}: DRY, nnU-Net masks unavailable ({why.splitlines()[0]}); "
                "using fake boxes from the ground truth to exercise the code path")
            fake = True
        state = {"k": 0}

        def predict_nn(im, gt, masks=masks, fake=fake, state=state, name=name):
            k = state["k"]
            state["k"] = k + 1
            if fake:
                m = gt
            else:
                m = np.asarray(masks[k])
                if m.shape[:2] != im.shape[:2]:
                    raise ValueError(f"{name} image {k}: nnU-Net mask {m.shape} "
                                     f"vs image {im.shape[:2]}")
            b = mask_boxes(m)
            out = {}
            if sam_model is not None:
                out["sam_nnbox"] = C.postprocess_prob(
                    sam_proc_prob(sam_model, sam_proc, im, b), 0.5)
            if med is not None:
                out["medsam_fixed_nnbox"] = C.postprocess_prob(
                    medsam_prob(med, im, b, cache), 0.5)
            return out
        _score_one(G_NNBOX, n_nnbox, predict_nn, s)
    cache.clear()


if __name__ == "__main__":
    log(f"stage sam2 on {DEV}{' (DRY)' if DRY else ''}")
    stage_sam2()
