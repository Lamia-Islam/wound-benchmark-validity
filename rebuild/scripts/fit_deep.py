#!/usr/bin/env python3
"""Train the ten modern architectures on the same data as the original U-Net.

Same 393 FUSeg training images, same 128 px, same Dice+BCE loss, same Adam, same
horizontal-flip augmentation, same 30 epochs. No pretrained weights anywhere.

Checkpointed per epoch, so this can be run repeatedly and will pick up where it
left off. Pass --budget to cap how long each architecture gets in one call.

Writes results/deep/<name>.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import deep as D                        # noqa: E402
from fit_learned import FUSEG_TRAIN, OUT, load_pairs    # noqa: E402

CKPT = os.path.join(OUT, "deep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=240,
                    help="seconds per architecture per call")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    images, masks = load_pairs(os.path.join(FUSEG_TRAIN, "images"),
                               os.path.join(FUSEG_TRAIN, "labels"),
                               0, D_N := 393)
    print(f"{len(images)} FUSeg training images", flush=True)

    specs = D.DEEP_SPECS
    if args.only:
        want = set(args.only.split(","))
        specs = [s for s in specs if s[0] in want]

    t0 = time.time()
    status = []
    for name, arch, enc in specs:
        path = os.path.join(CKPT, name + ".pt")
        import torch
        ep0 = 0
        if os.path.exists(path):
            ep0 = torch.load(path, map_location="cpu",
                             weights_only=False).get("epoch", 0)
        if ep0 >= D.EPOCHS:
            status.append((name, ep0))
            print(f"  {name:<18} done ({ep0}/{D.EPOCHS})", flush=True)
            continue
        print(f"  {name:<18} {arch}/{enc}, resuming from epoch {ep0}",
              flush=True)
        _, ep = D.train_one(name, arch, enc, images, masks, CKPT,
                            time_budget=args.budget)
        status.append((name, ep))

    print("\n" + "=" * 56)
    print(f"{'architecture':<20}{'epochs':>10}{'state':>14}")
    print("-" * 56)
    for name, ep in status:
        print(f"{name:<20}{ep:>4d}/{D.EPOCHS:<5d}"
              f"{'trained' if ep >= D.EPOCHS else 'partial':>14}")
    n_done = sum(1 for _, e in status if e >= D.EPOCHS)
    print(f"\n{n_done}/{len(specs)} complete   "
          f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
