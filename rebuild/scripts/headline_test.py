#!/usr/bin/env python3
"""Headline test: 14-feature classifiers + generator B at CLINICAL texture.

Two corrections combined, each independently motivated:
  1. pixel classifiers on the identified 14-feature set (a_localsd dropped)
  2. generator B at mottle_amplitude=40, whose measured background texture is
     4.40, inside the clinical range 3.97-5.07. Every earlier run used amp=11
     (texture 2.45), which is nearer generator A (2.10) than clinical.
"""
import json, os, pickle, sys
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woundbench.generator_a import build as build_a
from woundbench.generator_b import GeneratorB, GenBConfig
from woundbench.methods import METHOD_NAMES
from woundbench.metrics import spearman_permutation
from woundbench.imagestats import texture_of_set
from rank_validity import DATA, OUT, load_pairs, score_benchmark, TARGETS

# Registry14 was defined in refit_14's __main__, so make it resolvable here
# before unpickling.
import refit_14
sys.modules["__main__"].Registry14 = refit_14.Registry14

AMP = 40.0
reg = pickle.load(open(os.path.join(OUT,"models_14.pkl"),"rb"))["registry"]
names = METHOD_NAMES
b = {}
print("generator A", flush=True)
ai,am,_ = build_a(n_per_class=15)
print(f"  measured background texture {texture_of_set(ai,am)[0]:.2f}", flush=True)
b["generator_a"] = score_benchmark("gA",ai,am,reg,names,log=None)[0]
print(f"generator B @ amp={AMP:g}", flush=True)
recs = GeneratorB(seed=2024, config=replace(GenBConfig(), mottle_amplitude=AMP)).generate(15)
bi=[r[2] for r in recs]; bm=[r[3] for r in recs]
print(f"  measured background texture {texture_of_set(bi,bm)[0]:.2f}  "
      f"(clinical 3.97-5.07)", flush=True)
b["generator_b"] = score_benchmark("gB",bi,bm,reg,names,log=None)[0]
print("medetec", flush=True)
mi,mm = load_pairs(os.path.join(DATA,"Medetec_foot_ulcer_224","train","images"),
                   os.path.join(DATA,"Medetec_foot_ulcer_224","train","labels"))
b["medetec"] = score_benchmark("me",mi,mm,reg,names,log=None)[0]
print("fuseg", flush=True)
fi,fm = load_pairs(os.path.join(DATA,"Foot Ulcer Segmentation Challenge","validation","images"),
                   os.path.join(DATA,"Foot Ulcer Segmentation Challenge","validation","labels"),0,120)
b["fuseg"] = score_benchmark("fu",fi,fm,reg,names,log=None)[0]

ks=list(b); out={}
print("\n"+"="*72); print("HEADLINE TEST"); print("="*72)
print(f"{'comparison':<32}{'rho':>9}{'p':>9}{'target':>10}")
for i,x in enumerate(ks):
    for yv in ks[i+1:]:
        r,p = spearman_permutation([b[x][n]["mean_dice"] for n in names],
                                   [b[yv][n]["mean_dice"] for n in names])
        out[f"{x}__{yv}"]={"rho":r,"p":p}
        t=TARGETS.get((x,yv)); ts=f"{t[0]:>10.3f}" if t else " "*10
        print(f"{x+' vs '+yv:<32}{r:>9.3f}{p:>9.3f}{ts}")
json.dump({"amp":AMP,"benchmarks":b,"rank_agreement":out},
          open(os.path.join(OUT,"headline_test.json"),"w"), indent=2)
print(f"\nsaved {os.path.join(OUT,'headline_test.json')}")
