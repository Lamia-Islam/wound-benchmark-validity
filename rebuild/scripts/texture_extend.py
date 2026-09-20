#!/usr/bin/env python3
"""Extend the texture sweep upward to locate the plateau."""
import json, os, pickle, sys
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woundbench.generator_b import GenBConfig
from woundbench.methods import METHOD_NAMES
from woundbench.metrics import spearman_permutation
from rank_validity import DATA, OUT, load_pairs, score_benchmark
from dose_response import build_variant, measured_contrast, spearman_trend

LEVELS = [40.0, 55.0, 75.0]
with open(os.path.join(OUT,"models.pkl"),"rb") as f: reg=pickle.load(f)["registry"]
names=METHOD_NAMES
prev=json.load(open(os.path.join(OUT,"dose_response.json")))
med_v=prev["clinical"]["medetec"]; fus_v=prev["clinical"]["fuseg"]
anchor=spearman_permutation(med_v,fus_v)[0]
print(f"clinical-clinical ceiling: rho {anchor:+.3f}", flush=True)
rows=list(prev["texture_sweep"])
for lv in LEVELS:
    imgs,masks=build_variant(replace(GenBConfig(), mottle_amplitude=lv),2024,15)
    sat,_=measured_contrast(imgs,masks)
    summ,_=score_benchmark("tex",imgs,masks,reg,names,log=None)
    v=[summ[n]["mean_dice"] for n in names]
    rm,pm=spearman_permutation(v,med_v); rf,pf=spearman_permutation(v,fus_v)
    rows.append({"mottle_amplitude":lv,"measured_sat_delta":sat,"rho_medetec":rm,
                 "p_medetec":pm,"rho_fuseg":rf,"p_fuseg":pf,"mean_rho":(rm+rf)/2,
                 "per_method_dice":{n:summ[n]["mean_dice"] for n in names}})
    print(f"  amp={lv:>5.1f}  rho_med {rm:+.3f}  rho_fuseg {rf:+.3f}  mean {(rm+rf)/2:+.3f}",flush=True)
x=[r["mottle_amplitude"] for r in rows]; y=[r["mean_rho"] for r in rows]
rt,pt=spearman_trend(x,y)
print(f"\nfull texture sweep, {len(rows)} levels:")
print(f"  {'amp':>7}{'mean rho':>11}{'rho_med':>10}{'rho_fuseg':>11}{'vs ceiling':>12}")
for r in rows:
    print(f"  {r['mottle_amplitude']:>7.1f}{r['mean_rho']:>11.3f}"
          f"{r['rho_medetec']:>10.3f}{r['rho_fuseg']:>11.3f}"
          f"{r['mean_rho']/anchor:>11.0%}")
print(f"  monotone trend: Spearman {rt:+.3f}, p {pt:.4f}")
prev["texture_sweep_extended"]=rows
prev["trends"]["texture_extended"]={"rho":rt,"p":pt}
prev["clinical_ceiling"]=anchor
json.dump(prev,open(os.path.join(OUT,"dose_response.json"),"w"),indent=2)
print("saved")
