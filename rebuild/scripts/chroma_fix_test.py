#!/usr/bin/env python3
"""Does fixing generator B's chroma texture improve rank agreement?

The profile descriptor diagnosed generator B as having roughly six times the
clinical chroma background texture at every level, while its luminance texture
matched clinical closely. That defect has been corrected (fine sensor noise is
now predominantly luminance). Profile distance to clinical fell from 1.04 to
0.22 at the default level.

If profile matching buys predictive validity, the corrected generator should
score higher against clinical than the 0.431 to 0.536 the uncorrected one gave.
If it does not, profile distance is a descriptor rather than a lever.

Resumable via results/chromafix_cells.jsonl.
"""
import json, os, pickle, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woundbench.generator_b import GeneratorB, GenBConfig
from woundbench.methods import METHOD_NAMES
from woundbench.metrics import spearman_permutation
from woundbench import profile as P
from rank_validity import OUT, score_benchmark

AMPS=[0.0, 11.0, 28.0, 40.0]; SEEDS=[2024, 7, 101, 555, 31337]
CELLS=os.path.join(OUT,"chromafix_cells.jsonl")
OLD={0.0:0.431, 11.0:0.439, 28.0:0.523, 40.0:0.536}   # uncorrected, 5 seeds

def done():
    d={}
    if os.path.exists(CELLS):
        for l in open(CELLS):
            if l.strip():
                r=json.loads(l); d[(r["amp"],r["seed"])]=r
    return d

reg=pickle.load(open(os.path.join(OUT,"models.pkl"),"rb"))["registry"]
names=METHOD_NAMES
cache=json.load(open(os.path.join(OUT,"clinical_ranking_cache.json")))
med_v,fus_v=cache["medetec"],cache["fuseg"]
ceiling=spearman_permutation(med_v,fus_v)[0]
ref=json.load(open(os.path.join(OUT,"profile_test.json")))["reference"]
print(f"ceiling {ceiling:+.3f}", flush=True)
D=done(); todo=[(a,s) for a in AMPS for s in SEEDS if (a,s) not in D]
print(f"{len(D)} done, {len(todo)} to go\n", flush=True)
for amp,seed in todo:
    recs=GeneratorB(seed=seed,config=replace(GenBConfig(),mottle_amplitude=amp)).generate(15)
    I=[x[2] for x in recs]; M=[x[3] for x in recs]
    dist=P.distance(P.profile_of_set(I,M), ref)
    summ,_=score_benchmark("bx",I,M,reg,names,log=None)
    v=[summ[n]["mean_dice"] for n in names]
    rm=spearman_permutation(v,med_v,n_perm=4000)[0]; rf=spearman_permutation(v,fus_v,n_perm=4000)[0]
    row={"amp":amp,"seed":seed,"distance":dist,"rho_medetec":rm,"rho_fuseg":rf,"mean_rho":(rm+rf)/2}
    open(CELLS,"a").write(json.dumps(row)+"\n")
    print(f"  amp {amp:>5.1f} seed {seed:>6d}  dist {dist:.3f}  mean rho {(rm+rf)/2:+.3f}", flush=True)
D=done()
print("\n"+"="*70); print("CHROMA-CORRECTED GENERATOR B"); print("="*70)
print(f"{'amp':>6}{'dist':>8}{'n':>4}{'corrected':>12}{'uncorrected':>13}{'change':>9}{'% ceil':>9}")
rows=[]
for a in AMPS:
    c=[r for (aa,_),r in D.items() if aa==a]
    if not c: continue
    v=np.array([x["mean_rho"] for x in c]); dd=float(np.mean([x["distance"] for x in c]))
    rows.append({"amp":a,"distance":dd,"n":len(v),"mean_rho":float(v.mean()),
                 "sd":float(v.std(ddof=1)) if len(v)>1 else None,"old":OLD[a]})
    print(f"{a:>6.0f}{dd:>8.3f}{len(v):>4d}{v.mean():>12.3f}{OLD[a]:>13.3f}"
          f"{v.mean()-OLD[a]:>+9.3f}{v.mean()/ceiling:>8.0%}")
if len(rows)>=3:
    r,p=spearman_permutation([x["distance"] for x in rows],[x["mean_rho"] for x in rows])
    print(f"\nprofile distance vs agreement, corrected generator: Spearman {r:+.3f}, p {p:.4f}")
    best=max(rows,key=lambda x:x["mean_rho"])
    print(f"best: amp {best['amp']:g}, dist {best['distance']:.3f}, "
          f"rho {best['mean_rho']:.3f} = {best['mean_rho']/ceiling:.0%} of ceiling")
# seed-matched paired comparison against the uncorrected generator
paired=[]
UC=os.path.join(OUT,"cross_seed_cells.jsonl")
if os.path.exists(UC):
    U={}
    for l in open(UC):
        if l.strip():
            r=json.loads(l); U[(r["amplitude"],r["seed"])]=r["mean_rho"]
    for (a,s),r in sorted(D.items()):
        if (a,s) in U:
            paired.append({"amp":a,"seed":s,"corrected":r["mean_rho"],
                           "uncorrected":U[(a,s)],"delta":r["mean_rho"]-U[(a,s)]})
if paired:
    print("\n"+"="*70); print("SEED-MATCHED PAIRED COMPARISON"); print("="*70)
    print(f"{'amp':>6}{'seed':>8}{'corrected':>12}{'uncorrected':>13}{'delta':>9}")
    for r in paired:
        print(f"{r['amp']:>6.0f}{r['seed']:>8d}{r['corrected']:>12.3f}"
              f"{r['uncorrected']:>13.3f}{r['delta']:>+9.3f}")
    for a in AMPS:
        sub=[r["delta"] for r in paired if r["amp"]==a]
        if sub:
            print(f"  amp {a:>5.0f}: mean delta {np.mean(sub):+.3f} "
                  f"over {len(sub)} matched seeds, {sum(d>0 for d in sub)}/{len(sub)} positive")
    allsub=[r["delta"] for r in paired]
    pos=sum(d>0 for d in allsub)
    print(f"\noverall: mean delta {np.mean(allsub):+.3f}, "
          f"{pos}/{len(allsub)} pairs improved")
    clin=[r["delta"] for r in paired if r["amp"]>=28]
    if clin:
        print(f"clinical-range levels only (28, 40): mean delta {np.mean(clin):+.3f}, "
              f"{sum(d>0 for d in clin)}/{len(clin)} improved")

json.dump({"rows":rows,"ceiling":ceiling,"paired":paired},open(os.path.join(OUT,"chroma_fix.json"),"w"),indent=2)
print(f"\nsaved {os.path.join(OUT,'chroma_fix.json')}")
