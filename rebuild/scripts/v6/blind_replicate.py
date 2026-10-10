import json, sys, itertools, re
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = '/home/claude/wound-recovery/rebuild'
sys.path.insert(0, ROOT)
from woundbench.combined import ALL_METHOD_FAMILY as FAM

REFS = 'medetec co2wounds fuseg azh azh_train_slice kvasir_disjoint kvasir_sessile cvc_clinicdb cvc_colondb etis_larib ph2 tn3k_test'.split()
MAIN = [r for r in REFS if r != 'azh_train_slice']
WOUND = ['medetec', 'co2wounds', 'fuseg', 'azh']
MOD = {'medetec': 'w', 'co2wounds': 'w', 'fuseg': 'w', 'azh': 'w', 'kvasir_disjoint': 'c', 'kvasir_sessile': 'c',
       'cvc_clinicdb': 'c', 'cvc_colondb': 'c', 'etis_larib': 'c', 'ph2': 'd', 'tn3k_test': 'u'}
EXC = json.load(open(f'{ROOT}/results/exclusions_v5.json'))


def load(path):
    return json.load(open(path))


def keep_mask(ref, n):
    m = np.ones(n, bool)
    for i, _, _ in EXC.get(ref, []):
        m[i] = False
    return m


# per-image dice matrices for every method (kept images), by ref
PI = {}      # ref -> dict name -> per-image dice array (kept)
GTE = {}
IDS = {}
PATHS = {}
for r in REFS:
    d = load(f'{ROOT}/results/perimage32/{r}.json')
    k = keep_mask(r, len(d['ids']))
    D = np.array(d['dice'], float)[k]
    PI[r] = {n: D[:, j] for j, n in enumerate(d['names'])}
    GTE[r] = np.array(d['gt_empty'])[k]
    IDS[r] = [x for x, kk in zip(d['ids'], k) if kk]
    PATHS[r] = [x for x, kk in zip(d['image_paths'], k) if kk]
    for g in ['pretrained', 'nnunet', 'kvasir', 'sam']:
        s = load(f'{ROOT}/results/kaggle/scores/{g}/{r}.json')
        S = np.array(s['dice'], float)[k]
        for j, n in enumerate(s['names']):
            PI[r][n] = S[:, j]

names32 = load(f'{ROOT}/results/perimage32/medetec.json')['names']
F = [n for n in names32 if FAM[n] in ('learned', 'CNN') or FAM[n].startswith('deep')]
U = [n for n in names32 if n not in F]
PRE = load(f'{ROOT}/results/kaggle/scores/pretrained/medetec.json')['names']
M = PRE + ['nnunet_2d']
K = load(f'{ROOT}/results/kaggle/scores/kvasir/medetec.json')['names']
assert len(F) == 14 and len(U) == 18 and len(M) == 11 and len(K) == 14
GROUPS = {'U': U, 'F': F, 'K': K, 'M': M, 'U+F': U + F, 'U+M': U + M}

MEAN = {r: {n: v.mean() for n, v in PI[r].items()} for r in REFS}


def vec(r, meths):
    return np.array([MEAN[r][n] for n in meths])


def agree(a, b, meths):
    return spearmanr(vec(a, meths), vec(b, meths))[0]


out = {}
# 1. wound pair agreement
print('1. wound-pair mean agreement')
for g in ['U', 'F', 'K', 'M', 'U+M']:
    v = [agree(a, b, GROUPS[g]) for a, b in itertools.combinations(WOUND, 2)]
    print(f'  {g}: {np.mean(v):.4f}')

# 2. modality difference with exact permutation test
pairs = list(itertools.combinations(range(11), 2))
labels = [MOD[r] for r in MAIN]
# distinct assignments of multiset
assigns = set(itertools.permutations(labels))
assigns = np.array(sorted(assigns))
print(f'2. distinct assignments: {len(assigns)}, min p = {1/len(assigns):.6f}')
pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
same_mat = (assigns[:, pi] == assigns[:, pj])  # n_assign x 55
for g in ['U', 'F', 'K', 'M', 'U+F', 'U+M']:
    A = np.array([agree(MAIN[i], MAIN[j], GROUPS[g]) for i, j in pairs])
    s = same_mat.astype(float)
    diffs = (s @ A) / s.sum(1) - ((1 - s) @ A) / (1 - s).sum(1)
    obs_same = np.array([labels[i] == labels[j] for i, j in pairs])
    obs = A[obs_same].mean() - A[~obs_same].mean()
    p = np.mean(diffs >= obs - 1e-12)
    print(f'  {g}: diff={obs:.4f} p={p:.5f} (same={A[obs_same].mean():.4f}, diff-mod={A[~obs_same].mean():.4f}, n_same={obs_same.sum()})')

# 3. generator B amp 40 agreement
SEEDS = [2024, 7, 101]
syn = {}
for line in open(f'{ROOT}/results/pool_decomposition_cells.jsonl'):
    rr = json.loads(line)
    if rr['amp'] == 40.0:
        syn[rr['seed']] = dict(zip(rr['names'], rr['dice']))
for s in SEEDS:
    for g in ['pretrained', 'nnunet']:
        d = load(f'{ROOT}/results/kaggle/scores/{g}/genB_amp40_s{s}.json')
        D = np.array(d['dice'], float)
        for j, n in enumerate(d['names']):
            syn[s][n] = D[:, j].mean()
print('3. genB amp40 agreement (mean over seeds, then over wound sets)')
for g in ['U', 'M', 'U+M']:
    meths = GROUPS[g]
    per = {}
    for w in WOUND:
        per[w] = np.mean([spearmanr([syn[s][n] for n in meths], vec(w, meths))[0] for s in SEEDS])
    print(f'  {g}: mean={np.mean(list(per.values())):.4f} ' + ' '.join(f'{w}={v:.3f}' for w, v in per.items()))

# 4. selection among M on wound sets
print('4. selection among M')
Mm = np.array([[MEAN[w][n] for n in M] for w in WOUND])  # 4 x 11
best = Mm.max(1)
for i, w in enumerate(WOUND):
    print(f'  top on {w}: {M[Mm[i].argmax()]} ({best[i]:.4f})')
losses = [best[b] - Mm[b, Mm[a].argmax()] for a in range(4) for b in range(4) if a != b]
print(f'  cross-selection loss mean over 12 pairs: {np.mean(losses):.4f}')
rand = np.mean([(best[i] - Mm[i]).mean() for i in range(4)])
print(f'  random-choice expected loss: {rand:.4f}')
synM = np.array([np.mean([syn[s][n] for s in SEEDS]) for n in M])
cs = synM.argmax()
print(f'  genB-chosen: {M[cs]} (syn {synM[cs]:.4f}); mean loss over 4 = {np.mean(best - Mm[:, cs]):.4f}')
gaps = (best[:, None] - Mm).max(0)
for n, gp in sorted(zip(M, gaps), key=lambda x: x[1]):
    print(f'    maxgap {n}: {gp:.4f}')
print(f'  minimax model: {M[gaps.argmin()]} ({gaps.min():.4f})')

# 5. miss rates
print('5. miss rate (non-empty GT, dice<0.1)')
for r in ['azh', 'fuseg']:
    ne = ~GTE[r]
    mr = {n: np.mean(PI[r][n][ne] < 0.1) for n in M}
    pre = np.array([mr[n] for n in PRE])
    print(f'  {r} (n_nonempty={ne.sum()}): pretrained mean={pre.mean():.4f} range=[{pre.min():.4f},{pre.max():.4f}] nnunet={mr["nnunet_2d"]:.4f}')

# 6. split-half reliability for medetec
print('6. split-half reliability medetec, 57 methods')
ALL57 = names32 + PRE + ['nnunet_2d'] + K
assert len(ALL57) == 57
X = np.stack([PI['medetec'][n] for n in ALL57], 1)  # imgs x 57
ids = IDS['medetec']
photo = [re.sub(r'_\d$', '', i) for i in ids]
uph = sorted(set(photo))
print(f'  images={len(ids)}, photographs={len(uph)}')
rng = np.random.default_rng(0)


def sb(r):
    return 2 * r / (1 + r)


for mode in ['image', 'photo']:
    rs_s, rs_p = [], []
    for _ in range(500):
        if mode == 'image':
            perm = rng.permutation(len(ids)); h = np.zeros(len(ids), bool); h[perm[:len(ids) // 2]] = True
        else:
            perm = rng.permutation(len(uph)); sel = set(np.array(uph)[perm[:len(uph) // 2]])
            h = np.array([p in sel for p in photo])
        a, b = X[h].mean(0), X[~h].mean(0)
        rs_s.append(sb(spearmanr(a, b)[0])); rs_p.append(sb(pearsonr(a, b)[0]))
    print(f'  by {mode}: SB-Spearman median={np.median(rs_s):.4f}  (SB-Pearson median={np.median(rs_p):.4f})')

# 7. near-black share
import cv2
print('7. near-black pixel share (max channel < 10)')
for r in ['azh', 'fuseg']:
    d = load(f'{ROOT}/results/perimage32/{r}.json')
    k = keep_mask(r, len(d['ids']))
    sh = []
    for p in d['image_paths']:
        im = cv2.imread(p if p.startswith('/') else f'{ROOT}/{p}', cv2.IMREAD_COLOR)
        sh.append(np.mean(im.max(2) < 10))
    sh = np.array(sh)
    print(f'  {r}: kept mean={sh[k].mean():.4f} (n={k.sum()}); all mean={sh.mean():.4f} (n={len(sh)})')
