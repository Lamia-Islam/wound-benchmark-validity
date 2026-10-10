#!/usr/bin/env python3
"""Build wb_gpu_run2.ipynb: the phase-2 Kaggle notebook. The seven kit-2 code
files are embedded as a base64 zip (real files, no symlinks) and written over
the unpacked wb-kit dataset."""
import base64
import hashlib
import io
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["kit_run.py", "k2_common.py", "k2_train.py", "k2_woundseg.py",
         "k2_rescore.py", "k2_sam.py", "k2_run.py"]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.realpath(os.path.join(HERE, f)), f)
raw = buf.getvalue()
B64 = base64.b64encode(raw).decode()
MD5 = hashlib.md5(raw).hexdigest()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip("\n")}


def stage(title, st):
    return code(f"""
# {title}
!cd "{{KIT}}" && python k2_run.py {st}
!cd "{{KIT}}" && python k2_run.py pack
!tail -n 25 /kaggle/working/out/log.txt
""")


cells = [
    md("""
# Benchmark validity: GPU run 2

Adds to run 1, which this notebook reads back as an input:

1. **woundseg**: a fifth wound dataset (WoundSeg, Hugging Face), 150 images
2. **seeds**: the ten ImageNet-pretrained architectures again at seeds 1 and 2
3. **rescore**: every trained network and nnU-Net under three post-processing rules, five NSD tolerances, a 256 px FUSeg copy and a threshold grid
4. **sam2**: MedSAM with its official preprocessing; SAM and MedSAM prompted with nnU-Net boxes instead of reference boxes
5. **modern2**: eight more modern architectures at three seeds each

**Before running:**
- Settings, Accelerator: **GPU T4 x2** (or P100). Settings, Internet: **On**.
- Add Input, Your Datasets: **wb-kit** (the same dataset as run 1).
- Add Input, Your Work (notebooks): **the run-1 notebook** (the one that produced wb_results.zip).

Then **Save Version, Save & Run All (Commit)**. It takes about 5 to 7 hours.
The results are packed after every step, so a crash or the 12 hour limit still leaves a usable **wb_results2.zip** in Output. Download that file.
"""),
    code("""
# 0. Unpack the kit (same as run 1)
import glob, os, hashlib, subprocess
parts = sorted(glob.glob('/kaggle/input/**/wb-kit.part*', recursive=True))
assert parts, 'Add the wb-kit dataset to this notebook (Add Input, Your Datasets)'
print(len(parts), 'parts found')
with open('/kaggle/working/wb-kit.zip', 'wb') as out:
    for p in parts:
        out.write(open(p, 'rb').read())
want = open(glob.glob('/kaggle/input/**/wb-kit.md5', recursive=True)[0]).read().strip()
got = hashlib.md5(open('/kaggle/working/wb-kit.zip', 'rb').read()).hexdigest()
assert got == want, f'kit is incomplete or corrupted: {got} != {want}; re-upload the parts'
subprocess.run(['unzip', '-q', '-o', '/kaggle/working/wb-kit.zip', '-d', '/kaggle/working/kit'], check=True)
os.remove('/kaggle/working/wb-kit.zip')
KIT = '/kaggle/working/kit'
os.environ['OUT'] = '/kaggle/working/out'
os.makedirs('/kaggle/working/out', exist_ok=True)
print('kit ready at', KIT)
"""),
    code(f"""
# 1. Write the run-2 code into the kit
import base64, io, zipfile, hashlib
CODE = '{B64}'
raw = base64.b64decode(CODE)
assert hashlib.md5(raw).hexdigest() == '{MD5}', 'code bundle damaged; re-import the notebook'
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    z.extractall(KIT)
    print('code written:', z.namelist())
"""),
    code("""
# 2. Packages
!nvidia-smi
!pip install -q segmentation-models-pytorch==0.5.0 nnunetv2==2.8.1
import torch, transformers, timm, cv2
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'transformers', transformers.__version__, 'timm', timm.__version__, 'cv2', cv2.__version__)
assert torch.cuda.is_available(), 'No GPU: set Settings, Accelerator to GPU'
"""),
    code("""
# 3. Restore run 1 (checkpoints, thresholds, nnU-Net model) and check it
!cd "{KIT}" && python k2_run.py restore
import os
need = ['ckpt_pretrained', 'ckpt_kvasir', 'kvasir_learned.pkl', 'thresholds_pretrained.json',
        'thresholds_kvasir.json',
        'nnunet/results/Dataset501_FUSeg/nnUNetTrainer_WB__nnUNetPlans__2d/fold_all/checkpoint_final.pth']
missing = [n for n in need if not os.path.exists(os.path.join('/kaggle/working/out', n))]
print(open('/kaggle/working/out/log.txt').read()[-1500:])
assert not missing, ('Run-1 output not found or incomplete: ' + str(missing) +
                     '. Add Input, Your Work, and attach the run-1 notebook.')
print('run 1 restored:', len(os.listdir('/kaggle/working/out/ckpt_pretrained')), 'pretrained checkpoints')
"""),
    stage("4. WoundSeg, the fifth wound dataset (about 5 minutes)", "woundseg"),
    stage("5. Ten pretrained architectures at seeds 1 and 2 (about 80 minutes)", "seeds"),
    stage("6. Rescore: post-processing, tolerance, resolution, threshold grid (about 60 minutes)", "rescore"),
    stage("7. MedSAM official preprocessing; SAM and MedSAM with nnU-Net boxes (about 30 minutes)", "sam2"),
    stage("8. Eight more modern architectures at three seeds (about 2.5 hours)", "modern2"),
    stage("9. Rescore the new architectures (about 30 minutes)", "rescore"),
    code("""
# 10. Summary
import os, glob
print(sorted(f for f in os.listdir('/kaggle/working/out') if f.startswith('DONE_')))
fails = [l for l in open('/kaggle/working/out/log.txt') if 'FAILED' in l or 'failed' in l]
print(len(fails), 'failure lines'); print(''.join(fails[-20:]))
!ls -la /kaggle/working
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "wb_gpu_run2.ipynb")
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst, len(raw), "byte bundle, md5", MD5)
