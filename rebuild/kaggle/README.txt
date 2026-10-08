Benchmark validity: Kaggle GPU kit

1. On kaggle.com: Create, New Dataset. Upload wb-kit.zip. Title it wb-kit, keep it Private, Create.
2. Create, New Notebook. File, Import Notebook, choose wb_gpu_run.ipynb.
3. In the notebook: Add Input, Your Datasets, wb-kit.
   Settings: Accelerator GPU (T4 x2 or P100), Internet On (needs a phone-verified account).
4. Save Version, Save & Run All (Commit). It runs in the background for about 4 to 6 hours;
   you can close the browser.
5. When the version shows as complete, open it, Output tab, download wb_results.zip,
   and upload that file back to Claude.

Contents: refs/ (12 clinical references, exactly as scored locally), train/ (FUSeg and
Kvasir training images), woundbench/ (the study's code), kit_run.py (the run),
kit_synth.py (regenerates the 21 synthetic sets and checks they match), manifest.json.
