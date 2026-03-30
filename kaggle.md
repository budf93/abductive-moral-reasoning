# Kaggle Setup for ARGOS (Full Pipeline)

Follow these steps to run the complete ARGOS neuro-symbolic pipeline on Kaggle.

---

## CLUTRR (Kinship Reasoning)

### Cell 0a: Copy Code to /kaggle/working
```python
import shutil, os

input_path  = '/kaggle/input/datasets/fikribudianto/argos-kaggle/argos-kaggle'
output_path = '/kaggle/working/argos-kaggle'

os.makedirs(output_path, exist_ok=True)
shutil.copytree(input_path, output_path, dirs_exist_ok=True)

os.chdir(output_path)
print("CWD:", os.getcwd())
print(os.listdir("."))
```

### Cell 1: Installation of Dependencies
```python
!pip install -q transformers accelerate bitsandbytes sympy openai z3-solver tqdm pandas scikit-learn scipy
```

### Cell 2: Compile SAT Tools
CaDiCaL and CadiBack need to be compiled. This cell handles the compilation.
```python
import os

ARGOS_DIR   = '/kaggle/working/argos-kaggle'
CADICAL_DIR = f'{ARGOS_DIR}/sat_gen/sat_tools/postprocess/cadical'
CADIBACK_DIR = f'{ARGOS_DIR}/main/cadiback'

# 1. Compile CaDiCaL
print("--- Compiling CaDiCaL ---")
%cd {CADICAL_DIR}
# Fix permissions for the configure script AND all scripts in the scripts folder
!chmod +x configure
!chmod -R +x scripts/
!./configure && make

# 2. Compile CadiBack
print("\n--- Compiling CadiBack ---")
%cd {ARGOS_DIR}/main
!rm -rf cadical
!ln -s ../sat_gen/sat_tools/postprocess/cadical cadical

%cd {CADIBACK_DIR}
# Fix permissions for the configure script AND the generate script
!chmod +x configure
!chmod +x generate

# Create config.hpp manually to ensure it exists even if generate fails
with open("config.hpp", "w") as f:
    f.write('#define VERSION "1.5.0"\n')
    f.write('#define DATE "Kaggle Build"\n')
    f.write('#define IDENTIFIER "manual"\n')
    f.write('#define GITID "manual"\n')
    f.write('#define BUILD "g++ -Wall -O3 -DNDEBUG"\n')

!./configure
!make

print("\n--- Compilation Complete ---")
```

### Cell 3: Environment and File Patching
This cell patches all necessary files to replace hardcoded paths with Kaggle-compatible paths.
```python
import os
ARGOS_DIR = '/kaggle/working/argos-kaggle'
%cd {ARGOS_DIR}

def patch_file(file_path, replacements):
    with open(file_path, "r") as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(file_path, "w") as f:
        f.write(content)

# Global path replacement for the repo — single pattern avoids double-slash issues
global_replace = {
    '/mnt/c/Tugas_Akhir/ARGOS_public_anon': ARGOS_DIR
}

# 1. Patch SAT-LM/run_manual.py (Disable flash-attn)
patch_file("SAT-LM/run_manual.py", {'"flash_attention_2"': '"sdpa"'})

# 2. Create .env with HF Token
with open(".env", "w") as f:
    f.write("HF_TOKEN=your_token_here")

# 3. Apply global replacements to all Python files
import glob
all_scripts = glob.glob(f'{ARGOS_DIR}/**/*.py', recursive=True)

for script in all_scripts:
    patch_file(script, global_replace)
    if "cot_met_clutrr.py" in script:
        patch_file(script, {'attn_implementation="flash_attention_2"': 'attn_implementation="sdpa"'})
    if "cadical_solve.py" in script:
        patch_file(script, {"Pool(70)": "Pool(4)"})

# 4. Create required blank directories
for d in [
    'SAT-LM/tmp', 'SAT-LM/misc', 'main/dimacs', 'main/dimacs_output','main/dimacs_csvs', 'main/tempfilesclutrr_1', 'main/workfilesclutrr_1']:
    os.makedirs(f'{ARGOS_DIR}/{d}', exist_ok=True)

# 5. Fix executable permissions for binaries — must run before any script uses them
!chmod +x {ARGOS_DIR}/main/cadiback/cadiback
!chmod +x {ARGOS_DIR}/sat_gen/sat_tools/postprocess/cadical/build/cadical
!find {ARGOS_DIR}/sat_gen/sat_tools -name "cadical" -type f -exec chmod +x {{}} \;

print("All CLUTRR files patched and directories created.")
```

### Cell 4: Logic Generation (SAT-LM)
```python
%cd /kaggle/working/argos-kaggle/SAT-LM

# Run with only the first data entry for testing
!python3 run_manual.py --task clutrr --num_dev 1 --manual_prompt_id satlm --style_template satlm --run_prediction

# Run with larger subset (original script)
!python3 run_manual.py \
    --task clutrr \
    --manual_prompt_id satlm \
    --style_template satlm \
    --run_prediction \
    --eval_split test \
    --engine Qwen/Qwen2.5-Coder-3B-Instruct \
    --first_k 10
```

### Cell 5: SAT Formulation (DIMACS)
```python
%cd /kaggle/working/argos-kaggle/SAT-LM
!python3 cluttr_to_sat.py
```

### Cell 6: Initial SAT Solving
```python
import os
ARGOS_DIR = '/kaggle/working/argos-kaggle'

# Force patch cadical_solve.py and create output dir before running
with open(f'{ARGOS_DIR}/main/cadical_solve.py', 'r') as f:
    content = f.read()
content = content.replace('/mnt/c/Tugas_Akhir/ARGOS_public_anon', ARGOS_DIR)
# Also replace os.mkdir with os.makedirs to avoid errors if dir already exists
content = content.replace('os.mkdir(output_path)', 'os.makedirs(output_path, exist_ok=True)')
with open(f'{ARGOS_DIR}/main/cadical_solve.py', 'w') as f:
    f.write(content)

os.makedirs(f'{ARGOS_DIR}/main/dimacs_output', exist_ok=True)

%cd /kaggle/working/argos-kaggle/main
!python3 cadical_solve.py
```

### Cell 7: Result Aggregation
```python
import os
ARGOS_DIR = '/kaggle/working/argos-kaggle'

# Force patch parse_gen.py and create csv dir before running
with open(f'{ARGOS_DIR}/main/parse_gen.py', 'r') as f:
    content = f.read()
content = content.replace('/mnt/c/Tugas_Akhir/ARGOS_public_anon', ARGOS_DIR)
content = content.replace('os.mkdir(csv_directory)', 'os.makedirs(csv_directory, exist_ok=True)')
with open(f'{ARGOS_DIR}/main/parse_gen.py', 'w') as f:
    f.write(content)

os.makedirs(f'{ARGOS_DIR}/main/dimacs_csvs', exist_ok=True)

%cd /kaggle/working/argos-kaggle/main
!python3 parse_gen.py
```

### Cell 8: ARGOS Refinement Loop
```python
%cd /kaggle/working/argos-kaggle/main
# Ensure any other hardcoded paths in the script that we missed are handled
!python3 argos/cot_met_clutrr.py
```

### Cell 9: Analysis
```python
%cd /kaggle/working/argos-kaggle/main/analysis
!python3 run_analysis_clutrr.py
```

---

## ExplainEthics (Moral Norm Reasoning)

### Cell 3b: Patch Paths for ExplainEthics
Run this **instead of** (or **after**) Cell 3 to configure ExplainEthics.

```python
import os
ARGOS_DIR = '/kaggle/working/argos-kaggle'
%cd {ARGOS_DIR}

def patch_file(file_path, replacements):
    with open(file_path, "r") as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(file_path, "w") as f:
        f.write(content)

# Global path replacement for the repo — single pattern avoids double-slash issues
global_replace = {
    '/mnt/c/Tugas_Akhir/ARGOS_public_anon': ARGOS_DIR
}

# 1. Apply global replacements to all Python files
import glob
all_scripts = glob.glob(f'{ARGOS_DIR}/**/*.py', recursive=True)

for script in all_scripts:
    patch_file(script, global_replace)
    if "cot_met_explain_ethics.py" in script or "cot_baseline_explain_ethics.py" in script:
        patch_file(script, {'attn_implementation="flash_attention_2"': 'attn_implementation="sdpa"'})
    
    # Fix task/filename inconsistency: data file is explainethics_test.json (no underscore)
    # but some scripts look for explain_ethics_test.json (with underscore)
    patch_file(script, {
        '"explain_ethics_test"': '"explainethics_test"',
        "'explain_ethics_test'": "'explainethics_test'",
        '"explain_ethics_train"': '"explainethics_train"',
        "'explain_ethics_train'": "'explainethics_train'",
        'task == "explain_ethics"': 'task == "explainethics"',
        "task == 'explain_ethics'": "task == 'explainethics'",
    })

# 2. Create required blank directories
# for d in ['SAT-LM/tmp', 'SAT-LM/misc', 'main/dimacs', 
# 'main/dimacs_output','main/dimacs_csvs', 'main/tempfilesexplain_ethics_1', 'preds'
# ]:
#     os.makedirs(f'{ARGOS_DIR}/{d}', exist_ok=True)

# 3. Fix executable permissions for binaries — must run before any script uses them
!chmod +x {ARGOS_DIR}/main/cadiback/cadiback
!chmod +x {ARGOS_DIR}/sat_gen/sat_tools/postprocess/cadical/build/cadical
!find {ARGOS_DIR}/sat_gen/sat_tools -name "cadical" -type f -exec chmod +x {{}} \;

print("All ExplainEthics files patched and directories created.")
```

### Cell 4b: Logic Generation (ExplainEthics)
```python
%cd /kaggle/working/argos-kaggle/SAT-LM
!python3 run_manual.py \
    --task explainethics \
    --manual_prompt_id satlm \
    --style_template satlm \
    --run_prediction \
    --eval_split test \
    --engine Qwen/Qwen2.5-Coder-3B-Instruct \
    # --first_k 1
```

### Cell 5b: SAT Formulation (ExplainEthics)
```python
%cd /kaggle/working/argos-kaggle/SAT-LM
!python3 explain_ethics_to_sat.py
```

### Cell 6–7: Initial SAT Solving + Aggregation
*(Same as CLUTRR — Cells 6 and 7 above)*

### Cell 8b: ARGOS Refinement Loop (ExplainEthics)
```python
%cd /kaggle/working/argos-kaggle/main
!python3 argos/cot_met_explain_ethics.py
```

### Cell 8c: (Optional) Regular CoT Baseline (ExplainEthics)
Run the standalone CoT baseline equivalent (loads model once, loops 20 times):
```python
%cd /kaggle/working/argos-kaggle/main
!python3 argos/cot_baseline_explain_ethics.py --start_iter 0 --end_iter 9
```

### Cell 9b: Analysis (ExplainEthics)
```python
import os
ARGOS_DIR = '/kaggle/working/argos-kaggle'

# Guard run_analysis_explainethics.py against empty predictions crashing resample()
with open(f'{ARGOS_DIR}/main/analysis/run_analysis_explainethics.py', 'r') as f:
    content = f.read()
content = content.replace('/mnt/c/Tugas_Akhir/ARGOS_public_anon', ARGOS_DIR)
# Wrap bootstrap resampling in a guard so it skips gracefully when there are 0 predictions
content = content.replace(
    'bs_outs_acc = [np.sum(resample(outs_pred_val',
    'bs_outs_acc = [np.sum(resample(outs_pred_val' if 'if len(outs_pred_val) > 0' in content
    else '# guarded\nbs_outs_acc = [] if len(outs_pred_val) == 0 else [np.sum(resample(outs_pred_val'
)
with open(f'{ARGOS_DIR}/main/analysis/run_analysis_explainethics.py', 'w') as f:
    f.write(content)

%cd /kaggle/working/argos-kaggle/main/analysis
!python3 run_analysis_explainethics.py
```

### Cell 10: Clean Up Cache Before Downloading Output
Run this to drastically shrink your final Kaggle `Output.zip` download!
```bash
# Delete all .cache files (HuggingFace models, PyCache, etc) 
!rm -rf /kaggle/working/argos-kaggle/main/.cache
!rm -rf /kaggle/working/argos-kaggle/SAT-LM/.cache
# Delete default system caches just in case
!rm -rf ~/.cache/huggingface
```
