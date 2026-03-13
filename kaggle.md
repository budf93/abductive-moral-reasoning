# Kaggle Setup for ARGOS (Full Pipeline)

Follow these steps to run the complete ARGOS neuro-symbolic pipeline on Kaggle.

### Cell 0a: Move Code to kaggle/working
```python
# pindahin kodingan ke kaggle/working
import shutil
import os

# Define paths
input_path = '/kaggle/input/datasets/fikribudianto/argos-kaggle/argos-kaggle'      # Replace with your actual dataset name
output_path = '/kaggle/working/argos-kaggle'  # This becomes available in "output" tab

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Copy all files and subdirectories
shutil.copytree(input_path, output_path, dirs_exist_ok=True)
```

### Cell 0b: Move code to kaggle/working
```python
os.system(f"mv /kaggle/input/argos-kaggle/argos-kaggle/")
```

### Cell 0c: Move code to kaggle/working
```python
# Cell 1: move to code directory
import os

# Change working directory to argos-kaggle
os.chdir("/kaggle/working/argos-kaggle")

# Verify the current working directory
print("Current Directory:", os.getcwd())

# List files to confirm
print(os.listdir("."))
```

### Cell 1: Installation of Dependencies
```python
# Install required packages
!pip install -q transformers accelerate bitsandbytes sympy openai z3-solver tqdm pandas
```

### Cell 2: Compile SAT Tools
CaDiCaL and CadiBack need to be compiled. This cell handles the compilation.
```python
import os

# Define paths
WORKING_DIR = "/kaggle/working"
ARGOS_DIR = os.path.join(WORKING_DIR, "argos-kaggle")
CADICAL_DIR = os.path.join(ARGOS_DIR, "sat_gen/sat_tools/postprocess/cadical")
CADIBACK_DIR = os.path.join(ARGOS_DIR, "main/cadiback")

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

%cd {ARGOS_DIR}

def patch_file(file_path, replacements):
    with open(file_path, "r") as f:
        content = f.read()
    for old, new in replacements.items():
        content = content.replace(old, new)
    with open(file_path, "w") as f:
        f.write(content)

# 1. Patch SAT-LM/run_manual.py (Disable flash-attn, update token if needed)
patch_file("SAT-LM/run_manual.py", {
    '"flash_attention_2"': '"sdpa"'
})

# 1b. Create .env file with your HuggingFace token
with open(".env", "w") as f:
    f.write("HF_TOKEN=your_token_here")

# 2. Patch SAT-LM/cluttr_to_sat.py
# Identify and replace paths for tmp, data, and dimacs output
patch_file("SAT-LM/cluttr_to_sat.py", {
    "'/home/XXXX/XXXX/fs_backup_feb13/SAT-LM/tmp_clutrr_good/'": "'/kaggle/working/argos-kaggle/SAT-LM/tmp/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/SAT-LM/data/clutrr_test.json'": "'/kaggle/working/argos-kaggle/SAT-LM/data/clutrr_test.json'",
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_new//'": "'/kaggle/working/argos-kaggle/dimacs/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/clutrr_new_labels.csv'": "'/kaggle/working/argos-kaggle/clutrr_new_labels.csv'"
})

# 3. Patch main/cadical_solve.py
patch_file("main/cadical_solve.py", {
    "'//home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new/'": "'/kaggle/working/argos-kaggle/dimacs/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_output/'": "'/kaggle/working/argos-kaggle/dimacs_output/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/sat_gen/sat_tools/postprocess/cadical/build/cadical'": f"'{CADICAL_DIR}/build/cadical'",
    "Pool(70)": "Pool(4)" # Kaggle usually has 2 or 4 cores
})

# 4. Patch main/parse_gen.py
patch_file("main/parse_gen.py", {
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new/'": "'/kaggle/working/argos-kaggle/dimacs/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_output/'": "'/kaggle/working/argos-kaggle/dimacs_output/'",
    "'/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_csvs/'": "'/kaggle/working/argos-kaggle/dimacs_csvs/'"
})

# 5. Patch main/argos/cot_met_clutrr.py
patch_file("main/argos/cot_met_clutrr.py", {
    "USER_PATH = '/home/XXXX/XXXX/'": "USER_PATH = '/kaggle/working/argos-kaggle/'",
    "'/home/XXXX/XXXX/LLM-project/dimacs_clutrr_csvs/solver_finished.csv'": "'/kaggle/working/argos-kaggle/dimacs_csvs/solver_finished.csv'",
    "'/home/XXXX/XXXX/SAT-LM/data/clutrr_test.json'": "'/kaggle/working/argos-kaggle/SAT-LM/data/clutrr_test.json'",
    "'/home/XXXX/XXXX/LLM-project/dimacs_clutrr/'": "'/kaggle/working/argos-kaggle/dimacs/'",
    "'/home/XXXX/XXXX/LLM-project/tempfiles'": "'/kaggle/working/argos-kaggle/tempfiles'",
    "'/home/XXXX/XXXX/LLM-project/workfiles'": "'/kaggle/working/argos-kaggle/workfiles'",
    "'/home/XXXX/XXXX/LLM-project/cadiback/cadiback'": f"'{CADIBACK_DIR}/cadiback'",
    "'/home/XXXX/XXXX/all_outs_cot_met_clutrr_'": "'/kaggle/working/argos-kaggle/all_outs_cot_met_clutrr_'",
    "'/home/XXXX/XXXX/LLM-project/clutrr_labels.csv'": "'/kaggle/working/argos-kaggle/clutrr_labels.csv'",
    "attn_implementation=\"flash_attention_2\"": "attn_implementation=\"sdpa\""
})

# Create required directories
os.makedirs("/kaggle/working/argos-kaggle/SAT-LM/tmp", exist_ok=True)
os.makedirs("/kaggle/working/argos-kaggle/dimacs", exist_ok=True)
os.makedirs("/kaggle/working/argos-kaggle/dimacs_output", exist_ok=True)
os.makedirs("/kaggle/working/argos-kaggle/dimacs_csvs", exist_ok=True)
os.makedirs("/kaggle/working/argos-kaggle/tempfilesclutrr_1", exist_ok=True)
os.makedirs("/kaggle/working/argos-kaggle/workfilesclutrr_1", exist_ok=True)

print("All files patched and directories created.")
```

### Cell 4: Step 2 - Logic Generation (SAT-LM)
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

### Cell 5: Step 3 - SAT Formulation ***
```python
%cd /kaggle/working/argos-kaggle/SAT-LM
!python3 cluttr_to_sat.py
```

### Cell 6: Step 4 - Initial SAT Solving
```python
%cd /kaggle/working/argos-kaggle/main
!python3 cadical_solve.py
```

### Cell 7: Step 5 - Result Aggregation
```python
%cd /kaggle/working/argos-kaggle/main
!python3 parse_gen.py
```

### Cell 8: Step 6 - argos-kaggle Neuro-Symbolic Refinement
```python
%cd /kaggle/working/argos-kaggle/main
# Ensure any other hardcoded paths in the script that we missed are handled
!python3 argos/cot_met_clutrr.py
```
