# ARGOS: CLUTRR Dataset Execution Guide

This guide provides step-by-step instructions to run the ARGOS neuro-symbolic pipeline on the **CLUTRR** dataset (kinship relation reasoning).

## ⚠️ Important: Path Configuration
Most scripts contain hardcoded paths (e.g., `/mnt/c/Tugas_Akhir/...`). The key variable to update is `USER_PATH` in `cot_met_clutrr.py`. Other paths are derived from it automatically.

---

## Step 1: Environment Setup
1.  **Setup Virtual Environment:**
    *   **Windows (PowerShell):**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **Unix / Linux / WSL:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
2.  **Install Python Dependencies:**
    ```bash
    # Step A: Install core build dependencies first
    pip install packaging ninja wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    # Step B: Install the rest
    pip install -r requirements.txt
    ```
3.  **Compile SAT Tools** *(requires Unix/WSL)*:
    *   **CaDiCaL:**
        ```bash
        cd sat_gen/sat_tools/postprocess/cadical
        ./configure && make
        # Binary: sat_gen/sat_tools/postprocess/cadical/build/cadical
        ```
    *   **CadiBack:**
        ```bash
        cd main
        ln -s ../sat_gen/sat_tools/postprocess/cadical cadical
        cd cadiback
        # Create config manually if git is not present
        echo '#define VERSION "1.5.0"' > config.hpp
        echo '#define DATE "manual"' >> config.hpp
        echo '#define IDENTIFIER "manual"' >> config.hpp
        echo '#define GITID "manual"' >> config.hpp
        echo '#define BUILD "g++ -Wall -O3 -DNDEBUG"' >> config.hpp
        ./configure && make
        ```

---

## Step 2: Logic Generation (SAT-LM)
Translate natural language kinship stories into Z3 Python logic programs.

1.  **Configure:** Create a `.env` file in the project root:
    ```
    HF_TOKEN=hf_xxxx
    ```
2.  **Run:**
    ```bash
    cd SAT-LM
    python3 run_manual.py \
        --task clutrr \
        --manual_prompt_id satlm \
        --style_template satlm \
        --run_prediction \
        --eval_split test \
        --engine Qwen/Qwen2.5-Coder-3B-Instruct \
        --first_k 10
    ```
3.  **Output:** Z3 Python scripts in `SAT-LM/tmp/` named `clutrrN.py`.

---

## Step 3: SAT Formulation (Conversion to DIMACS)
Convert Z3 programs into DIMACS CNF files for the SAT solver.

1.  **Configure:** Open `SAT-LM/cluttr_to_sat.py` and verify:
    - `tmp` directory → `SAT-LM/tmp/`
    - `dataset` path → `SAT-LM/data/clutrr_test.json`
    - DIMACS output directory → `main/dimacs/`
2.  **Run:**
    ```bash
    cd SAT-LM
    python3 cluttr_to_sat.py
    ```
3.  **Output:** `clutrrN.cnf`, `pos_clutrrN.cnf`, `neg_clutrrN.cnf`, `clutrrN.mapping` files in your `dimacs/` directory.

---

## Step 4: Initial SAT Solving
Check which CNF problems are satisfiable using CaDiCaL.

1.  **Configure:** Open `main/cadical_solve.py` and verify:
    - `path` → your `dimacs/` folder
    - `output_path` → `dimacs_output/`
    - Path to `cadical` binary
2.  **Run:**
    ```bash
    cd main
    python3 cadical_solve.py
    ```

---

## Step 5: Result Aggregation
Summarise solver outputs into a CSV readable by the ARGOS loop.

1.  **Configure:** Open `main/parse_gen.py` and verify:
    - `directory` → `dimacs/`
    - `log_directory` → `dimacs_output/`
    - `csv_directory` → `main/dimacs_csvs/`
2.  **Run:**
    ```bash
    python3 parse_gen.py
    ```
3.  **Output:** `main/dimacs_csvs/solver_finished.csv`
    - Rows where `pos=SAT, neg=SAT` are the ambiguous cases passed to ARGOS.

---

## Step 6: ARGOS Neuro-Symbolic Refinement Loop
Run the iterative backbone-driven loop to resolve ambiguous cases with an LLM.

1.  **Configure:** Open `main/argos/cot_met_clutrr.py`:
    - Set `USER_PATH` to your project root (e.g., `/mnt/c/Tugas_Akhir/ARGOS_public_anon`)
    - Set `seedrun = 'clutrr_1'` (controls temp folder naming)
    - Verify `dataset` path → `SAT-LM/data/clutrr_test.json`
2.  **Run:**
    ```bash
    cd main
    python3 argos/cot_met_clutrr.py
    ```
3.  **Output:**
    - Accuracy printed to console every iteration
    - `all_outs_cot_met_clutrr_<config>.pkl` saved to project root every 50 steps

---

## Step 7: Analysis
Load the saved `.pkl` file and compute full statistics.

```bash
cd main/analysis
python3 -i run_analysis_clutrr.py
```

**Produces:**
- Overall accuracy, CoT vs SAT breakdown, confusion matrix
- Bootstrap 95% CI, Wilcoxon signed-rank test vs self-consistency baseline
- Confidence trajectory plots, iteration histogram (`clutrr_lenhist.pdf`, `clutrr_threedhist.pdf`)
