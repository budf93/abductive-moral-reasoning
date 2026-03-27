# ARGOS: CLUTRR Dataset Execution Guide

This guide provides the step-by-step instructions to run the ARGOS neuro-symbolic pipeline using the **CLUTRR** dataset.

## ⚠️ Important: Path Configuration
Most scripts in this repository contain hardcoded absolute paths (e.g., `C:/Tugas Akhir/...` or `/home/XXXX/...`). Before running each step, you **must** open the corresponding Python file and update variables like `USER_PATH`, `path`, and `dataset` to match your local workspace.

---

## Step 1: Environment Setup
1.  **Setup Virtual Environment (Recommended):**
    *   **Windows (PowerShell):**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **Unix / Linux / macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
2.  **Install Python Dependencies:**
    Installing `flash_attn` requires `torch` and `packaging` to be pre-installed. Run these commands in order:
    ```bash
    # Step A: Install core build dependencies
    pip install packaging ninja wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    # Step B: Install the rest of the requirements
    pip install -r requirements.txt
    ```
3.  **Compile SAT Tools:**
    *   **CaDiCaL:** 
        Navigate to `sat_gen/sat_tools/postprocess/cadical/` and run:
        ```bash
        ./configure && make
        ```
        The executable will be generated at `sat_gen/sat_tools/postprocess/cadical/build/cadical`. 
        *Note: Requires a Unix-like environment (Linux, macOS, or WSL on Windows) with GNU Make.*
    *   **CadiBack:** 
        Navigate to `main/` and set up the dependency link, then compile:
        ```bash
        # Create link to the compiled CaDiCaL
        cd main
        ln -s ../sat_gen/sat_tools/postprocess/cadical cadical
        
        # Compile CadiBack
        cd cadiback
        # Manually create config if git is not present
        echo '#define VERSION "1.5.0"' > config.hpp
        echo '#define DATE "'$(date)'"' >> config.hpp
        echo '#define IDENTIFIER "manual"' >> config.hpp
        echo '#define GITID "manual"' >> config.hpp
        echo '#define BUILD "g++ -Wall -O3 -DNDEBUG"' >> config.hpp
        
        ./configure
        make
        ```

---

## Step 2: Logic Generation (SAT-LM)
Translate natural language stories into symbolic logic programs.
1.  **Configure:** Create a `.env` file in the project root and set your HuggingFace token: `HF_TOKEN=hf_xxxx`.
2.  **Run:**
    ```powershell
    cd SAT-LM
    python3 run_manual.py --task clutrr --num_dev 1 --manual_prompt_id satlm --style_template satlm --run_prediction
    ```
3.  **Output:** Python logic scripts will appear in `SAT-LM/tmp/`.

---

## Step 3: SAT Formulation (Conversion to DIMACS)
Convert the logic programs into DIMACS CNF format.
1.  **Configure:** Open `SAT-LM/cluttr_to_sat.py`. Update paths for `os.listdir` (pointing to `SAT-LM/tmp/`), `js` (pointing to `SAT-LM/data/clutrr_test.json`), and the output `dimacs` directory.
2.  **Run:**
    ```powershell
    python cluttr_to_sat.py
    ```
3.  **Output:** `.cnf`, `.mapping`, and `.maptxt` files in your specified DIMACS directory.

---

## Step 4: Initial SAT Solving
Check for initial consistency using the CaDiCaL solver.
1.  **Configure:** Open `main/cadical_solve.py`. Update `path` (your DIMACS folder), `output_path` (for logs), and the absolute path to your `cadical` binary.
2.  **Run:**
    ```powershell
    cd ../main
    python cadical_solve.py
    ```

---

## Step 5: Result Aggregation
Summarize the solver outputs into a format readable by the ARGOS loop.
1.  **Configure:** Open `main/parse_gen.py`. Update `directory` (DIMACS folder), `log_directory` (solver logs), and `csv_directory`.
2.  **Run:**
    ```powershell
    python parse_gen.py
    ```
3.  **Output:** `solver_finished.csv` in your specified CSV directory.

---

## Step 6: ARGOS Neuro-Symbolic Refinement
Run the iterative refinement loop to resolve uncertainties using an LLM.
1.  **Configure:** Open `main/argos/cot_met_clutrr.py`.
    *   Update paths for `dataset` (`clutrr_test.json`), `c` (`solver_finished.csv`), and the `dimacs` directory.
    *   Set `seedrun` (e.g., `clutrr_1`) to distinguish temporary working folders.
2.  **Run:**
    ```powershell
    python argos/cot_met_clutrr.py
    ```
3.  **Output:** Final accuracy results and detailed reasoning traces.
