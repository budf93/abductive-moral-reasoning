# ARGOS: ExplainEthics Dataset Execution Guide

This guide provides step-by-step instructions to run the ARGOS neuro-symbolic pipeline on the **ExplainEthics** dataset (moral norm reasoning).

## ⚠️ Important: Path Configuration
The key variable to update is `USER_PATH` in `cot_met_explain_ethics.py`. Other paths are derived from it automatically. The dataset uses `/mnt/c/...` WSL-style paths throughout.

## Dataset Overview
ExplainEthics frames moral reasoning as a **decoy detection** problem:
- Each example has a **shown label** (the moral norm being tested, e.g., `violate_fairness`)
- `gt = 'true'`  → the scenario genuinely violates the shown norm
- `gt = 'false'` → the shown norm is a **decoy**; the scenario violates a different norm (`gold_foundation`)
- The model must predict `'true'` or `'false'` (i.e., does the scenario violate the **shown** norm?)

---

## Step 1: Environment Setup
*Same as CLUTRR — see `instructions.md`. Skip if already done.*

---

## Step 2: Logic Generation (SAT-LM)
Translate ethical scenarios into Z3 propositional logic programs.

1.  **Configure:** Create a `.env` file in the project root:
    ```
    HF_TOKEN=hf_xxxx
    ```
2.  **Run:**
    ```bash
    cd SAT-LM
    python3 run_manual.py \
        --task explain_ethics \
        --manual_prompt_id satlm \
        --style_template satlm \
        --run_prediction \
        --eval_split test \
        --engine Qwen/Qwen2.5-Coder-3B-Instruct \
        --first_k 10
    ```
3.  **Output:** Z3 Python scripts in `SAT-LM/tmp/` named `explainethicsN.py`, containing `Implies(A, B)` chains and a `result = s.check(violate_xxx)` query.

---

## Step 3: SAT Formulation (Conversion to DIMACS)
Parse the Z3 scripts and convert propositional implication chains into DIMACS CNF files.

1.  **Configure:** Open `SAT-LM/explain_ethics_to_sat.py` and verify:
    - `tmp_dir` → `SAT-LM/tmp/`
    - `dataset` path → `SAT-LM/data/explainethics_test.json`
    - DIMACS output directory → `main/dimacs/`
2.  **Run:**
    ```bash
    cd SAT-LM
    python3 explain_ethics_to_sat.py
    ```
3.  **Output:**
    - `explainethicsN.cnf`, `pos_explainethicsN.cnf`, `neg_explainethicsN.cnf` in `dimacs/`
    - `explain_ethics_labels.csv` mapping each file to its shown norm label
    - Parsing logic reads `s.add(Implies(A, B))` and `s.add(A)` lines from the Z3 scripts

---

## Step 4: Initial SAT Solving
Check which CNF problems are satisfiable/unsatisfiable using CaDiCaL.

1.  **Configure:** Open `main/cadical_solve.py` and verify:
    - `path` → your `dimacs/` folder
    - `output_path` → `dimacs_output/`
    - Path to `cadical` binary
2.  **Run:**
    ```bash
    cd main
    python3 cadical_solve.py
    ```
3.  **Note on pre-solved cases:**
    - `pos=SAT, neg=UNSAT` → norm IS violated (conclusive, skip ARGOS loop)
    - `pos=UNSAT, neg=SAT` → norm NOT violated (conclusive, skip ARGOS loop)
    - `pos=SAT, neg=SAT`   → ambiguous → passed to ARGOS backbone loop

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

---

## Step 6: ARGOS Neuro-Symbolic Refinement Loop
Run the iterative backbone-driven inference loop using the LLM.

1.  **Configure:** Open `main/argos/cot_met_explain_ethics.py`:
    - Set `USER_PATH` (e.g., `/mnt/c/Tugas_Akhir/ARGOS_public_anon`)
    - Set `seedrun = 'explain_ethics_1'` (controls temp folder naming)
    - Verify `dataset` path → `SAT-LM/data/explainethics_test.json`
    - Verify CSV path → `main/dimacs_csvs/solver_finished.csv`
2.  **Run:**
    ```bash
    cd main
    python3 argos/cot_met_explain_ethics.py
    ```
3.  **Output:**
    - Accuracy printed to console per iteration
    - `all_outs_cot_met_explain_ethics_<config>.pkl` saved every 50 steps
    - The pipeline enters an **interactive `breakpoint()`** at the end — inspect `preds` and `labels` directly

---

## Step 7: Analysis
Load the saved `.pkl` file and compute full statistics.

```bash
cd main/analysis
python3 -i run_analysis_explainethics.py
```

**Produces:**
| Output | Description |
|---|---|
| Overall accuracy | `pred == data[i]['gt']` |
| Confusion matrix | TP/TN/FP/FN for violation detection |
| Per-norm accuracy | Breakdown by `gold_foundation` (care, fairness, loyalty, …) |
| Decoy resistance | How often the model correctly rejects the decoy norm |
| Bootstrap 95% CI | `scipy.stats.t.interval` on resampled accuracy |
| Wilcoxon test | ARGOS vs self-consistency baseline (if CoT iter files present) |
| Trajectory plots | `explainethics_trajectories.pdf`, `explainethics_lenhist.pdf` |

---

## Optional Step 8: Regular Chain-of-Thought (CoT) Baseline
You can also run a standalone CoT baseline (no SAT logic, just pure LLM text generation) to compare against ARGOS. This produces the `FewShotCOTExplainEthics_iter*` files that `run_analysis_explainethics.py` looks for.

1.  **Configure:** Check `main/argos/cot_baseline_explain_ethics.py`. Output paths and models are defined at the top.
2.  **Run (Single Pass):**
    ```bash
    cd main
    python3 argos/cot_baseline_explain_ethics.py --iter 0
    ```
3.  **Run (Self-Consistency / SC):**
    To run 20 independent passes efficiently (loads the model only once):
    ```bash
    cd main
    python3 argos/cot_baseline_explain_ethics.py --start_iter 0 --end_iter 19
    ```
    Once these files exist in `preds/`, they will automatically be read by Step 7's analysis script.

---

## Key File Map

| File | Role |
|---|---|
| `SAT-LM/run_manual.py` | LLM logic generation (Step 2) |
| `SAT-LM/explain_ethics_to_sat.py` | DIMACS conversion (Step 3) |
| `SAT-LM/prog_solver/explain_ethics_solver.py` | Z3 propositional solver |
| `SAT-LM/task_helper.py` → `ExplainEthicsTaskHelper` | Prompt formatting |
| `SAT-LM/task_evaluator.py` → `ExplainEthicsEvaluator` | Prediction scoring |
| `main/argos/cot_met_explain_ethics.py` | Main ARGOS loop (Step 6) |
| `main/analysis/run_analysis_explainethics.py` | Post-run analysis (Step 7) |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `[main] Skipping explainethicsN.cnf: cannot parse index` | Filename replace bug | Check `name.replace('explainethics', '')` in `cot_met_explain_ethics.py` |
| `acc: 0.0000` immediately | `preds` dict empty / dataset path wrong | Verify `explainethics_test.json` exists at `dataset` path |
| `FileNotFoundError` on bbone file | CadiBack failed silently | Check `breakpoint()` in `get_bb()` and inspect the `.bbone` path |
| LLM outputs `Implies(A,B)` not parsed | Capitalisation mismatch | `explain_ethics_to_sat.py` is case-insensitive by default |
