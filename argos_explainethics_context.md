# ARGOS Pipeline: ExplainEthics Reimplementation — Full Context Document

> **Purpose:** This document captures the complete context of the ExplainEthics ARGOS adaptation so a new conversation can continue without losing history. It covers the pipeline architecture, file-by-file reasoning, code differences from the CLUTRR baseline, experimental results, and a post-mortem analysis.

---

## 1. Project Overview

This project reimplements the **ARGOS (Abductive Reasoning with Grounded Output Structures)** neuro-symbolic pipeline — originally validated on the **CLUTRR kinship-reasoning dataset** — for the **ExplainEthics** moral reasoning dataset. The central thesis question is:

> *Does the ARGOS neuro-symbolic refinement loop generalize from deterministic relational domains (kinship) to ambiguous moral reasoning?*

The answer from experiment: **Not effectively**, at least not with a 3B-parameter model and the current prompt design. The detailed results and post-mortem below explain why.

**Repository root:** `/mnt/c/Tugas_Akhir/ARGOS_public_anon/` (WSL) or `C:\Tugas_Akhir\ARGOS_public_anon\` (Windows)

---

## 2. The Full Pipeline — Step by Step

### Step 0: Dataset Curation (`SAT-LM/data/explainethics_test.json`)

The ExplainEthics dataset is sourced from `hard_question.json`, which contains scenarios paired with:
- `context`: A short natural language description of an action (e.g., *"I tore down the birthday decorations for tomorrow."*)
- `explanation`: Ground-truth moral justification text
- `label`: The moral norm being shown (e.g., `violate_care`, `violate_authority`)
- `gt`: Whether the shown label is the **true** norm (`"true"`) or a **decoy** label (`"false"`)

**Key design choice:** Half the dataset items are "decoys" — the shown norm label is *not* the actual norm being violated. This tests if the system can detect when a label is wrong, not just confirm when it's right. This is the most significant structural difference from CLUTRR (which never presents wrong labels).

---

### Step 1: SAT Translation (`SAT-LM/run_manual.py` + `SAT-LM/explain_ethics_to_sat.py`)

**What it does:** The LLM reads each scenario and generates a Python file encoding the moral reasoning as propositional implication chains.

**CLUTRR equivalent:** `SAT-LM/cluttr_to_sat.py` + `SAT-LM/run_manual.py`

**CLUTRR prompt structure:**
```python
relation(Jason, Guillermina) = (grandfather, granddaughter)
relation(Myrna, Guillermina) = (mother, daughter)
return relation(Jason, Myrna)
```

**ExplainEthics prompt structure (from `SAT-LM/manual_prompts/explainethics.jsonline`):**
```python
implies(covering_up_truth, deception) = True
implies(deception, dishonest_behavior) = True
implies(dishonest_behavior, violate_fairness) = True
return violate_fairness(I)
```

**Key design differences:**
| Aspect | CLUTRR | ExplainEthics |
|---|---|---|
| Logic type | Relational (`R(A,B) = label`) | Propositional (`implies(X,Y) = True`) |
| Variable names | Named persons + relationships | Moral concept predicates |
| Query variable | Specific relation between two people | Boolean: does `violate_X` hold? |
| Few-shot support | `clutrr.jsonline` (8 examples) | `explainethics.jsonline` (3 examples, one per norm) |
| Prompt includes | Context only | Context + Explanation (from ground truth) |

**Output:** `SAT-LM/tmp/explainethicsN.py` — a Python pseudocode file with `implies()` statements.

**Why this design:** Moral reasoning is fundamentally *causal and transitive* (action → concept → norm violation), mirroring how kinship chains work (grandfather → granddaughter → mother). The `implies()` primitive directly encodes this transitivity.

---

### Step 2: DIMACS CNF Conversion (`SAT-LM/explain_ethics_to_sat.py`)

**What it does:** Parses the `implies()` statements from the generated `.py` files and converts them into DIMACS CNF format for the SAT solver. Uses `sympy` for Boolean formula manipulation.

**CLUTRR equivalent:** `SAT-LM/cluttr_to_sat.py`

**Key differences:**
| Aspect | CLUTRR | ExplainEthics |
|---|---|---|
| Variable space | `PeopleSort` enum + relational function `R()` | Simple propositional symbols |
| Query construction | `R(personA, personB) = relation_label` | `violate_X = True` |
| Label file output | `main/clutrr_new_labels.csv` | `main/explain_ethics_labels.csv` |
| CNF pair | `pos_clutrrN.cnf` + `neg_clutrrN.cnf` | `pos_explainethicsN.cnf` + `neg_explainethicsN.cnf` |

**Two CNF files per example:**
- **`pos_` file:** Asserts the shown norm label as `True` (hypothesis: label is correct)
- **`neg_` file:** Asserts the shown norm label as `False` (counter-hypothesis: label is wrong/decoy)

This polarity test is the mathematical heart of ARGOS:
- `pos=SAT, neg=UNSAT` → label is definitively proven correct → backbone pre-solve
- `both=SAT` → the formula is too weak/ambiguous → falls to the refinement loop

**Output:** `main/dimacs/pos_explainethicsN.cnf` and `main/dimacs/neg_explainethicsN.cnf`

---

### Step 3: SAT Solving (`main/cadical_solve.py`)

**Unchanged from CLUTRR.** Feeds all `.cnf` files into the CaDiCaL SAT solver (C++ binary) and writes verdict output files to `main/dimacs_output/`.

**Output:** `main/dimacs_output/pos_explainethicsN.out` — raw solver verdicts (`SAT`/`UNSAT`/`TIMEOUT`)

---

### Step 4: Result Aggregation (`main/parse_gen.py`)

**Unchanged from CLUTRR.** Reads all `.out` solver files and categorizes each example by `(pos_result, neg_result)` pair. Writes the master spreadsheet.

**Output:** `main/dimacs_csvs/solver_finished.csv`

**Formula for pre-solving:**
- `pos=SAT, neg=UNSAT` → definitively proved → backbone handles it (no LLM needed)
- Otherwise → goes to the `next_var()` refinement loop

---

### Step 5: ARGOS Refinement Loop (`main/argos/cot_met_explain_ethics.py`)

This is the main script. It mirrors `cot_met_clutrr.py` structurally with every domain-specific part adapted.

#### 5a. Pre-Solving
When `solver_finished.csv` marks an example as pre-solved, the result is injected directly:
```python
all_outs[row[1]] = (vv, {'pos': ['dummy'], 'neg': []}, None, False, {}, False, [], [])
```
The dummy `solout` dict is essential — the analysis script runs `len(solout['neg'])` which would crash on a string.

#### 5b. The `next_var()` Loop
For each non-pre-solved example:
1. Extract backbone (propositions fixed across all SAT solutions)
2. Map variable IDs → predicate names using `.maptxt` mapping file (e.g., `"deception_"`)
3. Pick two undetermined predicates from the backbone
4. Ask LLM: *"Is there an ethical implication rule linking [predA] and [predB]?"*
5. If yes (passes `rule_check()`), add it as a new SAT clause and re-solve the backbone
6. If backbone becomes conclusive OR loop limit exceeded → exit or fall to CoT

**CLUTRR variable extraction:** Looks for person name pairs and constructs kinship relation triples  
**ExplainEthics variable extraction:** Looks for propositional predicate names (strips trailing `_` from stored names like `"deception_"`)

#### 5c. The `cot()` Fallback
When the refinement loop can't resolve the example, a standalone Chain-of-Thought prompt is used:
- Builds a prompt with the scenario context + any accumulated rules
- Asks the LLM to reason through whether the norm is violated
- Parses the response for keywords like "Therefore, Yes/No" or falls back to NLI scoring

#### 5d. The `yn()` Scoring Function
Used to score yes/no confidence via log-probabilities:
```python
log_p_yes = model.generate(" yes", given prefix)
log_p_no  = model.generate(" no", given prefix)
score = softmax([log_p_yes, log_p_no])
```
**CoT threshold (`cot_thresh=1.0`):** Only stops early if `max(score) >= 1.0` — effectively never, since softmax can never reach 1.0, meaning the loop always runs to `looplim`.

#### 5e. The `rule_check()` Filter
Before adding any LLM-generated rule to the SAT formula, it passes two plausibility checks:
1. **Moral plausibility:** "Does this rule seem morally true?" (logprob-scored)
2. **Context relevance:** "Is this rule relevant to the given scenario?" (logprob-scored)

Both use `rulethresh=0.3` — a very permissive threshold.

**Output:** `all_outs_cot_met_explain_ethics_<config>.pkl`

**PKL tuple structure (8-element):**
```
(vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts)
[0]  [1]     [2]    [3]          [4]          [5]       [6]  [7]
```

---

### Step 6: Analysis (`main/analysis/run_analysis_explainethics.py`)

Loads the `.pkl` file and `explain_ethics_labels.csv`, then computes:
- Overall accuracy, SAT backbone accuracy, CoT fallback accuracy
- Confusion matrix (TP, FP, TN, FN)
- Per-norm accuracy breakdown
- Decoy resistance rate
- Self-consistency baseline comparison (loads `preds/FewShotCOTExplainEthics_iter*`)
- Bootstrap confidence intervals (scikit-learn `resample`)
- Wilcoxon signed-rank test (ARGOS vs SC baseline)
- Trajectory flip analysis (how many answers change between iterations)

**Outputs saved to:** `main/analysis/analysis_outputs/`
- `explainethics_analysis_report.txt` (text metrics)
- `explainethics_results.png` (summary bar chart)
- `explainethics_trajectories.pdf`
- `explainethics_good_flips.pdf`
- `explainethics_bad_flips.pdf`
- `explainethics_lenhist.pdf`
- `explainethics_threedhist.pdf`

---

## 3. Experimental Results (67-example subset, Qwen 2.5 3B)

**Config:** `rulethresh=0.3, cot_thresh=1.0, dynamic=True, llama3B, no rules in prompt`

```
Total preds: 67
  SAT backbone: 19 (28%)
  CoT fallback: 48 (72%)
  Missed:        0

Overall accuracy : 0.4627  (31/67)
Correct via SAT  : 10/19 = 52.6%
Correct via CoT  : 21/48 = 43.8%

Confusion matrix:
  TP=25  FP=29  TN=6  FN=7
  n_true=32  n_false=35 (decoy examples)

Per-norm accuracy (gold foundation):
  violate_care       23/48 = 47.9%
  violate_fairness    3/6  = 50.0%
  violate_authority   5/12 = 41.7%
  violate_sanctity    0/1  =  0.0%

Decoy resistance  : 6/35  = 17.1%    ← critical failure
True-label recall : 25/32 = 78.1%

Self-consistency baseline: 35/67 = 52.2%
Bootstrap ARGOS mean: 46.1%
Bootstrap SC mean:    51.7%

Wilcoxon p=1.0000 — ARGOS NOT significantly better than SC baseline

Trajectory flags:
  Unflipped wrong:   18
  Unflipped correct: 16
  Flipped correct:    5   ← loop helped
  Flipped incorrect:  9   ← loop hurt (net -4)
```

---

## 4. Post-Mortem Analysis

### 4.1 Why ARGOS Underperformed

**Primary cause — Decoy Label Architecture Mismatch:**
ARGOS is designed for domains where the label/hypothesis is always correct. It receives a norm label (e.g., `violate_fairness`) and builds the *entire* SAT formula to prove/disprove it. When the label is a decoy, ARGOS is trying to prove something false using a biased formula — it unintentionally amplifies the wrong hypothesis.

**Secondary cause — Model Moral Sycophancy (3B bias):**
The Qwen 3B model defaults to predicting `"true"` (violated) for any ethically charged scenario. The confusion matrix shows 54/67 predictions were `"true"` — only 6 TN out of 35 decoy examples. The model lacks the nuance to distinguish between *mentioning* a sensitive topic and *actually violating a norm*.

**Tertiary cause — `cot_thresh=1.0` Renders SAT Backbone Useless:**
With `cot_thresh=1.0`, the confidence threshold can never be met (softmax < 1.0 always). Every example that enters the loop runs to `looplim` and falls back to CoT. The SAT backbone's refinement is effectively bypassed for 72% of examples.

**Quaternary cause — Constraint-Correction Overload:**
The refinement loop feeds mathematical contradiction feedback to the LLM. A 3B model lacks the capacity to process strict symbolic feedback without hallucinating new mistakes. Evidence: 9 flipped-incorrect vs 5 flipped-correct (net loss of 4 correct answers).

**Structural cause — Boolean Logic vs Moral Nuance:**
Moral norms are contextual, probabilistic, and culturally dependent. Forcing them into Boolean SAT constraints (`True`/`False` only) eliminates the very ambiguity that makes ethical reasoning hard — and correct. CLUTRR questions have ground-truth mathematical proofs; ethics questions often don't.

**Methodological cause — `yn()` Logprob Bias:**
The yes/no scoring function (`yn()`) uses log-probability of the tokens ` yes` vs ` no`. For small models on ethics questions, this is vulnerable to position bias (the token appearing first in training) and framing bias (vocabulary in the prompt emotionally biasing the softmax). The scores don't reliably reflect the model's actual certainty.

---

### 4.2 What Worked

- The **pipeline architecture is sound** — all 6 steps execute without crashes on the full dataset
- The **propositional implication chain** design cleanly maps to DIMACS CNF format
- Pre-solving via SAT backbone works (trivially satisfiable examples do get resolved correctly at ~52.6%)
- The **analysis script** is comprehensive and matches the metrics used in the original ARGOS paper
- **Decoy resistance metric** revealed a fundamentally important weakness that wouldn't be visible in a non-decoy dataset

---

## 5. Code Comparison: CLUTRR vs ExplainEthics

| File | CLUTRR | ExplainEthics | Key Difference |
|---|---|---|---|
| Few-shot prompt | `clutrr.jsonline` | `explainethics.jsonline` | Relational → Propositional logic; includes `# Explanation:` field |
| SAT converter | `cluttr_to_sat.py` | `explain_ethics_to_sat.py` | `PeopleSort` enum removed; `implies()` clause instead of `R()` |
| Label file | `clutrr_new_labels.csv` | `explain_ethics_labels.csv` | Kinship labels → moral norm labels |
| Main loop | `cot_met_clutrr.py` | `cot_met_explain_ethics.py` | Variable names, query format, dataset path |
| Variable mapping | Person name pairs | Predicate names (with trailing `_` stripped) | e.g., `"deception_"` → `"deception"` |
| Seedrun prefix | `clutrr_1` | `explain_ethics_1` | Prevents workfile conflicts |
| PKL output | `all_outs_cot_met_clutrr_*.pkl` | `all_outs_cot_met_explain_ethics_*.pkl` | Same 8-tuple structure |
| Dataset file | `SAT-LM/data/clutrr_test.json` | `SAT-LM/data/explainethics_test.json` | Different JSON schema; ethics has `explanation` field |
| Analysis script | `run_analysis_clutrr.py` | `run_analysis_explainethics.py` | Adds decoy resistance metric, per-norm accuracy |

---

## 6. File Map

```
ARGOS_public_anon/
├── SAT-LM/
│   ├── data/
│   │   └── explainethics_test.json          # Source dataset (145 items)
│   ├── manual_prompts/
│   │   ├── clutrr.jsonline                  # Few-shot prompts for CLUTRR
│   │   └── explainethics.jsonline           # Few-shot prompts for ExplainEthics
│   ├── tmp/
│   │   └── explainethicsN.py                # LLM-generated implies() programs
│   ├── cluttr_to_sat.py                     # CLUTRR SAT converter
│   ├── explain_ethics_to_sat.py             # ExplainEthics SAT converter [NEW]
│   └── run_manual.py                        # Orchestrates LLM generation + SAT conversion
│
├── main/
│   ├── dimacs/
│   │   ├── pos_explainethicsN.cnf           # Positive hypothesis CNF
│   │   └── neg_explainethicsN.cnf           # Negative hypothesis CNF
│   ├── dimacs_output/
│   │   └── *.out                            # CaDiCaL solver verdicts
│   ├── dimacs_csvs/
│   │   └── solver_finished.csv              # Aggregated solver results
│   ├── explain_ethics_labels.csv            # Ground-truth label map
│   ├── cadical_solve.py                     # SAT solver runner (unchanged)
│   ├── parse_gen.py                         # Result aggregator (unchanged)
│   ├── argos/
│   │   ├── cot_met_clutrr.py               # Original ARGOS loop (CLUTRR)
│   │   └── cot_met_explain_ethics.py        # Adapted ARGOS loop (ExplainEthics) [NEW]
│   └── analysis/
│       ├── run_analysis_clutrr.py           # CLUTRR analysis
│       ├── run_analysis_explainethics.py    # ExplainEthics analysis [NEW]
│       ├── explainethics_analysis_report.txt # Text metrics (old location)
│       └── analysis_outputs/               # All figures + report (new location)
│           ├── explainethics_analysis_report.txt
│           ├── explainethics_results.png
│           └── *.pdf                        # Trajectory/histogram plots
│
├── preds/
│   └── FewShotCOTExplainEthics_iter*.json  # CoT baseline iteration files
├── all_outs_cot_met_explain_ethics_*.pkl   # ARGOS pipeline output (root dir)
└── kaggle.md                               # Step-by-step Kaggle deployment guide
```

---

## 7. Open Research Questions

1. **RQ1 — SAT Translation Quality:** Does the LLM faithfully encode the moral reasoning, or does it hallucinate incorrect `implies()` chains that produce trivially satisfiable formulas?
2. **RQ2 — Threshold Sensitivity:** How does `cot_thresh` values (0.3, 0.5, 0.7) affect SAT backbone utilization and final accuracy?
3. **RQ3 — `yn()` Reliability:** Do logprob yes/no scores correlate with actual correctness, or is position/framing bias the bottleneck?
4. **RQ4 — Refinement Loop Quality:** Which specific refinement iterations cause correct→incorrect flips? Is it the SAT feedback format or the model's capacity?
5. **RQ5 — Decoy Handling:** Can explicit skeptical prompting ("the shown label may be wrong") improve decoy resistance without hurting true-label recall?

---

## 8. Proposed Next Steps

1. **Threshold sweep:** Run pipeline with `--cot_thresh 0.3`, `0.5`, `0.7` and compare results
2. **Skeptical prompt injection:** Add a decoy-awareness sentence to `explainethics.jsonline`
3. **Longer implication chains:** Rewrite few-shot examples to produce 5–6 `implies()` steps
4. **Two-stage prompting:** Separate norm-agnostic translation from norm verification
5. **`yn()` ablation:** Replace logprob scoring with direct `"Answer true or false:"` prompt parsing
6. **Scale up:** Test the full 145-example dataset (after fixing the above issues) to see if results generalize

---

## 9. Environment Notes

- **Model:** Qwen 2.5 Coder 3B Instruct (quantized 4-bit via BitsAndBytesConfig on Kaggle T4 GPU)
- **Kaggle notebook:** `fikribudianto/argos-kaggle-notebook`
- **Local path (WSL):** `/mnt/c/Tugas_Akhir/ARGOS_public_anon/`
- **Kaggle path:** `/kaggle/working/argos-kaggle/`
- **Key env variable:** `USER_PATH` in `cot_met_explain_ethics.py` (line 47) — must match environment
- **Kaggle deployment guide:** See `kaggle.md` for step-by-step cell execution order
