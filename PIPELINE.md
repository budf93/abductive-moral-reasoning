# ARGOS: Neuro-Symbolic SAT Pipeline

This document describes the end-to-end pipeline for ARGOS (Abductive Reasoning with Generalization Over Symbolics), a neuro-symbolic framework that combines Large Language Models (LLMs) with SAT solvers to solve complex logical reasoning tasks.

## 1. Logic Generation & Preprocessing
*   **Input:** Natural language reasoning datasets (e.g., CLUTRR, Folio, ProntoQA).
*   **Action:** The system uses an LLM to translate natural language premises and questions into symbolic logic programs (Python scripts using Z3-like syntax).
*   **Location:** `SAT-LM/`
*   **Intermediate Output:** Python logic files in `SAT-LM/tmp/`.

## 2. SAT Conversion
*   **Input:** Python logic programs.
*   **Action:** The symbolic logic is converted into DIMACS CNF (Conjunctive Normal Form) format.
*   **Hypotheses:** For every problem, two instances are created:
    *   **Positive (`pos`):** Assumes the query is True.
    *   **Negative (`neg`):** Assumes the query is False.
*   **Key Scripts:** `[dataset]_to_sat.py` scripts in the root or `SAT-LM/`.
*   **Mapping:** Generates `.mapping` and `.arity` files to track the relationship between SAT variables and logical predicates.

## 3. SAT Solving & Backbone Extraction
*   **Input:** CNF files.
*   **Action:** 
    *   **Solving:** The **CaDiCaL** SAT solver determines if the hypotheses are satisfiable (SAT) or unsatisfiable (UNSAT).
    *   **Backbones:** The **cadiback** tool identifies "backbones"—literals that must be True in every satisfying assignment of the formula.
*   **Key Scripts:** `main/cadical_solve.py`, `main/cadiback/`.

## 4. ARGOS Neuro-Symbolic Loop
*   **Core Logic:** If the SAT solver cannot immediately reach a conclusion, ARGOS enters an iterative refinement loop.
*   **Mechanism:**
    1.  **Identify Uncertainty:** It looks for variables that are not yet part of the backbone (undecided).
    2.  **LLM Guidance:** It uses the LLM (via Chain-of-Thought) to check the plausibility of these undecided variables or to generate new "common-sense" rules.
    3.  **Refinement:** These new rules are converted into SAT clauses and appended to the original CNF.
    4.  **Re-solve:** The solver runs again on the augmented CNF.
*   **Key Scripts:** `main/argos/cot_met_[dataset].py`.

## 5. Verification & Analysis
*   **Final Output:** The pipeline determines the truth value of the original question based on which hypothesis (`pos` or `neg`) remains consistent.
*   **Evaluation:** Results are parsed and summarized into CSVs for performance metrics.
*   **Location:** `main/parse_gen.py` and notebooks in `main/analysis/`.
