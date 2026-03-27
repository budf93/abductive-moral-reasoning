"""
explain_ethics_to_sat.py
=========================
ExplainEthics adaptation of clutrr_to_sat.py.

Purpose
-------
This script reads LLM-generated Python files from SAT-LM/tmp/ (named like
explainethics{idx}.py) that contain implies(X, Y) = True statements, and
converts them into DIMACS CNF files in main/dimacs/.

Two polarity variants are written for each example:
  - pos_{filename}.cnf  (hypothesis asserted as True)
  - neg_{filename}.cnf  (hypothesis asserted as False / negated)

These files are later consumed by cadical_solve.py (unchanged) to test
satisfiability, and the results feed into cot_met_explain_ethics.py.

Key differences from clutrr_to_sat.py
--------------------------------------
- Variables are propositional (Bool), not relational (R(A,B)==relation).
- The query variable is the moral norm label (e.g. 'violate_care_').
- No PeopleSort enum or relational function R() is needed.
- The ExplainEthics dataset is loaded from hard_question.json.
- Labels are written to explain_ethics_labels.csv instead of clutrr_new_labels.csv.

Usage
-----
    cd SAT-LM/
    python explain_ethics_to_sat.py

# >>> [ExplainEthics Adaptation] BEGIN: Module
"""

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import to_cnf, And, Or, Not
from sympy import symbols
from copy import deepcopy
import re as regex
import os
import json
import random
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reuse the DimacsMapping and DimacsFormula helper classes from clutrr_to_sat.
# We copy them here so this file is self-contained and does not require runtime
# modifications to clutrr_to_sat.py (which has module-level side effects).
# ---------------------------------------------------------------------------
class DimacsMapping:
    """Maps symbolic variable names to integer DIMACS variable IDs."""
    def __init__(self):
        self._symbol_to_variable = {}
        self._variable_to_symbol = {}
        self._total_variables = 0

    @property
    def total_variables(self):
        return self._total_variables

    def new_variable(self):
        self._total_variables += 1
        return self._total_variables

    def get_variable_for(self, symbol):
        result = self._symbol_to_variable.get(symbol)
        if result is None:
            result = self.new_variable()
            self._symbol_to_variable[symbol] = result
            self._variable_to_symbol[result] = symbol
        return result

    def get_symbol_for(self, variable):
        return self._variable_to_symbol[variable]

    def __str__(self) -> str:
        return str(self._variable_to_symbol)


class DimacsFormula:
    """Holds a DimacsMapping and a list of clauses, renders as DIMACS text."""
    def __init__(self, mapping, clauses):
        self._mapping = mapping
        self._clauses = clauses

    @property
    def mapping(self):
        return self._mapping

    @property
    def clauses(self):
        return self._clauses

    def __str__(self):
        header = f"p cnf {self._mapping.total_variables} {len(self._clauses)}"
        body = "\n".join(
            " ".join([str(literal) for literal in clause] + ["0"])
            for clause in self._clauses
        )
        return "\n".join([header, body])


def to_dimacs_formula(sympy_cnf, dimacs_mapping=None):
    """
    Convert a sympy CNF expression to a DimacsFormula.
    Identical logic to the same function in clutrr_to_sat.py.
    """
    print(f"[explain_ethics_to_sat] sympy_cnf: {sympy_cnf}")
    if dimacs_mapping is None:
        dimacs_mapping = DimacsMapping()
    dimacs_clauses = []

    assert type(sympy_cnf) == And, f"Expected And, got {type(sympy_cnf)}"

    for sympy_clause in sympy_cnf.args:
        dimacs_clause = []
        if type(sympy_clause) != Or:
            # Single literal clause (unit clause)
            sympy_literal = sympy_clause
            if type(sympy_literal) == Not:
                sympy_symbol, polarity = sympy_literal.args[0], -1
            elif type(sympy_literal) == Symbol:
                sympy_symbol, polarity = sympy_literal, 1
            else:
                raise AssertionError("invalid cnf literal")
            dimacs_variable = dimacs_mapping.get_variable_for(sympy_symbol)
            dimacs_clause.append(dimacs_variable * polarity)
            dimacs_clauses.append(dimacs_clause)
            continue

        for sympy_literal in sympy_clause.args:
            if type(sympy_literal) == Not:
                sympy_symbol, polarity = sympy_literal.args[0], -1
            elif type(sympy_literal) == Symbol:
                sympy_symbol, polarity = sympy_literal, 1
            else:
                raise AssertionError("invalid cnf literal")
            dimacs_variable = dimacs_mapping.get_variable_for(sympy_symbol)
            dimacs_literal = dimacs_variable * polarity
            dimacs_clause.append(dimacs_literal)

        dimacs_clauses.append(dimacs_clause)

    return DimacsFormula(dimacs_mapping, dimacs_clauses)


# ---------------------------------------------------------------------------
# Paths — using Windows-native paths since this runs on the project machine.
# These mirror the conventions established in clutrr_to_sat.py.
# ---------------------------------------------------------------------------
TMP_DIR      = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp'           # LLM output .py files
DIMACS_DIR   = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs'          # output .cnf files
DATASET_PATH = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/data/explainethics_test.json'
LABELS_CSV   = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/explain_ethics_labels.csv'  # ground-truth labels


def parse_implies_lines(lines):
    """
    Extract implication pairs and action facts from Z3 Python files in tmp/.

    These files are generated by explain_ethics_solver.py and look like:

        from z3 import *
        covering_up_truth = Bool('covering_up_truth')
        deception = Bool('deception')
        ...
        s = Solver()
        s.add(Implies(covering_up_truth, deception))
        s.add(covering_up_truth)          # action fact (hard True)
        result = s.check(violate_fairness)
        if result == sat:
            print('violate_fairness')

    We parse:
      - s.add(Implies(A, B)) -> implication pair (A, B)
      - s.add(A) (single arg, not Implies) -> action fact A is True

    Returns
    -------
    (terms: list[str], pairs: list[(str, str)], action_terms: list[str])
    """
    all_terms = set()
    pairs = []         # (antecedent, consequent) from Implies() lines
    action_terms = []  # predicate terms asserted True (known facts)

    for line in lines:
        line = line.strip()
        # Normalize to lowercase for matching — handles both Implies and implies
        line_lower = line.lower()

        if line_lower.startswith('s.add(implies('):
            # Parse: s.add(Implies(antecedent, consequent))
            try:
                # Extract inner args of Implies(...) regardless of casing
                inner = line_lower.split('s.add(implies(')[1].split('))')[0]
                parts = [p.strip() for p in inner.split(',')]
                if len(parts) == 2:
                    # Sanitize: replace hyphens with underscores for valid Python identifiers
                    ant  = parts[0].replace('-', '_')
                    cons = parts[1].replace('-', '_')
                    all_terms.add(ant)
                    all_terms.add(cons)
                    pairs.append((ant, cons))
            except Exception:
                continue

        elif line_lower.startswith('s.add(') and 'implies(' not in line_lower:
            # Parse: s.add(predicate) — a hard True fact (action from context)
            try:
                # Extract the predicate name inside s.add(...)
                inner = line_lower.split('s.add(')[1].rstrip(')')
                pred = inner.strip().replace('-', '_')
                if pred:
                    all_terms.add(pred)
                    action_terms.append(pred)
            except Exception:
                continue

    return list(all_terms), pairs, action_terms


def get_query_var_from_file(lines):
    """
    Extract the target moral norm variable from the Z3 file.
    Looks for: result = s.check(violate_xxx)

    Returns the norm string (e.g. 'violate_fairness') or None.
    """
    for line in lines:
        line_lower = line.strip().lower()
        # Match: result = s.check(violate_fairness)
        if line_lower.startswith('result = s.check('):
            try:
                norm = line_lower.split('result = s.check(')[1].rstrip(')')
                if norm:
                    return norm.replace('-', '_')
            except Exception:
                pass
    return None



def build_sympy_formula(terms, pairs, action_terms, query_var, negate_query=False):
    """
    Build a sympy Boolean formula from the parsed implies() pairs.

    The formula asserts:
      - All action() predicates as True (unit clauses)
      - Each implies(A, B) as the implication A → B (i.e. ¬A ∨ B in CNF)
      - The query variable as True (pos) or False (neg) for the polarity check

    Parameters
    ----------
    terms        : list of propositional variable name strings
    pairs        : list of (antecedent, consequent) tuples from implies() lines
    action_terms : list of predicates asserted as True by action() lines
    query_var    : the norm label variable (e.g. 'violate_care_')
    negate_query : if True, assert NOT(query_var); if False, assert query_var

    Returns
    -------
    sympy And expression representing the full CNF formula, or None on failure
    """
    # Declare one sympy Symbol per unique term
    sym_dict = {}
    for term in terms:
        # We append trailing '_' to variable names to match clutrr_to_sat convention
        safe_name = term if term.endswith('_') else term + '_'
        sym_dict[term] = symbols(safe_name)

    # Ensure the query variable has a symbol even if not seen in pairs
    if query_var not in sym_dict:
        safe_name = query_var if query_var.endswith('_') else query_var + '_'
        sym_dict[query_var] = symbols(safe_name)

    conjuncts = []

    # Assert each known action predicate as True (hard fact from the story)
    for at in action_terms:
        if at in sym_dict:
            conjuncts.append(sym_dict[at])

    # Convert each implies(A, B) to its sympy Implies form (¬A ∨ B).
    # Sympy's to_cnf() will further transform it into strict Conjunctive Normal Form.
    from sympy import Implies
    for (ant, cons) in pairs:
        if ant in sym_dict and cons in sym_dict:
            conjuncts.append(Implies(sym_dict[ant], sym_dict[cons]))

    # Append the query literal (positive or negative polarity)
    q_sym = sym_dict[query_var]
    if negate_query:
        conjuncts.append(Not(q_sym))      # neg polarity: hypothesis is False
    else:
        conjuncts.append(Not(Not(q_sym))) # pos polarity: hypothesis is True (double-negation keeps it non-trivial)

    if not conjuncts:
        return None

    formula = And(*conjuncts)
    return formula


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
# Load the ExplainEthics dataset (list of dicts with keys: context, explanation, label, gt, gold_foundation, id)
print(f"[explain_ethics_to_sat] Loading dataset from {DATASET_PATH}")
with open(DATASET_PATH, 'r') as df:
    data = json.loads(df.read())

# Process each LLM-generated .py file in the tmp directory
for file in tqdm(os.listdir(TMP_DIR)):
    skip_problem = False

    # Only process files named explainethics{idx}.py
    # (clutrr files have names like clutrr{idx}.py — this check separates them)
    if not file.startswith('explainethics') or not file.endswith('.py'):
        print(f"[explain_ethics_to_sat] Skipping non-ethics file: {file}")
        continue

    # Parse the index from the filename to look up the dataset entry
    try:
        idx = int(file.replace('explainethics', '').replace('.py', ''))
    except ValueError:
        print(f"[explain_ethics_to_sat] Could not parse index from: {file}")
        continue

    if idx >= len(data):
        print(f"[explain_ethics_to_sat] Index {idx} out of range")
        continue

    prob = data[idx]

    # The label field holds the predicted norm from the LLM's translation step.
    # We append '_' to match the variable naming convention used in sympy symbol names.
    query_var = prob['label'].replace('-', '_').strip()
    gt        = prob.get('gt', 'true')  # 'true' or 'false' — whether the label is correct

    print(f"[explain_ethics_to_sat] Processing {file}, query={query_var}, gt={gt}")

    full_path = os.path.join(TMP_DIR, file)
    lines = open(full_path, 'r').readlines()

    # Parse all Implies() and action facts from the Z3 Python file
    terms, pairs, action_terms = parse_implies_lines(lines)

    # Prefer the query variable from the Z3 file itself (most reliable),
    # fall back to the dataset 'label' field if not found
    query_var_from_file = get_query_var_from_file(lines)
    if query_var_from_file:
        query_var = query_var_from_file
        print(f"[explain_ethics_to_sat] query_var from file: {query_var}")
    else:
        print(f"[explain_ethics_to_sat] query_var from dataset label: {query_var}")
    if not pairs and not action_terms:
        # LLM produced no usable logic — skip this file
        print(f"[explain_ethics_to_sat] No logic found in {file}, skipping")
        continue

    # Add the query variable itself to the term list so it gets a symbol
    if query_var not in terms:
        terms.append(query_var)

    # Write one DIMACS file for each polarity (pos=hypothesis True, neg=hypothesis False).
    # The SAT solver tests both; if pos=SAT and neg=UNSAT we conclude query_var is entailed.
    f_dimacs = None
    for q in ['pos', 'neg']:
        negate = (q == 'neg')

        try:
            formula = build_sympy_formula(terms, pairs, action_terms, query_var, negate_query=negate)
            if formula is None:
                print(f"[explain_ethics_to_sat] Empty formula for {file} polarity={q}")
                skip_problem = True
                break
            cnf = to_cnf(formula)   # convert to strict CNF using sympy
        except Exception as e:
            print(f"[explain_ethics_to_sat] CNF conversion failed for {file}: {e}")
            skip_problem = True
            break

        try:
            if q == 'neg':
                # Reuse the mapping from the pos formula to keep variable IDs consistent
                f_dimacs = to_dimacs_formula(cnf, dimacs_mapping=f_dimacs.mapping)
            else:
                f_dimacs = to_dimacs_formula(cnf)
        except Exception as e:
            print(f"[explain_ethics_to_sat] to_dimacs_formula failed for {file} polarity={q}: {e}")
            skip_problem = True
            break

        # Write the DIMACS CNF file using the same naming convention as clutrr_to_sat:
        # {polarity}_{filename_without_extension}.cnf
        dimacs_pth = os.path.join(DIMACS_DIR, q + '_' + file[:-3] + '.cnf')
        print(f"[explain_ethics_to_sat] Writing DIMACS: {dimacs_pth}")
        with open(dimacs_pth, 'w') as dimacs_f:
            dimacs_f.write(str(f_dimacs))

        # Write the human-readable mapping file (variable_id → symbol_name)
        maptxt_pth = os.path.join(DIMACS_DIR, q + '_' + file[:-3] + '.maptxt')
        with open(maptxt_pth, 'w') as maptxt_f:
            maptxt_f.write(str(f_dimacs.mapping))

        # Write the binary mapping file (numpy format, same as clutrr_to_sat)
        import numpy as np
        mapping_pth = os.path.join(DIMACS_DIR, q + '_' + file[:-3] + '.mapping')
        with open(mapping_pth, 'wb') as mapping_f:
            np.save(mapping_f, f_dimacs.mapping)

    if skip_problem:
        continue

    # Append this example's ground truth to the labels CSV.
    # Format: filename, gt_value  (same as clutrr_new_labels.csv but for ethics)
    with open(LABELS_CSV, 'a') as labels_f:
        labels_f.write(file + ', ' + str(gt) + '\n')

# <<< [ExplainEthics Adaptation] END: Module
# <<< END OF FILE
