"""
explain_ethics_to_sat_2.py
===========================
ExplainEthics-FOL adaptation of pronto_to_sat.py.

Purpose
-------
Reads LLM-generated Python files from SAT-LM/tmp/ named like
explainethics2{idx}.py that contain proofd5-style FOL statements:

    def solution():
        ForAll([x], Implies(covering_up_truth(x), deception(x)))
        ForAll([x], Implies(deception(x), dishonest_behavior(x)))
        covering_up_truth(I)           # ground fact
        return violate_fairness(I)     # query variable

This is the FOL counterpart of explain_ethics_to_sat.py (which handles the
older z3 Implies(A,B) format).  The two pipelines coexist:
  - explainethics{idx}.py   → explain_ethics_to_sat.py   (z3 / implies style)
  - explainethics2{idx}.py  → explain_ethics_to_sat_2.py (FOL / ForAll style)

Key design
----------
- Variable space: unary propositional symbols of the form   pred_I_   or
  pred_entity_   (arity=1), matching the naming convention used in
  cot_met_explain_ethics_2.py.
- ForAll([x], Implies(A(x), B(x))) is grounded over all entities that appear
  in the problem (same entity expansion trick as pronto_to_sat.py).
- Conjunction ForAll([x], Implies(And(A(x), B(x)), C(x))) is also handled.
- Ground-fact lines like  covering_up_truth(I)  add a unit clause.
- The return line  return violate_fairness(I)  determines the query variable.
- Two polarity DIMACS files are produced: pos_ (query True) and neg_ (query negated).
- A .arity file is written alongside so cot_met_explain_ethics_2.py knows
  which predicates are arity-1.
- Mapping, maptxt, and labels CSV follow the same convention as the original.

Usage
-----
    cd SAT-LM/
    python explain_ethics_to_sat_2.py
"""

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import to_cnf, And, Or, Not
from sympy import symbols, Implies
from copy import deepcopy
import re
import os
import json
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# DimacsMapping / DimacsFormula — identical to the pronto_to_sat versions
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
    """Convert a sympy CNF And-expression to a DimacsFormula."""
    if dimacs_mapping is None:
        dimacs_mapping = DimacsMapping()
    dimacs_clauses = []

    assert type(sympy_cnf) == And, f"Expected And, got {type(sympy_cnf)}"

    for sympy_clause in sympy_cnf.args:
        dimacs_clause = []
        if type(sympy_clause) != Or:
            sympy_literal = sympy_clause
            if type(sympy_literal) == Not:
                sympy_symbol, polarity = sympy_literal.args[0], -1
            elif type(sympy_literal) == Symbol:
                sympy_symbol, polarity = sympy_literal, 1
            else:
                raise AssertionError("invalid cnf")
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
                raise AssertionError("invalid cnf")
            dimacs_variable = dimacs_mapping.get_variable_for(sympy_symbol)
            dimacs_literal = dimacs_variable * polarity
            dimacs_clause.append(dimacs_literal)

        dimacs_clauses.append(dimacs_clause)

    return DimacsFormula(dimacs_mapping, dimacs_clauses)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TMP_DIR      = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp'
DIMACS_DIR   = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs'
DATASET_PATH = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/data/explainethics_test.json'
LABELS_CSV   = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/explain_ethics_labels_2.csv'


# ---------------------------------------------------------------------------
# Parsing helpers — handle the proofd5 / FOL output style
# ---------------------------------------------------------------------------

_FORALL_IMPLIES_RE = re.compile(
    r'ForAll\(\[x\],\s*Implies\((.+)\)\s*\)', re.IGNORECASE
)
_FORALL_AND_IMPLIES_RE = re.compile(
    r'ForAll\(\[x\],\s*Implies\(And\((.+?)\),\s*(.+?)\)\s*\)', re.IGNORECASE
)
_GROUND_FACT_RE = re.compile(
    r'^([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)\s*$'
)
_NOT_GROUND_FACT_RE = re.compile(
    r'^Not\(([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)\)\s*$', re.IGNORECASE
)
_RETURN_RE = re.compile(
    r'return\s+([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)'
)


def _pred_to_sym_name(pred: str, entity: str) -> str:
    """Build the canonical symbol name: pred__entity__  (double-underscore convention)."""
    return pred + '__' + entity + '__'


def parse_fol_lines(lines):
    """
    Parse a proofd5-style FOL solution() block.

    Returns
    -------
    entities   : set of entity strings that appear in the problem
    arity      : dict {pred_name: 1}  (all preds are unary in this scheme)
    forall_imp : list of (antecedents: list[str], consequent: str)
                 where each element is a predicate NAME (without entity) —
                 grounding is done later
    ground_facts : list of (pred, entity, positive:bool)
    query        : (pred, entity) or None
    """
    entities = set()
    arity = {}
    forall_imp = []        # (list[antecedent_pred], consequent_pred)
    ground_facts = []      # (pred, entity, positive:bool)
    query = None

    in_solution = False
    for raw in lines:
        line = raw.strip()
        if line.startswith('def solution():'):
            in_solution = True
            continue
        if not in_solution:
            continue

        # ---- return line (query variable) ----
        m = _RETURN_RE.search(line)
        if m:
            pred, entity = m.group(1).lower(), m.group(2)
            query = (pred, entity)
            entities.add(entity)
            arity[pred] = 1
            continue

        # ---- ForAll([x], Implies(And(A(x), B(x)), C(x))) ----
        m2 = _FORALL_AND_IMPLIES_RE.search(line)
        if m2:
            ants_raw = m2.group(1)   # e.g. "touching_child(x), Not(consent(x))"
            cons_raw = m2.group(2).strip()
            # parse antecedents — may include Not(pred(x))
            ants = []
            for part in ants_raw.split(','):
                part = part.strip()
                nm = re.match(r'Not\(([a-zA-Z_][a-zA-Z0-9_]*)\(x\)\)', part, re.IGNORECASE)
                pm = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\(x\)', part)
                if nm:
                    pred_name = nm.group(1).lower()
                    ants.append(('NOT', pred_name))
                    arity[pred_name] = 1
                elif pm:
                    pred_name = pm.group(1).lower()
                    ants.append(('POS', pred_name))
                    arity[pred_name] = 1
            # parse consequent
            nm_c = re.match(r'Not\(([a-zA-Z_][a-zA-Z0-9_]*)\(x\)\)', cons_raw, re.IGNORECASE)
            pm_c = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\(x\)', cons_raw)
            if ants and (nm_c or pm_c):
                if nm_c:
                    cons_pred = ('NOT', nm_c.group(1).lower())
                    arity[nm_c.group(1).lower()] = 1
                else:
                    cons_pred = ('POS', pm_c.group(1).lower())
                    arity[pm_c.group(1).lower()] = 1
                forall_imp.append((ants, cons_pred))
            continue

        # ---- ForAll([x], Implies(A(x), B(x))) ----
        m3 = _FORALL_IMPLIES_RE.search(line)
        if m3:
            inner = m3.group(1)
            # Split on the outermost comma: Implies(A, B) -> inner = "A(x), B(x)"
            # Simple split (works for single-predicate antecedent)
            parts = [p.strip() for p in inner.split(',', 1)]
            if len(parts) == 2:
                def _parse_pred(s):
                    nm = re.match(r'Not\(([a-zA-Z_][a-zA-Z0-9_]*)\(x\)\)', s, re.IGNORECASE)
                    pm = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\(x\)', s)
                    if nm:
                        return ('NOT', nm.group(1).lower())
                    elif pm:
                        return ('POS', pm.group(1).lower())
                    return None

                ant_parsed = _parse_pred(parts[0])
                cons_parsed = _parse_pred(parts[1])
                if ant_parsed and cons_parsed:
                    for pol, pred in [ant_parsed, cons_parsed]:
                        arity[pred] = 1
                    forall_imp.append(([ant_parsed], cons_parsed))
            continue

        # ---- ground fact: pred(entity) ----
        m4 = _GROUND_FACT_RE.match(line)
        if m4 and 'ForAll' not in line and 'Implies' not in line and 'def ' not in line and 'return' not in line:
            pred, entity = m4.group(1).lower(), m4.group(2)
            entities.add(entity)
            arity[pred] = 1
            ground_facts.append((pred, entity, True))
            continue

        # ---- negative ground fact: Not(pred(entity)) ----
        m5 = _NOT_GROUND_FACT_RE.match(line)
        if m5:
            pred, entity = m5.group(1).lower(), m5.group(2)
            entities.add(entity)
            arity[pred] = 1
            ground_facts.append((pred, entity, False))
            continue

    return entities, arity, forall_imp, ground_facts, query


def build_sympy_formula_fol(entities, arity, forall_imp, ground_facts, query, negate_query=False):
    """
    Ground out FOL rules over all entities and build a sympy And formula.

    For each ForAll([x], Implies(A(x), B(x))) we produce one clause per entity e:
        Implies(sym(A,e), sym(B,e))
    Conjuncts are assembled and passed to to_cnf().
    """
    if not entities:
        return None

    # Build a symbol dictionary: (pred, entity) -> sympy Symbol
    sym = {}
    for pred, ar in arity.items():
        for e in entities:
            key = (pred, e)
            sym[key] = symbols(_pred_to_sym_name(pred, e))

    # Ensure query symbol exists
    if query:
        q_pred, q_ent = query
        if (q_pred, q_ent) not in sym:
            sym[(q_pred, q_ent)] = symbols(_pred_to_sym_name(q_pred, q_ent))

    conjuncts = []

    # Ground facts
    for pred, entity, positive in ground_facts:
        if (pred, entity) not in sym:
            sym[(pred, entity)] = symbols(_pred_to_sym_name(pred, entity))
        s = sym[(pred, entity)]
        conjuncts.append(s if positive else Not(s))

    # Ground ForAll rules over each entity
    def _sym_for(polarity, pred, entity):
        key = (pred, entity)
        if key not in sym:
            sym[key] = symbols(_pred_to_sym_name(pred, entity))
        s = sym[key]
        return Not(s) if polarity == 'NOT' else s

    for ants, cons_parsed in forall_imp:
        for e in entities:
            ant_syms = [_sym_for(pol, pred, e) for pol, pred in ants]
            cons_sym = _sym_for(*cons_parsed, e)
            if len(ant_syms) == 1:
                conjuncts.append(Implies(ant_syms[0], cons_sym))
            else:
                conjuncts.append(Implies(And(*ant_syms), cons_sym))

    # Query literal
    if query is None:
        return None
    q_sym = sym.get((query[0], query[1]))
    if q_sym is None:
        return None
    if negate_query:
        conjuncts.append(Not(q_sym))
    else:
        conjuncts.append(Not(Not(q_sym)))

    if not conjuncts:
        return None
    return And(*conjuncts)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(f"[explain_ethics_to_sat_2] Loading dataset from {DATASET_PATH}")
with open(DATASET_PATH, 'r') as df:
    data = json.loads(df.read())

for file in tqdm(os.listdir(TMP_DIR)):
    skip_problem = False

    # Only process FOL-style files named explainethics2{idx}.py
    if not file.startswith('explainethics2') or not file.endswith('.py'):
        continue

    try:
        idx = int(file.replace('explainethics2', '').replace('.py', ''))
    except ValueError:
        print(f"[explain_ethics_to_sat_2] Cannot parse index from: {file}")
        continue

    if idx >= len(data):
        print(f"[explain_ethics_to_sat_2] Index {idx} out of range, skipping")
        continue

    prob = data[idx]
    gt   = prob.get('gt', 'true')
    full_path = os.path.join(TMP_DIR, file)
    lines = open(full_path, 'r').readlines()

    print(f"[explain_ethics_to_sat_2] Processing {file}, gt={gt}")

    entities, arity, forall_imp, ground_facts, query = parse_fol_lines(lines)

    if not forall_imp and not ground_facts:
        print(f"[explain_ethics_to_sat_2] No logic found in {file}, skipping")
        continue

    if query is None:
        # Fallback: use gold_foundation as query with entity 'I'
        query = (prob['gold_foundation'].replace('-', '_'), 'I')
        arity[query[0]] = 1
        entities.add('I')

    print(f"[explain_ethics_to_sat_2] entities={entities}, query={query}")
    print(f"[explain_ethics_to_sat_2] arity={arity}")
    print(f"[explain_ethics_to_sat_2] rules={len(forall_imp)}, facts={len(ground_facts)}")

    # Base filename without extension (same convention as pronto_to_sat)
    basename = file[:-3]   # e.g. explainethics20

    f_dimacs = None
    for q in ['pos', 'neg']:
        negate = (q == 'neg')
        try:
            formula = build_sympy_formula_fol(entities, arity, forall_imp, ground_facts, query, negate_query=negate)
            if formula is None:
                print(f"[explain_ethics_to_sat_2] Empty formula for {file} polarity={q}")
                skip_problem = True
                break
            cnf = to_cnf(formula)
        except Exception as e:
            print(f"[explain_ethics_to_sat_2] CNF error for {file} polarity={q}: {e}")
            skip_problem = True
            break

        try:
            if q == 'neg':
                f_dimacs = to_dimacs_formula(cnf, dimacs_mapping=f_dimacs.mapping)
            else:
                f_dimacs = to_dimacs_formula(cnf)
        except Exception as e:
            print(f"[explain_ethics_to_sat_2] DIMACS error for {file} polarity={q}: {e}")
            skip_problem = True
            break

        # Write DIMACS CNF file
        cnf_pth = os.path.join(DIMACS_DIR, q + '_' + basename + '.cnf')
        print(f"[explain_ethics_to_sat_2] Writing {cnf_pth}")
        with open(cnf_pth, 'w') as fout:
            fout.write(str(f_dimacs))

        # Write human-readable mapping file
        maptxt_pth = os.path.join(DIMACS_DIR, q + '_' + basename + '.maptxt')
        with open(maptxt_pth, 'w') as fout:
            fout.write(str(f_dimacs.mapping))

        # Write binary numpy mapping (for cot_met_explain_ethics_2.py)
        mapping_pth = os.path.join(DIMACS_DIR, q + '_' + basename + '.mapping')
        with open(mapping_pth, 'wb') as fout:
            np.save(fout, f_dimacs.mapping)

    if skip_problem:
        continue

    # Write .arity file (only neg_ needed by inference; write both for safety)
    # Format: numpy dict {pred_name: arity_int}
    for q in ['pos', 'neg']:
        arity_pth = os.path.join(DIMACS_DIR, q + '_' + basename + '.arity')
        with open(arity_pth, 'wb') as fout:
            np.save(fout, arity)

    # Append ground-truth label to the labels CSV
    with open(LABELS_CSV, 'a') as labels_f:
        labels_f.write(basename + '.cnf, ' + str(gt) + '\n')

print("[explain_ethics_to_sat_2] Done.")
