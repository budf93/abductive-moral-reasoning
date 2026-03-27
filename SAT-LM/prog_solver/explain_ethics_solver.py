"""
explain_ethics_solver.py
========================
ExplainEthics adaptation of the prog_solver module.

Purpose
-------
This file replaces clutrr_solver.py for the ExplainEthics task. Instead of
kinship-relation variables like `R(PersonA, PersonB)`, the LLM generates
propositional implication chains such as:

    implies(covering_up_truth, deception) = True
    implies(deception, harm_reputation) = True
    implies(harm_reputation, violate_fairness) = True
    return violate_fairness(I)

This module translates that Python-style pseudo-code into a real Z3 SAT problem,
executes it, and returns whether the indicated moral norm is violated.

Key differences from clutrr_solver.py
--------------------------------------
- No people/relations enum needed; variables are propositional (Bool in Z3)
- Implication rules become Z3 Implies() constraints
- The query is a single Bool variable (the norm label), not R(A, B)
- No transitive closure rules needed; just forward implication chains

# >>> [ExplainEthics Adaptation] BEGIN: Module
"""

import sys
sys.path.append('.')

import os

# Reuse the existing Z3 execution helper from the shared utility module
from prog_solver.z3_utils import execute_z3_test

# ---------------------------------------------------------------------------
# The six canonical moral norm labels in the ExplainEthics dataset.
# These are the possible values of the 'label' and 'gold_foundation' fields.
# We need this list to identify which variable is the query target when
# parsing the LLM-generated code.
# ---------------------------------------------------------------------------
MORAL_NORMS = [
    "violate_care",
    "violate_fairness",
    "violate_loyalty",
    "violate_authority",
    "violate_sanctity",
    "violate_liberty",
]


def parse_explain_ethics_sat_problem(code, return_code=False):
    """
    Parse LLM-generated Python pseudo-code into a list of Z3 source lines.

    The expected input format (from the satlm/satcotsolver prompt) looks like:

        action(I, covering_up_truth) = True
        implies(covering_up_truth, deception) = True
        implies(deception, harm_reputation) = True
        implies(harm_reputation, violate_fairness) = True
        return violate_fairness(I)

    We convert this to a runnable Z3 Python script that:
      1. Declares one Z3 Bool variable per unique predicate term.
      2. Adds each implies() as a Z3 Implies() constraint.
      3. Asks the SAT solver whether the 'return' variable can be True.
      4. Prints the result so execute_z3_test() can capture it.

    Parameters
    ----------
    code : str
        The stripped body of the LLM's def solution(): block.
    return_code : bool
        If True, return the Z3 source as a string instead of running it.

    Returns
    -------
    list[str]  (or str if return_code=True)
        Lines of a runnable Z3 Python script.
    """
    # Split code into individual lines; strip whitespace
    lines = [l.strip() for l in code.splitlines()]
    # Discard comment lines (LLM often includes reasoning comments)
    lines = [l for l in lines if l and not l.startswith('#')]

    # Identify the return line — it tells us which norm variable to query
    # Expected format: "return violate_care(I)" or "answer = violate_care(I)"
    return_line = None
    for l in lines:
        if l.startswith('return ') or (l.startswith('answer') and any(n in l for n in MORAL_NORMS)):
            return_line = l
            break

    # Fall back: scan for any line containing a known norm name with an assignment
    if return_line is None:
        for l in reversed(lines):
            if any(norm in l for norm in MORAL_NORMS):
                return_line = l
                break

    # Determine the target norm from the return/answer line
    target_norm = None
    if return_line:
        for norm in MORAL_NORMS:
            if norm in return_line:
                target_norm = norm
                break

    if target_norm is None:
        # Cannot determine target; return a no-op that prints UNKNOWN
        return ["from z3 import *", "print('UNKNOWN')"]

    # ---------------------------------------------------------------------------
    # Collect all unique predicate terms from implies() and action() lines.
    # Each unique term becomes a Z3 Bool variable.
    # ---------------------------------------------------------------------------
    all_terms = set()
    implication_pairs = []  # list of (antecedent_term, consequent_term)

    for l in lines:
        # Normalize to lowercase for matching — the LLM may emit 'Implies(A,B)' or 'implies(A,B)'
        l_lower = l.lower()
        if l_lower.startswith('implies('):
            # Extract: Implies(antecedent, consequent) = True  (case-insensitive)
            try:
                inner = l_lower.split('implies(')[1].split(')')[0]
                parts = [p.strip() for p in inner.split(',')]
                if len(parts) == 2:
                    ant, cons = parts[0], parts[1]
                    # Sanitize: replace hyphens with underscores in variable names
                    # (same fix applied in clutrr_to_sat.py to prevent Python exec crashes)
                    ant = ant.replace('-', '_')
                    cons = cons.replace('-', '_')
                    all_terms.add(ant)
                    all_terms.add(cons)
                    implication_pairs.append((ant, cons))
            except Exception:
                continue
        elif l_lower.startswith('action('):
            # Extract: Action(agent, predicate) = True  (case-insensitive)
            # The predicate term becomes a known True fact
            try:
                inner = l_lower.split('action(')[1].split(')')[0]
                parts = [p.strip() for p in inner.split(',')]
                if len(parts) == 2:
                    # Only the predicate (second arg) is a logical term
                    pred = parts[1].replace('-', '_')
                    all_terms.add(pred)
            except Exception:
                continue

    # Ensure the target norm is always in the term set even if LLM omitted it
    all_terms.add(target_norm)

    # ---------------------------------------------------------------------------
    # Build the Z3 source lines
    # ---------------------------------------------------------------------------
    z3_lines = []

    # Import the Z3 library
    z3_lines.append("from z3 import *")

    # Declare one Bool variable per unique predicate term encountered in the code.
    # Using safe Python identifier names (terms already sanitized above).
    z3_lines.append("# Declare one Z3 Bool variable per unique logical predicate")
    for term in sorted(all_terms):
        # Each term becomes a SAT variable: e.g. deception = Bool('deception')
        z3_lines.append(f"{term} = Bool('{term}')")

    # Create the Z3 Solver instance
    z3_lines.append("")
    z3_lines.append("s = Solver()")
    z3_lines.append("# Add each implies() line as a Z3 Implies() constraint")
    for ant, cons in implication_pairs:
        # Z3 Implies(p, q) means: if p is True then q must also be True
        z3_lines.append(f"s.add(Implies({ant}, {cons}))")

    # Add the action() terms as hard True facts (they are known to be true from context)
    z3_lines.append("# Assert known action facts as True")
    for l in lines:
        # Again normalize to lowercase so Action(I, x) is handled the same as action(I, x)
        l_lower = l.lower()
        if l_lower.startswith('action('):
            try:
                inner = l_lower.split('action(')[1].split(')')[0]
                parts = [p.strip() for p in inner.split(',')]
                if len(parts) == 2:
                    pred = parts[1].replace('-', '_')
                    z3_lines.append(f"s.add({pred})")  # assert the action as a hard fact
            except Exception:
                continue

    # Query: check whether the target norm variable can be True given all implications
    z3_lines.append("")
    z3_lines.append(f"# Query: check if {target_norm} is satisfiable (can be True)")
    z3_lines.append(f"result = s.check({target_norm})")
    z3_lines.append("if result == sat:")
    # If SAT, print the norm label so the calling code knows which norm was detected
    z3_lines.append(f"    print('{target_norm}')")
    z3_lines.append("elif result == unsat:")
    z3_lines.append("    print('UNSAT')")
    z3_lines.append("else:")
    z3_lines.append("    print('UNKNOWN')")

    return z3_lines


def explain_ethics_satlm_exec(code, prompting_style="satlm", filename=None):
    """
    Execute the LLM-generated ethics reasoning code via Z3.

    Analogous to `clutrr_satlm_exec()` in clutrr_solver.py.

    Parameters
    ----------
    code : str
        The body of the LLM's def solution(): block (stripped of the def line).
    prompting_style : str
        Prompting style string (kept for API compatibility; only 'satlm' supported).
    filename : str or None
        Optional filename hint for the temp file written by execute_z3_test.

    Returns
    -------
    (status: bool, answer: str)
        status — True if Z3 ran successfully, False on error/timeout
        answer — one of the MORAL_NORMS strings, 'UNSAT', 'UNKNOWN', or an error message
    """
    assert prompting_style in ("satlm", "satcotsolver", "satnosolver"), \
        f"Unsupported prompting style: {prompting_style}"

    # Parse the pseudo-code into Z3 source lines
    z3_lines = parse_explain_ethics_sat_problem(code)
    z3_source = "\n".join(z3_lines)
    print(f"[explain_ethics_solver] Z3 source:\n{z3_source}")

    # Execute via the shared z3_utils helper; it writes to a temp file and runs Python
    result_filename, result = execute_z3_test(
        z3_source,
        timeout=10.0,       # 10s is generous for these small propositional problems
        filename=filename,
        flag_keepfile=True  # keep the file for debugging
    )
    print(f"[explain_ethics_solver] result: {result}")

    status, answer = result
    # Strip trailing whitespace from Z3's printed output
    answer = answer.strip() if isinstance(answer, str) else answer
    return status, answer


# ---------------------------------------------------------------------------
# Simple smoke-test: run this file directly to verify the solver works.
# >>> [ExplainEthics Adaptation] BEGIN: __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_code = """
implies(covering_up_truth, deception) = True
implies(spreading_fake_news, deception) = True
implies(deception, harm_reputation) = True
implies(harm_reputation, violate_fairness) = True
action(I, covering_up_truth) = True
return violate_fairness(I)
""".strip()
    os.makedirs("tmp", exist_ok=True)  # ensure tmp/ dir exists for z3_utils
    status, answer = explain_ethics_satlm_exec(test_code, prompting_style="satlm")
    print(f"Status: {status}, Answer: {answer}")
    assert answer == "violate_fairness", f"Expected 'violate_fairness', got '{answer}'"
    print("Smoke test PASSED")
# <<< [ExplainEthics Adaptation] END: __main__ smoke test

# <<< END OF FILE
