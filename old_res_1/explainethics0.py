from z3 import *
# Declare one Z3 Bool variable per unique logical predicate
covering_up_truth = Bool('covering_up_truth')
deception = Bool('deception')
harm_reputation = Bool('harm_reputation')
spreading_fake_news = Bool('spreading_fake_news')
violate_fairness = Bool('violate_fairness')

s = Solver()
# Add each implies() line as a Z3 Implies() constraint
s.add(Implies(covering_up_truth, deception))
s.add(Implies(spreading_fake_news, deception))
s.add(Implies(deception, harm_reputation))
s.add(Implies(harm_reputation, violate_fairness))
# Assert known action facts as True
s.add(covering_up_truth)
s.add(spreading_fake_news)

# Query: check if violate_fairness is satisfiable (can be True)
result = s.check(violate_fairness)
if result == sat:
    print('violate_fairness')
elif result == unsat:
    print('UNSAT')
else:
    print('UNKNOWN')