from z3 import *
# Declare one Z3 Bool variable per unique logical predicate
harm_to_human = Bool('harm_to_human')
physical_harm = Bool('physical_harm')
shooting_hole_in_man_ear = Bool('shooting_hole_in_man_ear')
violate_care = Bool('violate_care')

s = Solver()
# Add each implies() line as a Z3 Implies() constraint
s.add(Implies(shooting_hole_in_man_ear, physical_harm))
s.add(Implies(physical_harm, harm_to_human))
# Assert known action facts as True
s.add(shooting_hole_in_man_ear)

# Query: check if violate_care is satisfiable (can be True)
result = s.check(violate_care)
if result == sat:
    print('violate_care')
elif result == unsat:
    print('UNSAT')
else:
    print('UNKNOWN')