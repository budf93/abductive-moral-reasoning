from z3 import *
ThingsSort, (_3) = EnumSort('ThingsSort', ['_3'])
prime = Function('prime', ThingsSort, BoolSort())
natural = Function('natural', ThingsSort, BoolSort())
real = Function('real', ThingsSort, BoolSort())
Mersenne = Function('Mersenne', ThingsSort, BoolSort())
integer = Function('integer', ThingsSort, BoolSort())
y, x, z = Consts('y x z', ThingsSort)
precond = []
precond.append(ForAll([x], Implies(prime(x), natural(x))))
precond.append(ForAll([x], Implies(natural(x), integer(x))))
precond.append(ForAll([x], Implies(integer(x), real(x))))
precond.append(ForAll([x], Implies(real(x), real(x))))
precond.append(Mersenne(_3))
s = Solver()
s.add(precond)
s.add(Not(real(_3)))
if s.check() == unsat:
    print('True')
else:
    print('False')