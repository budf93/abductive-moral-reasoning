from z3 import *
PeopleSort, (Clarence, Emily, Michael) = EnumSort('PeopleSort', ['Clarence', 'Emily', 'Michael'])
q0 = Michael
q1 = Clarence
RelationSort, (self, cousin, brother_in_law, sister_in_law, mother_in_law, daughter_in_law, father_in_law, son_in_law, grandson, niece, grandmother, wife, grandfather, uncle, mother, nephew, brother, granddaughter, husband, father, aunt, son, sister, daughter) = EnumSort('RelationSort', ['self', 'cousin', 'brother_in_law', 'sister_in_law', 'mother_in_law', 'daughter_in_law', 'father_in_law', 'son_in_law', 'grandson', 'niece', 'grandmother', 'wife', 'grandfather', 'uncle', 'mother', 'nephew', 'brother', 'granddaughter', 'husband', 'father', 'aunt', 'son', 'sister', 'daughter'])
relation_names = ['self', 'cousin', 'brother_in_law', 'sister_in_law', 'mother_in_law', 'daughter_in_law', 'father_in_law', 'son_in_law', 'grandson', 'niece', 'grandmother', 'wife', 'grandfather', 'uncle', 'mother', 'nephew', 'brother', 'granddaughter', 'husband', 'father', 'aunt', 'son', 'sister', 'daughter']
relations = [self, cousin, brother_in_law, sister_in_law, mother_in_law, daughter_in_law, father_in_law, son_in_law, grandson, niece, grandmother, wife, grandfather, uncle, mother, nephew, brother, granddaughter, husband, father, aunt, son, sister, daughter]
R = Function('R', PeopleSort, PeopleSort, RelationSort)
cer_precond = []
x, y, z = Consts('x y z', PeopleSort)
decl_conditions = []
decl_conditions.append(R(Clarence, Emily) == granddaughter)
decl_conditions.append(R(Emily, Clarence) == grandfather)
decl_conditions.append(R(Emily, Michael) == brother)
decl_conditions.append(R(Michael, Emily) == sister)
s = Solver()
s.add(cer_precond)
s.add(decl_conditions)
s.check()
answers = []
while s.check() == sat:
    a = s.model().eval(R(q1, q0))
    answers.append(a)
    s.add(Not(R(q1, q0) == a))
if answers:
    print(answers)
else:
    print('UNSAT')