

class DimacsMapping:
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


from sympy.core.symbol import Symbol
from sympy.logic.boolalg import to_cnf, And, Or, Not
from copy import deepcopy
import re as regex

# import re
# import re
# import re

def translate_expression(logical_expr):
    # Define the regex patterns for matching R(x, y) == <relation>
    pattern_r = regex.compile(r'R\((\w), (\w)\) == (\w+)')
    
    # Replace the "R(x, y) == <relation>" with "<relation>_x_y"
    def replace_r(match):
        x, y, relation = match.groups()
        return f"{relation}_{x}_{y}_"
    
    # Perform the replacement
    translated_expr = regex.sub(pattern_r, replace_r, logical_expr)
    
    # Return the translated expression
    return translated_expr

# Test the function with the given example
# logical_expr = "Implies(And(R(x, y) == sister_in_law, R(y, z) == brother), Or(R(x, z) == husband, R(x, z) == brother_in_law))"
# translated_expr = translate_expression(logical_expr)

# print(translated_expr)






# Example usage



# if not os.path.exists()
# os.mkdir('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs')
def to_dimacs_formula(sympy_cnf, dimacs_mapping=None):
    if dimacs_mapping == None:
        dimacs_mapping = DimacsMapping()
    dimacs_clauses = []
    # try:
    assert type(sympy_cnf) == And
    # except:
        # print(type(sympy_cnf))
        # breakpoint()
    for sympy_clause in sympy_cnf.args:
        dimacs_clause = []
        try:
            assert type(sympy_clause) == Or
        except:
            # sympy_clause=Or(sympy_clause,False)
            # print(sympy_clause)
            # print(sympy_clause)
            # breakpoint()
            sympy_literal=sympy_clause
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
        # breakpoint()

    return DimacsFormula(dimacs_mapping, dimacs_clauses)

from sympy import *
import os
import json
import random
from tqdm import tqdm
for file in tqdm(os.listdir('/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp')):
    skip_problem=False

# if 1 > 0:
    js = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/data/clutrr_test.json'
    jstr = open(js, 'r').read()
    data = json.loads(jstr)

    fn = int(file.split('clutrr')[1].split('.py')[0])

    query = data[fn]['query'].replace('(', '').replace(')', '').replace('\'', '').split(',')
    label = data[fn]['label']
    gt = data[fn]['gt']
    question = label.replace('-', '_') + '_' + query[0].strip(' ') + '_' + query[1].strip(' ') + '_'

    file = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp/' + file
    # file = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp/94a88eb2cc77dbb6.py'
    lines = open(file, 'r').readlines()
    rels = []
    for line in lines:
        if line.startswith('relation_names'):
            rnames = line.split('[')[1].split(']')[0]
            for r in rnames.split(', '):
                rels.append(r.strip('\''))
    
    vars = lines[1].split('PeopleSort, (')[1].split(') =')[0].replace(' ', '').split(',')

    # breakpoint()
    lits = {}
    conts = 0
    
    for i in range(len(vars)):
        for j in range(len(vars)):
            var1 = vars[i]
            var2 = vars[j]
            if var1 == var2:
                conts += 1
                continue
            for k in range(len(rels)):
                func=rels[k]
                lits[(var1, var2, func)] = len(vars)*len(rels)*i + len(rels)*j+k - conts
    
    # cer = []
    # cerlines = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/cer_lines.py', 'r').readlines()
    # for line in cerlines:
    #     if 'cer_precond.append(ForAll([x, y], (x == y) == (R(x, y) == self)))' in line: continue
    #     if 'y == z' in line: continue
    #     if line.startswith('cer'):
    #         if 'ForAll([x, y]' in line:
    #             splt = line.split('cer_precond.append(ForAll([x, y],')[1][:-3].split('== Or')
    #             # breakpoint()
    #             cer.append('')
    #             cer[-1] = 'cer_precond.append(ForAll([x, y],' +  ' Implies(' + splt[0].strip(' ') + ', Or' + splt[1].strip(' ') + ')))'
    #             cer.append('')
    #             cer[-1] = 'cer_precond.append(ForAll([x, y],' +  ' Implies(Or' + splt[1].strip(' ') + ', ' + splt[0].strip(' ') + ')))'
    #             # breakpoint()
    #             continue
    #         cer.append(line)
    # cer = cer[2:]
    # cer_g = []
    # for v1 in vars:
    #     for v2 in vars:
    #         if v1 == v2: continue
    #         flag=False
    #         for v3 in vars:
    #             if v1 == v3 or v2 == v3: continue
    #             for c in cer:
    #                 if 'cer_precond.append(ForAll([x, y, z], ' in c: 
    #                     tmp = translate_expression(c.split('cer_precond.append(ForAll([x, y, z], ')[1])
    #                     tmp = tmp.replace('x != z, ', '')
    #                     # tmp = tmp.replace
    #                     tmp = tmp.strip('\n')
    #                     tmp = tmp[:-2]
    #                     tmp = tmp.replace('_x_', '_' + v1 + '_')
    #                     tmp = tmp.replace('_y_', '_' + v2 + '_')
    #                     tmp = tmp.replace('_z_','_' +  v3 + '_')
    #                 elif 'cer_precond.append(ForAll([x, y],' in c and not flag:
    #                     tmp = translate_expression(c.split('cer_precond.append(ForAll([x, y], ')[1])
    #                     # tmp.replace('y == z')
    #                     tmp.strip('\n')
    #                     # tmp = tmp[:-1]
    #                     tmp = tmp.strip(')').strip(')').strip(')') + '))'
    #                     tmp = tmp.replace('_x_', '_' + v1 + '_')
    #                     tmp = tmp.replace('_y_', '_' + v2 + '_')

    #                 # else: breakpoint()
    #                 cer_g.append(tmp)
    #                 # breakpoint()
    #             flag=True
    # breakpoint()
    sym = []
    # print(lits)
    for i in range(len(lits.keys())):
        sym.append('')
    for key, value in lits.items():
        # print(key)
        # if skip_problem: break
        if len(key) == 3:
            var1 ,var2, func = key
            # print(key)
            # print(func + '_' + var1 + '_' + var2+'_' +'= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var1 + '_'  + var2 + '\''+ "+'_')")
            try:exec(func + '_' + var1 + '_' + var2+'_' +'= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var1 + '_'  + var2 + '\''+ "+'_')")
            except: 
                breakpoint()
                skip_problem=True
                break
        else: 
            var, func = key
            exec(func + '_' + var + '_' + '= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var + '\'' + "+'_')")
    if skip_problem:
        continue
    # breakpoint()
    conds = []
    for line in lines:
        if line.startswith('decl_conditions.append('):
            tmp = line
            n1 = line.split('R(')[1].split(', ')[0]
            n2 = line.split('R(')[1].split(', ')[1].split(')')[0]
            rel = line.split('R(' + n1 + ', ' + n2 + ') == ')[1].split(')\n')[0]

            conds.append(rel + '_' + n1 + '_' + n2 + '_')
    # conds += cer_g
    if len(conds) == 0:
        print(file)
        continue
    # breakpoint()
    # flipped = False
    # r = random.random()
    # if r > 0.5:
    #     question = 'Not(' + question + ')'
    #     flipped=True
        
    # breakpoint()
    for q in ['pos', 'neg']:
        # print(q)
        # new_question = 'Or(' + question + ', False)'
        new_question=deepcopy(question)
        if q == 'neg':
            # new_question = 'Or(Not(' +question + ')' ', False)'
            new_question = 'Not(' + question + ')'
        elif q == 'pos':
            new_question = 'Not(Not(' + question + '))'
        formula = 'And('
        for line in conds:
            formula += line + ','
        # for lit in lits: ### instantiating all possible fariables
        #     formula += 'Or(' + lit[2] + '_' + lit[1] + '_' + lit[0] + '_,Not(' +lit[2] + '_' + lit[1] + '_' + lit[0] + '_)),'
            # print(line)
        # print(new_question)
        # formula = formula[:-1]

        formula += new_question
        formula += ')'
    
        
        try:
            exec('f = to_cnf(' + formula + ')')
        except:
            print(file)
            # breakpoint()
            # print(formula)
            continue
        # try:
        if q == 'neg':
            f_dimacs = to_dimacs_formula(f, dimacs_mapping=f_dimacs.mapping)
        else:
            # breakpoint()
            try:
                f_dimacs = to_dimacs_formula(f)
            except:
                1 == 0
                # breakpoint()
        # flat=1
        # except:
        #     print('to_dimacs error', formula)
        #     continue
        # except:
        #     continue
        # print(f_dimacs)
        # print()
        # print(f_dimacs.mapping)
        # for clause in f_dimacs:
        #     print(clause)
        # print(f_dimacs)
        dimacs = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/' + q + '_' + file.split('/')[-1][:-3] + '.cnf', 'w')
        dimacs.write(str(f_dimacs))
        dimacs.close()
        maptxt = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/' + q + '_' + file.split('/')[-1][:-3] + '.maptxt', 'w')
        import numpy as np
        # np.save(mapping, 
        # acs.mapping)
        maptxt.write(str(f_dimacs.mapping))
        maptxt.close() 
        # breakpoint()

        mapping = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/' + q + '_' + file.split('/')[-1][:-3] + '.mapping', 'wb')
        np.save(mapping, f_dimacs.mapping)
        mapping.close()
        labels = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/clutrr_new_labels.csv', 'a')
        labels.write(file.split('/')[-1] + ', ' + str(gt)  + '\n')
        labels.close()
        

    # except:
    #     print(file)
# dim
# apply = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/apply.txt', 'r')
# al = apply.readlines()
# wl = []
# stack = 0
# for line in al:
#     for i in range(len(line)):
#         char = line[i]
#         if char == '(':
#             stack += 1
#         if char == ')':
#             stack -= 1
#     if char == '\n' and stack != 0:
#         wl.append(line.replace('\n', ''))
#     else:
#             wl.append(line)
#     # if line.endswith('(x)),\n') or line.endswith('(x),\n') or line.endswith('x,\n'):
#     #     wl.append(line.replace('\n', ''))
#     # else:
#     #     wl.append(line)
# # ae =  open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/apply_e.txt', 'w')
# # for line in wl:
# #    ae.write(line)
# or_open = False
# is_open = False
# not_open = False
# premise = ''

# stack=0
# line =   '  ForAll(x,        Or(Not(Or(Not(red(x)), Not(rough(x)))),            Not(green(x)),            Not(naive(x)))), '
# line = " ".join(line.split())
# line = line.split('ForAll(x, ')[1]
# # breakpoint()
# if 1 > 0:
#     print('in')
# # breakpoint()
# def break_args(string, args, rec=False, s=1):
#     prev_c = 0
#     # args = []
#     buffer = ''
#     line=string
#     # line = '('.join(string.split('(')[1:])
#     # breakpoint()
#     stack = s
#     alen = len(args)
#     # print(line)
#     for i in range(len(line)):
#         # breakpoint()
#         # print(stack)
#         char = line[i]
#         if char == '(':
#             stack += 1
#             # breakpoint()
#         elif char == ')':
#             stack -= 1
#             # breakpoint()
#         try:
#             buffer += char
#         except:
#             print(buffer,)
#         if stack <= 0 and char == ',':
#             args.append(buffer)
#             buffer = ''
#         if char == '\n':
#             bstack = 0
#             # print('in slash-n:', buffer)
#             new_buffer = ''
#             n_p = 0
#             flag = False
#             for j in range(len(buffer)):
#                 bchar = buffer[j]
#                 new_buffer += bchar
#                 # print('bstack:', bstack)
#                 if bchar == '(':
#                     bstack += 1
#                     flag = True
#                 if bchar == ')':
#                     bstack -= 1
#                 if flag:
#                     if bstack == 0:
#                         args.append(new_buffer)
#                         buffer = ''
#                         break
#     if rec:
#         if alen == len(args):
#             return args
#         else:
#             a = []
#             for arg in args[alen:]:
#                 a += break_args(arg, args)
#             # print(a)
#             return a
#     else:
#         # print(args)
#         # breakpoint()
#         return args
    
# # line = 'precond.append(ForAll([x], Implies(And(green(x), naive(x)), And(red(x), rough(x)))))\n'
# line = 'precond.append(cold(bob))\n'
# # line = 'precond.append(ForAll([x], Implies(And(big(x), kind(x), cold(x)), red(x))))\n'
# a = (break_args(line, [], s=-2))
# d = []
# for i in range(len(a)):
#     arg = a[i]
#     if arg == 'ForAll([x],':
#         d.append(i)
# for z in d[::-1]:
#     del a[z]
# # print(a)
# # print(a[2])
# # if 'And(' in a[2]:
# #     m =  break_args(a[2] + ',', [], s=-1)
# # print('m:', m)


# for line in us_lines:
#     print(line)
#     # print(func + '(' + var + ')', func + '_' + var + '_')
#     print(to_cnf(Implies(And(red_alan_, rough_alan_), green_alan_)))

# breakpoint()
# g, n, rd, rh, = symbols('g,n,rd,rg')
# asdf = Implies(And(g, n), And(rd, rh))
# print(Implies(And(g, n), And(rd, rh)))
# # breakpoint()


                






# from z3 import *

# ThingsSort, (Alan, Charlie, Dave) = EnumSort('ThingsSort', ['Alan', 'Charlie', 'Dave'])

# x = Const('x', ThingsSort)

# people = [Alan, Charlie, Dave]
# kind = Function('kind', ThingsSort, BoolSort())
# young = Function('young', ThingsSort, BoolSort())
# cold = Function('cold', ThingsSort, BoolSort())
# red = Function('red', ThingsSort, BoolSort())
# rough = Function('rough', ThingsSort, BoolSort())
# big = Function('big', ThingsSort, BoolSort())
# green = Function('green', ThingsSort, BoolSort())
# nice = Function('nice', ThingsSort, BoolSort())
# round = Function('round', ThingsSort, BoolSort())
# blue = Function('blue', ThingsSort, BoolSort())
# precond = []
# precond.append(ForAll([x], Implies(And(young(x), cold(x), blue(x)), kind(x))))
# precond.append(ForAll([x], Implies(young(x), red(x))))
# precond.append(ForAll([x], Implies(And(kind(x), round(x), rough(x)), young(x))))
# precond.append(ForAll([x], Implies(And(nice(x), rough(x)), big(x))))
# precond.append(ForAll([x], Implies(And(green(x), round(x)), kind(x))))
# precond.append(ForAll([x], Implies(And(rough(x), red(x)), cold(x))))

# for cond in precond:
#     breakpoint()








# for line in al:
#     if 'Or' in line:
#         word = ''
#         for char in line.split('Or')[1]:
#             if char == '(':
#                 stacks['or'] += 1
#                 if word == 'Not':
#                     not_open=True
#             if char == ')':
#                 stacks['or'] -= 1
#             word += char
