

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

# if not os.path.exists()
# os.mkdir('/home/XXXX/LLM-project/dimacs/')
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

for file in os.listdir('/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp/'):
# if 1 > 0:
    if not file.startswith('pronto'):
        continue

    file = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/tmp/' + file
    # file = '/home/XXXX/SAT-LM/tmp/94a88eb2cc77dbb6.py'
    lines = open(file, 'r').readlines()


    vars = []
    funcs_1 = []
    funcs_2 = []
    premises = []
    vars = lines[1].split('ThingsSort, (')[1].split(') =')[0].replace(' ', '').split(',')
    # print(vars)
    # breakpoint()
    vars_done = False
    start_premise=False
    for line in lines[2:]:
        if "Const" in line or "precond = []" in line or "=" not in line: break
        split = line.split('=')
        # funcs.append(line.split(' =')[0])
        name = split[0].strip()
        # print('Function(\'' + name  + '\', ThingsSort, ThingsSort, BoolSort())' )
        if  'Function(\'' + name  + '\', ThingsSort, ThingsSort, BoolSort())' in split[1]:
            funcs_2.append(split[0].strip())
        else:
            funcs_1.append(split[0].strip())

    # print(lines)

    lits = {}
    conts = 0
    for i in range(len(vars)):
        for j in range(len(funcs_1)):
            var = vars[i]
            func = funcs_1[j]
            lits[(var, func)] = i*(len(funcs_1)) + j
    for i in range(len(vars)):
        for j in range(len(vars)):
            var1 = vars[i]
            var2 = vars[j]
            # if var1 == var2:
            #     conts += len(funcs_2)
            #     continue
            for k in range(len(funcs_2)):
                func=funcs_2[k]
                lits[(var1, var2, func)] = (len(vars))*len(funcs_1) + len(vars)*len(funcs_2)*i + len(funcs_2)*j+k - conts
    


    sym = []
    # print(lits)
    for i in range(len(lits.keys())):
        sym.append('')
    for key, value in lits.items():
        # print(key)
        if len(key) == 3:
            var1 ,var2, func = key
            # print(key)
            # print(func + '_' + var1 + '_' + var2+'_' +'= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var1 + '_'  + var2 + '\''+ "+'_')")
            exec(func + '_' + var1 + '_' + var2+'_' +'= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var1 + '_'  + var2 + '\''+ "+'_')")
        else: 
            var, func = key
            exec(func + '_' + var + '_' + '= symbols(' + '\'' + func+ '\'' + "+'_'+" + '\'' + var + '\'' + "+'_')")
      
    # need_bear_cow = 0
    # breakpoint()

    flag=False
    us_lines = []
    # print(vars)
    # print(funcs)
    solver=False
    n = 0
    question = ''
    # breakpoint()
    for line in lines:
        if line.startswith('s = Solver()'):
            solver=True
            flag=False
        if solver:
            n += 1
            if n == 3:
                flag=True
                question = line.replace('s.add(', '')[:-2]
                # break
        tmp = line
        # breakpoint()
        if n == 3:
            tmp = question
        if flag and n != 3:
            for i in range(len(vars)):
                for j in range(len(vars)):
                    var1 = vars[i]
                    var2 = vars[j]
                    # if var1 == var2:
                    #     conts += len(funcs_2)
                    #     continue
                    for k in range(len(funcs_2)):
                        func = funcs_2[k]
                        tmp = tmp.replace(func + '(' + var1 + ', '+var2 + ')', func + '_' + var1 + '_'+ var2+ '_')
                        # print(var1, var2, )
                        # breakpoint()
                        tmp = tmp.replace(func + '(' + 'x, ' + var1 + ')', func + '_' + 'x' + '_' + var1 + '_')
                        tmp = tmp.replace(func + '(' + var1+ ' x' + ')', func + '_' + var1+ '_' 'x' + '_')

            for func in funcs_1:
                for var in vars:
                    # breakpoint()
                    tmp = tmp.replace(func + '(' + var + ')', func + '_' + var + '_')
                    # print(func, var, tmp)
                    # us_lines.append(tmp)
                tmp = tmp.replace(func + '(' + 'x' + ')', func + '_' + 'x' + '_')
                # breakpoint()
                # print(func, 'x', tmp)
                # us_lines.append(tmp)
            us_lines.append(tmp)
        elif flag and n == 3:
            # print(question)
            for i in range(len(vars)):
                for j in range(len(vars)):
                    var1 = vars[i]
                    var2 = vars[j]
                    # if var1 == var2:
                    #     conts += len(funcs_2)
                    #     continue
                    for k in range(len(funcs_2)):
                        func = funcs_2[k]
                        tmp = tmp.replace(func + '(' + var1 + ', '+var2 + ')', func + '_' + var1 + '_' + var2 + '_')
                        # print(var1, var2, func)
            for func in funcs_1:
                for var in vars:
                    tmp = tmp.replace(func + '(' + var + ')', func + '_' + var + '_')
                    # print(func, var, tmp)
                    # us_lines.append(tmp)
                tmp = tmp.replace(func + '(' + 'x' + ')', func + '_' + 'x' + '_')
            question=tmp
            # print(tmp)
        if n == 3:
            break
        if line.startswith('precond = []'):
            flag=True
    # print('question: ', question)
    flag=False
    # breakpoint()
    sym_lines = []
    for line in us_lines:
        tmp = line
        if 'ForAll(' not in line:
            flag=True
            
        tmp = tmp.replace('precond.append(', '').replace('ForAll([x],', '')
        tmp = tmp.strip('\n')
        stack = 0
        for i in range(len(tmp)):
            char = tmp[i]
            if char == '(':
                stack += 1
            if char == ')':
                stack -= 1
        # if flag==True:
        #     tmp = 'Or(' + tmp[:-1] + ', False)\n'
        #     flag=False
        if stack < 0:
            tmp = tmp[:stack]
        else:
            tmp = tmp.strip('\n')
        sym_lines.append(tmp)
        # print(tmp)
        # print(line)
        # print('tmp:', tmp)
    final = []
    # breakpoint()
    # breakpoint()
    for line in sym_lines:
        # print(line)
        if '_x_' in line:
            for var in vars:
                final.append(line.replace('x', var))
        else:
            final.append(line)
    # breakpoint()
    c = 0
    for line in final:
        if 'Implies' not in line:
            c += 1
        if 'Implies' in line:
            break
    new_final = final[c:] + final[:c]
    final = new_final
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
        for line in final:
            formula += line + ','
            # print(line)
        # print(new_question)
        # formula = formula[:-1]
        formula += new_question
        formula += ')'

        # print(formula)
        # print(to_cnf(formula))
            # print(line)
        # try:
        # print(formula)
        # print(len(lits))
        # to_cnf
        # exec('f = to_cnf(' + formula + ')')
        
        # breakpoint()
        try:
            exec('f = to_cnf(' + formula + ')')
        except:
            print(func, file)
            # print(formula)
            continue
        try:
            if q == 'neg':
                f_dimacs = to_dimacs_formula(f, dimacs_mapping=f_dimacs.mapping)
            else:
                f_dimacs = to_dimacs_formula(f)
            # flat=1
        except:
            print('to_dimacs error', formula)
            continue
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

    # except:
    #     print(file)
# dim
# apply = open('/home/XXXX/apply.txt', 'r')
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
# # ae =  open('/home/XXXX/apply_e.txt', 'w')
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
