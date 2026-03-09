

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
# os.mkdir('/home/XXXX/XXXX/LLM-project/dimacs_logic/')
def to_dimacs_formula(sympy_cnf, dimacs_mapping=None):
    if dimacs_mapping == None:
        dimacs_mapping = DimacsMapping()
    dimacs_clauses = []
    # try:
    assert type(sympy_cnf) == And
    # except:
        # print(type(sympy_cnf))
        #  
    for sympy_clause in sympy_cnf.args:
        dimacs_clause = []
        try:
            assert type(sympy_clause) == Or
        except:
            # sympy_clause=Or(sympy_clause,False)
            # print(sympy_clause)
            # print(sympy_clause)
            #  
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
        #  

    return DimacsFormula(dimacs_mapping, dimacs_clauses)

from sympy import *
import os
import subprocess
from subprocess import check_output
from pathlib import Path

# Get the directory where the script is located
base_path = Path(__file__).parent
fun3_files = []
errored = 0
bad_qs = 0

print(f"base path: {base_path}")

breakflag=False
# abs path
dimacs_dir = 'C:/Tugas Akhir/ARGOS_public_anon/main/dimacs_folio/'
tmp_dir = 'C:/Tugas Akhir/ARGOS_public_anon/SAT-LM/tmp'

# Define your relative paths
# dimacs_dir = base_path / "dimacs_[dataset]"
# tmp_dir = base_path / "tmp"

# print(len(os.listdir('/home/XXXX/XXXX/LLM-project/tmp_logic/')))
print(len(os.listdir(tmp_dir)))
for file in os.listdir(tmp_dir):
# if 1 > 0:

    file = tmp_dir + file
    try:
        output = check_output(["python", file], stderr=subprocess.STDOUT)
    except:
        # print('skipping ', file)
        errored += 1
        # breakpoint
        continue

    # file = '/home/XXXX/XXXX/SAT-LM/tmp/94a88eb2cc77dbb6.py'
    lines = open(file, 'r').readlines()
    i = 0

    for i in range(len(lines)):
        if '((' in lines[i]:
            tmp = lines[i]
            flag=False
            for j in range(len(tmp)):
                if tmp[j] == '(' and tmp[j+1] == '(': flag=True
                if flag and tmp[j] == ')' and tmp[j+1] == ')':
                    tmp = tmp[:j] + tmp[j+1:]
                    tmp = tmp.replace('((', '(')
                    break
            lines[i] = tmp

    vars = []
    funcs_1 = []
    funcs_2 = []
    premises = []
    try:
        vars = lines[1].split('ThingsSort, (')[1].split(') =')[0].replace(' ', '').split(',')
    except:
        if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
    if vars == ['']:
        # print(file)
        vars = ['something']
        # continue
    # print(vars)
    # print(file)
    for i in range(len(vars)):
        for j in range(len(vars[i])):
            if vars[i][j].isdigit() and vars[i][j-1] != '_' and not vars[i][j-1].isdigit():
                vars[i] = vars[i][:j] + '_' + vars[i][j:]
                # if file == '/home/XXXX/XXXX/SAT-LM/tmp/' + 'proofd5551.py':
                #     breakpoint()
                # break
    
    vars_done = False
    start_premise=False
    for line in lines[2:]:
        if "= Consts(" in line or '= Const(' in line: break
        split = line.split('=')
        # if len(split) == 1:
            #  if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
        # funcs.append(line.split(' =')[0])
        name = split[0].strip()
        # print('Function(\'' + name  + '\', ThingsSort, ThingsSort, BoolSort())' )
        try:
            if 'Function(\'' + name  + '\', ThingsSort, ThingsSort, BoolSort())' in split[1]:
                funcs_2.append(split[0].strip())
        
            elif 'ThingsSort, ThingsSort, ThingsSort' in split[1]:
                #  
                fun3_files.append(file)
                breakflag=True
                break
            else:
                funcs_1.append(split[0].strip())
        except:
            breakpoint()
    for i in range(len(funcs_1)):
        for j in range(len(funcs_1[i])):
            if funcs_1[i][j].isdigit() and funcs_1[i][j-1] != '_' and not funcs_1[i][j-1].isdigit():
                funcs_1[i] = funcs_1[i][:j] + '_' + funcs_1[i][j:]
    for i in range(len(funcs_2)):
        for j in range(len(funcs_2[i])):
            if funcs_2[i][j].isdigit() and funcs_2[i][j-1] != '_' and not funcs_2[i][j-1].isdigit():
                funcs_2[i] = funcs_2[i][:j] + '_' + funcs_2[i][j:]
    # if file == '/home/XXXX/XXXX/SAT-LM/tmp/' + 'proofd5551.py':
    #             breakpoint()
    if breakflag:
        breakflag=False
        continue
    # print(lines)
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j].isdigit() and lines[i][j-1] != '_' and not lines[i][j-1].isdigit():
                lines[i] = lines[i][:j] + '_' + lines[i][j:]
                # break
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
    #  

    # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5135.py':
    #     breakpoint()
    sym = []
    # print(lits)
    delkeys = []
    newkeys = []
    breakfile = False
    for i in range(len(lits.keys())):
        sym.append('')
    for key, value in lits.items():
        if len(key) == 3:
            var1, var2, func = key
        else:
            var1, func = key
        
        nv1, nv2, f1 = ['','','']
        # print(var1)
        if len(var1) == 0:
            print(key, file)
            # print(vars)
        #     breakfile = True
        #     break
        if var1[0].isdigit():
            nv1 = '_' + var1
        else:
            nv1 = var1
        if len(key) == 3:
            if var2[0].isdigit():
                nv2 = '_' + var2
            else:
                nv2 = var2
        # if len(key) == 3:
        if func[0].isdigit():
            f1 = '_' + func
        else:
            f1 = func
        
        delkeys.append(key)
        if len(key) == 3:
            newkeys.append([(nv1, nv2, f1), value])
        else:
            newkeys.append([(nv1, f1), value])

    #  
    # if breakfile: continue
    for key in delkeys:
        del lits[key]
    for key, value in newkeys:
        lits[key] = value
        
    for key, value in lits.items():
        # print(key)
        if len(key) == 3:
            var1 ,var2, func = key

            # print(key)
            # print(func + '__' + var1 + '__' + var2+'__' +'= symbols(' + '\'' + func+ '\'' + "+'__'+" + '\'' + var1 + '__'  + var2 + '\''+ "+'__')")
            exec(func + '__' + var1 + '__' + var2+'__' +'= symbols(' + '\'' + func+ '\'' + "+'__'+" + '\'' + var1 + '__'  + var2 + '\''+ "+'__')")
        else: 
            var, func = key
            # if key == ('first_road', 'road'):
            # 
            try:
                exec(func + '__' + var + '__' + '= symbols(' + '\'' + func+ '\'' + "+'__'+" + '\'' + var + '\'' + "+'__')")
            except:
                breakpoint()
    # need_bear_cow = 0
    #  
    # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5135.py':
    #     breakpoint()
    flag=False
    us_lines = []
    # print(vars)
    # print(funcs)
    solver=False
    n = 0
    question = ''
    #  
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
        #  
        if n == 3:
            tmp = question
        # breakpoint()
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
                        if func + '(' not in tmp:
                            continue
                        occurances = tmp.split(func + '(')
                        counter = 0
                        # breakpoint()
                        for o in range(1,len(occurances)):
                            # counter += 1
                            # if counter % 2 == 0 : continue
                            entities = occurances[o].split(')')[0].split(',')

                            old_str = func + '('
                            new_str = func 
                            for entity in entities:
                                # entity=entity.strip(' ')

                                old_str +=  entity + ','
                                new_str += '__' + entity.strip(' ')
                            old_str = old_str[:-1] + ')'
                            new_str += '__'
                            tmp = tmp.replace(old_str, new_str)

                            old_str = func + '('
                            new_str = func 
                            for entity in entities:
                                # entity=entity.strip(' ')

                                old_str +=  entity + ','
                                new_str += '__' + entity.strip(' ')
                            old_str = old_str[:-1] + ')'
                            new_str += '__'
                            tmp = tmp.replace(old_str, new_str)
                            # tmp = tmp.replace(old_str +)
                            # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
                        # tmp = tmp.replace(func + '(' + var1 + ', '+var2 + ')', func + '__' + var1 + '__'+ var2+ '__')
                        # # print(var1, var2, )
                        # #  
                        # tmp = tmp.replace(func + '(' + 'x, ' + var1 + ')', func + '__' + 'x' + '__' + var1 + '__')
                        # tmp = tmp.replace(func + '(' + var1+ ' x' + ')', func + '__' + var1+ '__' 'x' + '__')

            for func in funcs_1:
                # for var in vars:
                    #  
                    # tmp = tmp.replace(func + '(' + var + ')', func + '__' + var + '__')
                    # print(func, var, tmp)
                    # us_lines.append(tmp)
                # tmp = tmp.replace(func + '(' + 'x' + ')', func + '__' + 'x' + '__')
                if func + '(' not in tmp:
                    continue
                occurances = tmp.split(func + '(')
                counter = 0
                for o in range(1,len(occurances)):
                    # counter += 1
                    # if counter % 2 == 0 : continue
                    entities = occurances[o].split(')')[0].split(',')

                    old_str = func + '('
                    new_str = func 
                    for entity in entities:
                        # entity=entity.strip(' ')

                        old_str += entity + ','
                        new_str += '__' + entity.strip(' ')
                    old_str = old_str[:-1] + ')'
                    new_str += '__'
                    tmp = tmp.replace(old_str, new_str)
                    # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
                #  
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
                        if func + '(' not in tmp:
                            continue
                        occurances = tmp.split(func + '(')
                        counter = 0
                        for o in range(1,len(occurances)):
                            # counter += 1
                            # if counter % 2 == 0 : continue
                            entities = occurances[o].split(')')[0].split(',')

                            old_str = func + '('
                            new_str = func 
                            for entity in entities:
                                # entity=entity.strip(' ')

                                old_str +=  entity + ','
                                new_str += '__' + entity.strip(' ')
                            old_str = old_str[:-1] + ')'
                            new_str += '__'
                            tmp = tmp.replace(old_str, new_str)
                            # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
                            #  
                        # print(var1, var2, func)
            for func in funcs_1:
                for var in vars:
                    # func = funcs_2[k]
                    if func + '(' not in tmp:
                        continue
                    occurances = tmp.split(func + '(')
                    counter = 0
                    for o in range(1,len(occurances)):
                        # counter += 1
                        # if counter % 2 == 0 : continue
                        entities = occurances[o].split(')')[0].split(',')

                        old_str = func + '('
                        new_str = func 
                        for entity in entities:
                            entity=entity.strip(' ')

                            old_str += entity + ','
                            new_str += '__' + entity 
                        old_str = old_str[:-1] + ')'
                        new_str += '__'
                        tmp = tmp.replace(old_str, new_str)
                        # breakpoint()
                    # print(func, var, tmp)
                # tmp = tmp.replace(func + '(' + 'x' + ')', func + '__' + 'x' + '__')
            question=tmp
            # print(tmp)
        if n == 3:
            break
        if line.startswith('precond = []') or line.startswith('precond=[]'):
            flag=True
        # us_lines.append(tmp)

    # print('question: ', question)
    #  
    flag=False
    #  
    sym_lines = []
    for line in us_lines:
        tmp = line
        if 'ForAll(' not in line:
            flag=True
            
        # tmp = tmp.replace('precond.append(', '').replace('ForAll([x],', '').replace('ForAll([x,y]', '').replace("ForAll([x,y,z]", '')
        tmp = tmp.replace('precond.append(', '')
        # tmp = tmp.split('], ')[-1]
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
    #  
    final = []
    #  
    #  
    for line in sym_lines:
        # print(line)
        line = line.strip(', ')
        repl = []
        for v in ['__x__', '__y__', '__z__', '__a__', '__b__', '__c__']:
            if v in line:
                # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5434.py':
                #     breakpoint()
                repl.append(v)
        if len(repl) == 0:
            final.append(line)
            #  
            continue
        # breakpoint()
        forall=False
        exists=False
        if 'ForAll' in line:
            forall=True
        elif 'Exists' in line:
            exists=True
            
        line = line.split('], ')[-1]
        #  
        # if file == '/home/XXXX/XXXX/LLM-project/tmp/proofd5327.py':
        #     breakpoint()
        if forall:
            for var1 in vars:
                # if len(repl) == 0:
                #     # final.append(line)
                #     continue
                if len(repl) == 1:
                    final.append(line.replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__'))
                    # final.append(line.replace( repl[0], '__' + var1 + '__'))
                    # final.append(line.replace( repl[0], '__' + var1 + '__'))
                    final[-1] = final[-1][:-1]
                    continue
                #  
                for var2 in vars:
                    if var1 == var2: continue
                    # if len(repl) < 2: continue
                    elif len(repl) == 2:
                        final.append(line.replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        .replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        .replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        )
                        final[-1] = final[-1][:-1]

                        #  
                        continue
                    for var3 in vars:
                        # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5434.py': breakpoint()
                        if var1 == var3: continue
                        if var2 == var3: continue
                        if len(repl) < 3: continue
                        elif len(repl) == 3:
                            final.append(line.replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            .replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            .replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            )
                            final[-1] = final[-1][:-1]

                            #  
                            continue
        elif exists:
            tmp = 'Or('
            for var1 in vars:
                # if len(repl) == 0:
                #     # final.append(line)
                #     continue
                if len(repl) == 1:
                    tmp += line.replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__').replace( repl[0], '__' + var1 + '__') + ','
                    # final.append(line.replace( repl[0], '__' + var1 + '__'))
                    # final.append(line.replace( repl[0], '__' + var1 + '__'))
                    continue
                #  
                for var2 in vars:
                    if var1 == var2: continue
                    # if len(repl) < 2: continue
                    elif len(repl) == 2:
                        tmp += line.replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        .replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        .replace( repl[0], '__' + var1 + '__').replace( repl[1], '__' + var2 + '__')\
                        + ','

                        #  
                        continue
                    for var3 in vars:
                        # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5434.py': breakpoint()
                        if var1 == var3: continue
                        if var2 == var3: continue
                        if len(repl) < 3: continue
                        elif len(repl) == 3:
                            tmp += line.replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            .replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            .replace(repl[0], '__' + var1+ '__').replace(repl[1], '__' + var2 + '__').replace(repl[2], '__' + var3+'__')\
                            + ','

                            #  
                            continue
            
                    # for var in vars:
                    #     final.append(line.replace(v, var))
                    #      
            if exists: 
                final.append(tmp[:-1] + ')') 
                breakpoint()  
                # else:
                    # final.append(line)
                    #  
                    # print(' ')
    #  
    c = 0
    #  
    # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5542.py': breakpoint()
    for i in range(len(final)):
        if 'Exists' in final[i]:
            # print('ah hah')
            final[i] = final[i].replace('Exists([x],', '')
            final[i] = final[i][:-1]
    for line in final:
        if 'Implies' not in line:
            c += 1
        if 'Implies' in line:
            break
    new_final = final[c:] + final[:c]
    final = new_final
    # if 'Exists' in question or 'ForAll' in question or 'Or' in question or 'And' in question or 'Implies' in question or 'Xor' in question:
    #     # print('-------------', question, file)[]
    #     bad_qs += 1
    #     continue
        # break
    if 'Exists' in question:
        try:
            repl = question.split('Exists([')[1].split(']')[0].split(',')
        except: break
        question = question.split('], ')[-1]
        if len(repl) == 1:
            forall_question = 'Not(Or('
            for e in vars:
                # if file.endswith('575.py'):
                #     breakpoint()
                forall_question += question.replace('_' + repl[0] + '_', '_' + e + '_') + ','
            forall_question = forall_question[:-3] + '))'
        question=forall_question
    if 'ForAll' in question and 'Not(Implies' in question:
        # if file == '/home/XXXX/XXXX/LLM-project/tmp/proofd537.py':
        #     breakpoint()
        try:
            repl = question.split('ForAll([')[1].split(']')[0].split(',')
        except:
            breakpoint()
        question = question.split('], ')[-1]
        if len(repl) == 1:
            forall_question = 'And('
            for e in vars:
                # if file.endswith('575.py'):
                #     breakpoint()
                forall_question += 'Not(' + question.replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_')[:-2] + '),'
            forall_question = forall_question[:-1] + ')'
        question=forall_question
    elif 'ForAll' in question:
        # if file == '/home/XXXX/XXXX/LLM-project/tmp/proofd537.py':
        #     breakpoint()
        repl = question.split('ForAll([')[1].split(']')[0].split(',')
        question = question.split('], ')[-1]
        if len(repl) == 1:
            forall_question = 'Not(And('
            for e in vars:
                # if file.endswith('575.py'):
                #     breakpoint()
                forall_question += question.replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[0] + '_', '_' + e + '_')[:-2] + ','
            forall_question = forall_question[:-1] + '))'
        elif len(repl) == 2:
            for e in vars:
                for f in vars:
                    if e == f: continue
                    forall_question += question.replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[1] + '_', '_' + f + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[1] + '_', '_' + f + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[1] + '_', '_' + f + '_').replace('_' + repl[0] + '_', '_' + e + '_').replace('_' + repl[1] + '_', '_' + f + '_')[:-2] + ','
        try:
            question=forall_question
        except:
            breakpoint()
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

        #XXXX RIGHT HERE!!!
        # formula = formula[:-1]
        formula += new_question
        formula += ')'

        # try:
        # print(file)
        try:
            exec('f = to_cnf(' + formula + ')')
        except Exception as e:
            print(e, file)
            # if str(e).startswith('expecting bool'):

            # breakpoint()
            continue
            # breakpoint()
            # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd564.py': breakpoint()
            # continue
        # except:
        #     1 > 0
        #     print(func, file)
        #     # print(formula)
        #     continue
        # try:
        # breakpoint()
        try:
            if q == 'neg':
                f_dimacs = to_dimacs_formula(f, dimacs_mapping=f_dimacs.mapping)
            else:
                f_dimacs = to_dimacs_formula(f)
        except Exception as e:
            # breakpoint()
            print('to_dimacs error: ', e, file)
        #     # flat=1
        # except:
        #     print('to_dimacs error', formula)
        #     continue
        # if file == '/home/XXXX/XXXX/SAT-LM/tmp/proofd5150.py': breakpoint()
        dimacs = open(dimacs_dir + q + '_' + file.split('/')[-1][:-3] + '.cnf', 'w')
        dimacs.write(str(f_dimacs))
        dimacs.close()
        maptxt = open(dimacs_dir + q + '_' + file.split('/')[-1][:-3] + '.maptxt', 'w')
        import numpy as np
        # np.save(mapping, 
        # acs.mapping)
        maptxt.write(str(f_dimacs.mapping))
        maptxt.close() 
        #  

        mapping = open(dimacs_dir + q + '_' + file.split('/')[-1][:-3] + '.mapping', 'wb')
        np.save(mapping, f_dimacs.mapping)
        mapping.close()

        arities = {}
        for func in funcs_1:
            arities[func] = 1
        for func in funcs_2:
            arities[func] = 2

        pred_arity = open(dimacs_dir + q + '_' + file.split('/')[-1][:-3] + '.arity', 'wb')
        np.save(pred_arity, arities)
        pred_arity.close()
print(fun3_files)
print(len(fun3_files))
print('errored:', errored)
print('bad_qs:', bad_qs)