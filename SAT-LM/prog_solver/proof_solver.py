import sys
sys.path.append('.')

import re
import os
import pickle

from func_timeout import func_timeout

from prog_solver.z3_utils import make_z3_enum_line, execute_z3_test

def break_down_func_var():
    pass

VAR_REGEX = r"[_a-zA-Z0-9]+[,)]"
NUM_VAR_REGEX = r"[_0-9]+[,)]"
FUNC_REGEX = r"[_a-zA-Z0-9]+[(]"

PREDEFIND_FUNCS = ["ForAll", "Exists", "And", "Or", "Not", "Implies", "Xor"]
PREDEFIND_QUNT_VARS = ["x", "y", "z"]

def extract_var_and_func(line):
    all_vars = re.findall(VAR_REGEX, line)
    all_funcs = re.findall(FUNC_REGEX, line)
    all_vars = [all_vars.rstrip(",)") for all_vars in all_vars]
    all_funcs = [all_funcs.rstrip("(") for all_funcs in all_funcs]
    # breakpoint()
    return all_vars, all_funcs

def determine_func_n_args(code, func):
    # print(func)
    start_pos = code.find(func + "(")
    end_pos = code.find(")", start_pos)
    num_args = code[start_pos+len(func)+1:end_pos].count(",") + 1
    # if 'is_in' in func:
    #     # print(code[start_pos:])
    #     print('code:',code[start_pos:end_pos], start_pos, end_pos, code)
    # if func == 'in':
    #     print('IN IN IN', code[start_pos:end_pos])

    return num_args
    
        
def proof_satlm_exec(code, prompting_style, return_code=False, filename=None):
    # breakpoint()
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    if not lines[-1].startswith("return"):
        # breakpoint()
        return code, "Error no return"
    # breakpoint()
    result_line = lines[-1]
    lines = lines[:-1]
    # breakpoint()
    vars = set()
    functions = set()

    for line in lines + [result_line]:
        line_vars, line_funcs = extract_var_and_func(line)
        # breakpoint()
        vars.update(line_vars)
        functions.update(line_funcs)
        
    num_vars = []
    for var in vars:
        if var[0].isdigit():
            num_vars.append(var)
    for num_var in num_vars:
        vars.remove(num_var)
        vars.add('_' + num_var)
        for i in range(len(lines)):
            if num_var in lines[i]:
                lines[i] = lines[i].replace(num_var, '_' + num_var)
        code = code.replace(num_var, '_' + num_var)
    try:
        functions.remove('return')
    except:
        print('no \'return\'')
    try:
        functions.remove('in')
        functions.add('is_in')
        # print('removed \'is\'')
        for i in range(len(lines)):
            if 'in(' in lines[i]:
                lines[i] = lines[i].replace('in', 'is_in')
        code = code.replace('in', 'is_in')
        # code = code.replace('is(', 'is_in(')
    except:
        1 > 0
    consts = set()
    for line in lines:
        if 'ForAll' in line:
            c = line.split('ForAll([')[1].split(']')[0].split(',')
            c = [a.strip(' ') for a in c]
            consts.update(c)
    for c in PREDEFIND_QUNT_VARS:
        consts.update(c)
    vars = [v for v in vars if v not in list(vars.intersection(consts))]
    # breakpoint()
    # breakpoint()
    vars = [v for v in vars if v not in PREDEFIND_QUNT_VARS]
    for v in range(len(vars)):
        num_var = re.findall(r"[0-9]+", vars[v])
        if len(num_var) == 0:
            continue
        if num_var[0] == vars[v]:
            vars[v] = '_' + vars[v]
   
    functions = [f for f in functions if f not in PREDEFIND_FUNCS]
    # if filename == 'proofd589':
    #     breakpoint()
    func_n_args = {}
    for func in functions:
        func_n_args[func] = determine_func_n_args(code, func)
        # if func == 'is_in': print('code:', asdf)

    functions = sorted(functions, key=lambda x: func_n_args[x])

    translated_lines = []
    translated_lines.append(make_z3_enum_line("ThingsSort", vars))

    for func in functions:
        num_args = func_n_args[func]
        translated_lines.append("{} = Function('{}', {}, BoolSort())".format(func, func, ", ".join(["ThingsSort"]*num_args)))
    # translated_lines.append("x = Consts('x', ThingsSort)")
    set_consts = consts
    consts = list(consts)
    if len(consts) == 0:
        z = 1
    elif len(consts) > 1:
        const_str = ''
        const_arg_str = ''
        for c in consts:
            const_str += str(c) + ', '
            const_arg_str += str(c) + ' '
        const_str = const_str[:-2]
        const_arg_str = const_arg_str[:-1]
        translated_lines.append(const_str + ' = Consts(\'' + const_arg_str + '\', ThingsSort)')
    else:
        translated_lines.append(str(consts[0]) + ' = Const(\'' + str(consts[0]) + '\', ThingsSort)')
    # breakpoint()
    translated_lines.append("precond = []")

    for line in lines:
        # breakpoint()
        if line == '():':
            continue
        # if len(return_var) != 0:
        line_vars = re.findall(r" [0-9]+[,)]", line)
        line_vars = [var.strip(',)') for var in line_vars]
        for var in line_vars:
            if '_' + var in line:
                continue
            else:
                line = line.replace(var[1:], '_' + var[1:])
        args = set()
        for func in functions:
            if func in line:
                f_arg = line.split(func + '(')
                even = True
                for split in f_arg:
                    if even:
                        even=False
                        continue
                    arguments = split.split(')')[0].split(',')
                    args.update([a.strip(' ') for a in arguments])
                    # if 'Implies(universal(x), ForAll([p1, p2], Implies(And(know(p1, x), know(p2, x)), communicate(p1, p2))))' in line: print(args)

        func = 'ForAll'
        ForAllConst = set()
        # missing = []
        missing_str = ''
        # if 'Implies(universal(x), ForAll([p1, p2], Implies(And(know(p1, x), know(p2, x)), communicate(p1, p2)))))' in line: breakpoint()
        # if 'Implies(universal(x), ForAll([p1, p2], Implies(And(know(p1, x), know(p2, x)), communicate(p1, p2))))' in line: breakpoint()
        # print('testing, testing')
        if func in line:
            f_arg = line.split(func + '([')
            even = True
            for split in f_arg:
                if even:
                    even=False
                    continue
                arguments = split.split('],')[0].split(',')
                ForAllConst.update([a.strip(' ') for a in arguments])
                # for a in arguments:
                #     ForAllConst.update(a.strip(' '))
                # if 'Implies(universal(x), ForAll([p1, p2], Implies(And(know(p1, x), know(p2, x)), communicate(p1, p2))))' in line: print('ForAllConst:', func,ForAllConst)

        for const in consts:
            if const in args and not const in ForAllConst:
                # missing.append(const)
                missing_str += str(const) + ', '
            
        missing_str = missing_str[:-2]
        if missing_str != '':
            translated_lines.append("precond.append(ForAll(["+ missing_str + "], {}))".format(line))

        else: translated_lines.append("precond.append({})".format(line))

    translated_lines.append("s = Solver()")
    translated_lines.append("s.add(precond)")

    return_clause = result_line.split("return")[1].strip()
    return_var = re.findall(r"[0-9]+[,)]", return_clause)
    if len(return_var) != 0:
        return_uvar = re.findall(r"_[0-9]+[,)]", return_clause)
        if len(return_uvar) == 0:
            return_clause = return_clause.replace(return_var[0].strip(',)'), "_" + return_var[0].strip(',)'))

    # breakpoint()
    # if filename == 'proofd589':
    #     breakpoint()
    translated_lines.append("s.add(Not({}))".format(return_clause))
    translated_lines.extend([
        "if s.check() == unsat:",
        "    print('True')",
        "else:",
        "    print('False')",
    ])
    # translated_lines.extend(["dimacs = open('/home/XXXX/SAT-LM/dimacs/test.dimacs', 'w')", "dimacs.write(s.dimacs())", "dimacs.close()"])
    translated_lines = ["from z3 import *"] + translated_lines

    code = "\n".join(translated_lines)
    filename, result = execute_z3_test(code, flag_keepfile=True, filename=filename)
    if return_code:
        print(filename)
        return code, result
    else:
        print(filename)
        return  result


def proof_proglm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    function_wrap = "def solution():\n" + "\n".join(["    " + line for line in lines])
    def func():
        exec(function_wrap)
        return eval('solution()')

    result = func_timeout(1, func)
    if return_code:
        return function_wrap, result
    else:
        return result


def test_sat():
    gts = ["True", "False", "True", "False"]

    with open("temp.py") as f:
        output_code = f.read()
    examples = output_code.split('\n\n\n\n\n')

    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        code, (status, result) = proof_satlm_exec(ex, "satlm", return_code=True)
        print(result, gts[i])
        # if result != gts[i]:
        #     print(code)
        #     exit()

if __name__=="__main__":
    test_sat()
