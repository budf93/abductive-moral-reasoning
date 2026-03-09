import os
import json
import csv
import re
ds = json.load(open('/home/XXXX/XXXX/SAT-LM/data/proofd5_test.json', 'r'))

def convert_to_target_format(logic_str):
    # Replace universal quantifier ∀x with ForAll([x], ...)
    const = []
    
    if '∀' in logic_str:
        for z in logic_str.split('∀'):
            # print(z)
            if len(z) == 0: continue
            # print(z[0])
            if z[0] not in const:
                const.append(z[0])

    # print(const)
    # print(const)
    for c in const:
        logic_str = logic_str.replace('∀' + c, '').strip(' ')[1:-1]
    
    # print(logic_str)
    r = re.compile("¬([^\(][^ ][^\)]*\))")
    old_logic=logic_str
    logic_str = r.sub(r"Not(\1)", logic_str)
    # print('look here:', logic_str)
    # print(logic_str)
    # print(old_logic)
    while old_logic != logic_str:
        # r = re.compile("¬([^\(][^ ][.^\(]*\))")
        old_logic=logic_str
        logic_str = r.sub(r"Not(\1)", logic_str)
        # print(logic_str)
        # print(old_logic)
    
    last_op = ''
    consec_or = 1
    consec_and = 1
    tmp_or = 1
    tmp_and = 1
    ops = ['∨', '∧', '¬', '→', '⊕']
    for c in logic_str:
        # print(c)
        if c == '∨':
            if last_op == '∨':
                tmp_or += 1
            last_op = '∨'
            # print(last_op, c, last_op == c)
        elif c in ops and c != '∨':
            tmp_or = 1
            last_op = ''
            if consec_or < tmp_or: consec_or = tmp_or
            # print(c)
    if consec_or < tmp_or: consec_or = tmp_or

    last_op = ''
    for c in logic_str:
        if c == '∧':
            if last_op == '∧':
                tmp_and += 1
                print(c, logic_str)

            last_op = '∧'
            
        elif c in ops and c != '∧':
            consec_and = 1
            last_op = ''
            if consec_and < tmp_and: consec_and = tmp_and
            # print(c, logic_str)
    if consec_and < tmp_and: consec_and = tmp_and

    rstr = ''
    # print(consec_and)
    if consec_and > 1:
        for m in range(consec_and+1):
            rstr += '([^ ]*\([^\(]*\)) ∧ '
        rstr = rstr[:-3]
        r = re.compile(rstr)
        rsubstr = r'And('
        for m in range(1,1+1+consec_and):
            rsubstr += '\\' + str(m) + ', '
        rsubstr = rsubstr[:-2] + ')'
        # logic_str = r.sub(r
        logic_str = r.sub(rsubstr, logic_str)
        # print(rsubstr)
    if consec_or > 1:
        for m in range(consec_or+1):
            rstr += '([^ ]*\([^\(]*\)) ∨ '
        rstr = rstr[:-3]
        r = re.compile(rstr)
        rsubstr = r'Or('
        for m in range(1, 1+1+consec_or):
            rsubstr += '\\' + str(m) + ', '
        rsubstr = rsubstr[:-2] + ')'
        # logic_str = r.sub(r
        logic_str = r.sub(rsubstr, logic_str)
        # print(rsubstr)
    # print(consec_or)
    # Replace conjunction (∧) with And
    # print(logic_str)
    r = re.compile("¬\(([^ ]*\([^\(]*\)) ∨ ([^ ]*\([^\(^)]*\))\)")
    logic_str = r.sub(r"Not(Or(\1, \2))", logic_str)
    
    r = re.compile("([^ ]*\([^\(]*\)) ∨ ([^ ]*\([^\(]*\))")

    logic_str = r.sub(r"Or(\1, \2)", logic_str)    

    r = re.compile("¬\(([^ ]*\([^\(]*\)) ∧ ([^ ]*\([^\(^)]*\))\)")
    logic_str = r.sub(r"Not(And(\1, \2))", logic_str)
    
    r = re.compile("([^ ]*\([^\(]*\)) ∧ ([^ ]*\([^\(]*\))")
    logic_str = r.sub(r"And(\1, \2)", logic_str).replace('And ', 'And')

    
    r = re.compile("¬\(([^ ]*\([^\(]*\)) ⊕ ([^ ]*\([^\(^)]*\))\)")
    logic_str = r.sub(r"Not(Xor(\1, \2))", logic_str)

    r = re.compile("^(Not\([^ ]*\([^\(]*\)\)) ⊕ (Not\([^ ]*\([^\(^)]*\)\))$")
    logic_str = r.sub(r"Xor(\1, \2)", logic_str)
    
    r = re.compile("^(Not\([^ ]*\([^\(]*\)\)) ⊕ ([^ ]*\([^\(^)]*\))$")
    logic_str = r.sub(r"Xor(\1, \2)", logic_str)

    
    r = re.compile("^([^ ]*\([^\(]*\)) ⊕ (Not\([^ ]*\([^\(^)]*\)\))$")
    logic_str = r.sub(r"Xor(\1, \2)", logic_str)

    r = re.compile("^([^ ]*\([^\(]*\)) ⊕ ([^ ]*\([^\(^)]*\))$")
    logic_str = r.sub(r"Xor(\1, \2)", logic_str)

    r = re.compile("→ ([^ ]*\([^\(]*\)) ⊕ ([^ ]*\([^\(^)]*\))$")
    logic_str = r.sub(r"→ Xor(\1, \2)", logic_str)
    
    r = re.compile("\(([^ ]*\([^\(]*\)) ⊕ ([^ ]*\([^\(^)]*\))\)")
    logic_str = r.sub(r"Xor(\1, \2)", logic_str)

    # r = re.compile("¬\((.*)\)")

    # logic_str = r.sub(r"Not(\1)", logic_str)

    r = re.compile("(.*)→(.*)")
    logic_str = r.sub(r"Implies(\1, \2)", logic_str).replace(' ,', ',').replace('  ', ' ')
    
    # logic_str = re.sub(r"∀x", "ForAll([x],", logic_str)
    
    # Replace implication (→) with Implies
    
    
    # Replace negation (¬) with Not

    
    
    # Replace disjunction (∨) with Or
  
    # Make predicates lowercase for consistency
    # logic_str = re.sub(r"\b(Eel|Fish|Tree|DisplayedIn|Plant|LivingCreature|ComplexCell|Bacteria|Animal|Multicellular|ShownIn|seaSnake|seaEel)\b", lambda m: m.group(0).lower(), logic_str)
    old_logic_str = logic_str
    r = re.compile("(.*)([a-z])([A-Z])(.*)")
    logic_str = r.sub(r"\1\2_\3\4", logic_str)
 
    while old_logic_str != logic_str:
        old_logic_str = logic_str
        r = re.compile("(.*)([a-z])([A-Z])(.*)")
        logic_str = r.sub(r"\1\2_\3\4", logic_str)
        

    
    # Handle cases where functions or predicates are followed by variables
    logic_str = re.sub(r"([a-zA-Z]+)\((\w+)\)", r"\1(\2)", logic_str)
    if len(const) != 0:
        new_str = 'ForAll(['
        for c in const:
            new_str += c + ','
        logic_str = new_str[:-1] + '], ' + logic_str + ')'
    # if 'Implies' in logic_str:
        # logic_str = logic_str.replace('Implies(', 'Implies')[:-1]
    return logic_str.replace('( ', '(').replace(' (', '(')

def extract_predicates_and_objects(logic_str):
    # Regex to match predicates and objects inside parentheses
    predicate_pattern = r"([a-zA-Z]+)\(([^)]+)\)"
    
    # Find all matches
    matches = re.findall(predicate_pattern, logic_str)
    
    extracted = []
    
    # Process the matches to separate predicate and objects
    for predicate, objects in matches:
        object_list = objects.split(",")  # Split objects by comma if there are multiple
        extracted.append((predicate, object_list))
    
    return extracted

namecounter = 1
# breake=False
breakecount = 0
# print(len(ds))
for d in ds:
    breake=False
    logic_statements = d['premises-FOL'].split('\n') + [d['conclusion-FOL']]
    
    predicates = []
    entities = []
    arity = {}
    # print(logic_statements)


        # print("-" * 50)
    # print(entities, predicates)
    const = []
    for statement in logic_statements:
        if '∃' in statement:
            file = open('/home/XXXX/XXXX/LLM-project/tmp/proofd5' + str(namecounter) + '.py', 'w')
            file.write('skip this one')
            file.close()
            namecounter += 1
            breake = True
            breakecount += 1
            break
    if breake:continue
    space_ops = ['∧', '⊕', '∨']
    for s in range(len(logic_statements)):
        logic_statements[s] = logic_statements[s].replace('  ', ' ')
        statement = logic_statements[s]
        c = 0
        while c < (len(logic_statements[s])):
            # print(logic_statements[s])
            # if namecounter == 1:
            #     breakpoint()
            if logic_statements[s][c] in space_ops:
                # breakpoint()
                if logic_statements[s][c-1] != ' ':
                    logic_statements[s] = logic_statements[s][:c] + ' ' + logic_statements[s][c:]
                    # breakpoint()
                if logic_statements[s][c+1] != ' ':
                    logic_statements[s] = logic_statements[s][:c+2] + ' ' + logic_statements[s][c+2:]
                    # breakpoint()
            c += 1
    # print(c)
    # print(const)

    converted_statements = [convert_to_target_format(statement) for statement in logic_statements]
    for statement in converted_statements:
        if 'ForAll' in statement:
            consts = statement.split('ForAll([')[1].split(']')[0].split(',')
            # print('yeah')
            for c in consts:
                if c not in const: const.append(c)
    # print(const)
    for statement in logic_statements:
        extracted = extract_predicates_and_objects(statement)
        # print(extracted)
        # print(f"Logic Statement: {statement}")
        for predicate, objects in extracted:
            old_logic_str = predicate
            r = re.compile("(.*)([a-z])([A-Z])(.*)")
            logic_str = r.sub(r"\1\2_\3\4", predicate)
            while old_logic_str != logic_str:
                old_logic_str = logic_str
                r = re.compile("(.*)([a-z])([A-Z])(.*)")
                logic_str = r.sub(r"\1\2_\3\4", logic_str)
                logic_str = logic_str
            logic_str = logic_str.lower().strip(' ')

            if logic_str not in predicates:
                predicates.append(logic_str)
                arity[logic_str] = len(objects)
            # print(f"Predicate: {predicate}, Objects: {objects}")
            for o in objects:
                # print(o)
                old_logic_str = o
                r = re.compile("(.*)([a-z])([A-Z])(.*)")
                logic_str = r.sub(r"\1\2_\3\4", o)
                while old_logic_str != logic_str:
                    old_logic_str = logic_str
                    r = re.compile("(.*)([a-z])([A-Z])(.*)")
                    logic_str = r.sub(r"\1\2_\3\4", logic_str)
                logic_str = logic_str.lower().strip(' ')
                if o in const:continue
                if logic_str not in entities:
                    entities.append(logic_str)
    


    for i in range(len(converted_statements)):
        # converted_statements[i] = converted_statements[i].lower().replace('implies(', 'Implies(').replace(' not(', ' Not(').replace('(not(', '(Not(').replace('(or(', '(Or(').replace(' xor(', ' Xor(').replace('(xor(', '(Xor(').replace(' xOr', ' Xor').replace('(xOr', '(Xor').replace('forall(', 'ForAll(').replace(' and(', ' And(').replace('(and(', '(And(')
        converted_statements[i] = converted_statements[i].lower()
        
        r = re.compile('([^a-z]+)not([^a-z]+)')
        converted_statements[i] = r.sub(r"\1Not\2", converted_statements[i])

        r = re.compile('^not([^a-z]+)')
        converted_statements[i] = r.sub(r"Not\1", converted_statements[i])

        r = re.compile('([^a-z]+)and([^a-z]+)')
        converted_statements[i] = r.sub(r"\1And\2", converted_statements[i])

        r = re.compile('^and([^a-z]+)')
        converted_statements[i] = r.sub(r"And\1", converted_statements[i])
        # breakpoint()
        r = re.compile('([^a-z]+)or([^a-z]+)')
        converted_statements[i] = r.sub(r"\1Or\2", converted_statements[i])

        r = re.compile('^or([^a-z]+)')
        converted_statements[i] = r.sub(r"Or\1", converted_statements[i])


        r = re.compile('([^a-z]+)xor([^a-z]+)')
        converted_statements[i] = r.sub(r"\1Xor\2", converted_statements[i])

        r = re.compile('^xor([^a-z]+)')
        converted_statements[i] = r.sub(r"Xor\1", converted_statements[i])
        # r = re.compile('([^a-z])implies([^a-z])')
        # converted_statements[i] = r.sub(r"\1Implies\2", converted_statements[i])
        converted_statements[i] = converted_statements[i].replace('forall', 'ForAll').replace('implies', 'Implies')

        # if namecounter==57:
        #     breakpoint()
    for i in range(len(predicates)):
        if predicates[i] == 'with':
            predicates[i] = '_with'
            for j in range(len(converted_statements)):
                converted_statements[j] = converted_statements[j].replace('with', '_with')

            arity['_with'] = arity['with']
            del arity['with']
    file_str = ''
    file_str += 'from z3 import *\n'
    file_str += 'ThingsSort, (' 
    for e in entities:
        file_str +=  e + ', '
    file_str = file_str[:-2]    
    file_str += ') = EnumSort(\'ThingsSort\', ['

    for e in entities:
        file_str += '\'' + e + '\', '
    file_str = file_str[:-2] + '])\n'
    for p in predicates:
        file_str += p + ' = Function(\'' + p + '\', '
        for r in range(arity[p]):
            file_str += 'ThingsSort, '
        file_str += 'BoolSort())\n'
    # print(file_str)
    for c in const:
        file_str += c + ', '
    if len(const) == 1:
        file_str = file_str[:-2] + ' = Const(\''
    else:
        file_str = file_str[:-2] + ' = Consts(\''

    for c in const:
        file_str += c + ' '
    file_str = file_str[:-1] + '\', ThingsSort)\nprecond=[]\n'

    for statement in converted_statements[:-1]:
        file_str += 'precond.append(' + statement + ')\n'

    file_str += 's = Solver()\ns.add(precond)\n'
    file_str += 's.add(Not(' + converted_statements[-1] + '))\n'
    file_str += 'if s.check() == unsat:\n    print(\'True\')\nelse:\n    print(\'False\')'
    # print(file_str)
    file = open('/home/XXXX/XXXX/LLM-project/tmp/proofd5' + str(namecounter) + '.py', 'w')
    file.write(file_str)
    file.close()
    namecounter += 1
print(breakecount)