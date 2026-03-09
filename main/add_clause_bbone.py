import os
import json
import shutil
import numpy as np
import random

def add_clause(file):
    f = open(file, 'r')
    lines = f.readlines()
    writestr = ''
    for line in lines:
        if line.startswith('p cnf'):
            num_var, num_clause = [line.split(' ')[2], line.split(' ')[3]]
            writestr += 'p cnf ' + str(num_var) + ' ' + str(int(num_clause) + 1) + '\n'
        else:
            writestr += line
    f.close()

    f = open(file, 'w')
    f.write(writestr)
    f.close()
    # breakpoint()


og = '/home/XXXX/LLM-project/dimacs/neg_bd0c8ba22b3e7a16.cnf'

cnf = '/home/XXXX/LLM-project/neg_bd0c8ba22b3e7a16.cnf'

shutil.copy(og, cnf)

add_clause(cnf)

file = '/home/XXXX/LLM-project/tmp.cnf'

shutil.copy(cnf, file)

empath = '/home/XXXX/LLM-project/dimacs/'
# dst = '/home/XXXX/LLM-project/bbones/'
c = '/home/XXXX/LLM-project/dimacs/neg_bd0c8ba22b3e7a16.cnf'

bvars = []
# os.mkdir(dst)
# breakpoint()
while True:
    # breakpoint()
    file = '/home/XXXX/LLM-project/tmp.cnf'
    shutil.copy(cnf, file)

    tmp = []

    cf = open(file, 'r')
    for line in cf.readlines():
        if line.startswith('p cnf'):
            num_var, num_clause = [line.split(' ')[2], line.split(' ')[3]]
    cf.close()

    cf = open(file, 'a')
    
    s = np.random.randint(0, 2, size=2)

    v = random.sample(list(range(1, int(num_var) + 1)), k=2)

    new_clause = []

    for i in range(len(v)):
        if s[i] == 0:
            new_clause.append(-v[i])
        else:
            new_clause.append(v[i])
    writestr = '\n'
    for var in new_clause:
        writestr += str(var) + ' '
    writestr += '0'

    # writestr = '\n 23 -33 0'
    # print(new_clause)
    cf.write(writestr)
    cf.close()
    nc_trns = []
    em = empath + '/neg_bd0c8ba22b3e7a16.maptxt'
    maptxt = open(em, 'r').read()
    maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
    # print(maptxt)
    mapping = json.loads(maptxt)
    for lit in new_clause:
        # print(lit)
        if lit == 0:
            continue
        if lit < 0:
            nc_trns.append('Not ' + mapping[str(-lit)])
        else:
            nc_trns.append(mapping[str(lit)])
    # for trn in trns:
    #     # print(trn)
    #     tmp.append(trn)
    # print(trns)
    # print(writestr)
    # breakpoint()
# for file in [c]:
    os.system('/home/XXXX/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + file + '> ' + file.replace('dimacs', '')[:-4] + '.log')

    log = file.replace('dimacs', '')[:-4] + '.log'

    lf = open(log, 'r')
    lines = lf.readlines()

    el = lines[-1]
    # print(el)
    ec = el.split('exit ')[1].strip('\n')
    lf.close()
    if ec == '20':
        shutil.copy(file, cnf)
        break
    # if file.endswith('cnf') and not os.path.exists(file[:-4] + '.bbone'):
        # breakpoint()
    os.system("timeout 5000 /home/XXXX/LLM-project/cadiback/cadiback "  + file + " > "+  file[:-4] + ".bbone")

        # os.system("timeout 5000 /home/XXXX/sat_gen/CoreDetection/HardPSGEN/src/postprocess/drat-trim/drat-trim " + empath+file +" " + empath + file[:-4] + ".drat -c " + empath+file[:-4] + '_core')
    # else:
    #     print(file)
# file
# for file in os.listdir(dst):
    # file = '/home/XXXX/LLM-project/neg_bd0c8ba22b3e7a16.bbone'

# for file in [c]:
# if 1 > 0:
    # file = 'neg_b66e451ee5f9acc9.cnf'
    # if file.endswith('bbone'):
    if 1 > 0:
        # em = empath + file.split('/')[-1][:-4] + '.maptxt'
        em = empath + '/neg_bd0c8ba22b3e7a16.maptxt'
        maptxt = open(em, 'r').read()
        maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
        # print(maptxt)
        mapping = json.loads(maptxt)

        bbone= open( file[:-4] + '.bbone', 'r')
        lines = bbone.readlines()
        
        
        trns = []
        for line in lines:
            if line.startswith('b'):
                trns.append([])
                lits = line.split(' ')[1:]
                # if lits == ['0']:
                    # continue
                # breakpoint()

                for lit in lits:
                    lit = lit.strip()
                    # print(lit)
                    if lit == '0':
                        continue
                    if lit.startswith('-'):
                        trns[-1].append('Not ' + mapping[lit[1:]])
                    else:
                        trns[-1].append(mapping[lit])
        for trn in trns:
            # print(trn)
            tmp.append(trn)
        if len(tmp) > len(bvars):
            bvars = tmp
            add_clause(file)
            shutil.copy(file, cnf)
            print(new_clause)
            print(nc_trns)
        # else:
        print('loop')
        # print(file)
        # breakpoint()
        