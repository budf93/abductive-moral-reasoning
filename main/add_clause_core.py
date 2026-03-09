import os
import numpy as np
import random
import shutil
cnf = '/home/XXXX/LLM-project/neg_bd0c8ba22b3e7a16.cnf'
file = '/home/XXXX/LLM-project/tmp.cnf'
shutil.copy(cnf, file)
while True:
    

    
    file = '/home/XXXX/LLM-project/tmp.cnf'
    shutil.copy(cnf, file)
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
    print(new_clause)
    cf.write(writestr)
    cf.close()
    
    os.system('/home/XXXX/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + file + '> ' + file.replace('dimacs', '')[:-4] + '.log')

    log = file.replace('dimacs', '')[:-4] + '.log'

    lf = open(log, 'r')
    lines = lf.readlines()

    el = lines[-1]
    print(el)
    ec = el.split('exit ')[1].strip('\n')
    lf.close()
    if ec == '20':
        break

    # breakpoint()
