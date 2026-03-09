import os
import json
empath = '/home/XXXX/LLM-project/dimacs/'

# for file in os.listdir(empath):
#     if file.endswith('cnf') and not os.path.exists(empath+file[:-4] + '_core'):

#         os.system("timeout 5000 /home/XXXX/sat_gen/CoreDetection/HardPSGEN/src/postprocess/cadical/build/cadical " + empath+file + " --no-binary " +  empath + file[:-4] + ".drat> "+ empath + "/solve.log")

#         os.system("timeout 5000 /home/XXXX/sat_gen/CoreDetection/HardPSGEN/src/postprocess/drat-trim/drat-trim " + empath+file +" " + empath + file[:-4] + ".drat -c " + empath+file[:-4] + '_core')

# /home/XXXX/LLM-project/dimacs/neg_b66e451ee5f9acc9.cnf
# for file in os.listdir(empath):
if 1 > 0:
    file = 'neg_0d812018f286e3d2_core'
    if file.endswith('core'):
        em = empath + file[:-5] + '.maptxt'
        maptxt = open(em, 'r').read()
        maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
        print(maptxt)
        mapping = json.loads(maptxt)

        core= open(empath + file, 'r')
        lines = core.readlines()
        
        while not lines[0].startswith('p'):
            lines = lines[1:]
        lines = lines[1:]
        trns = []
        for line in lines:
            trns.append([])
            lits = line.split(' ')[:-1]
            for lit in lits:
                if lit.startswith('-'):
                    trns[-1].append('Not ' + mapping[lit[1:]])
                else:
                    trns[-1].append(mapping[lit])
        for trn in trns:
            print(trn)
        breakpoint()
        