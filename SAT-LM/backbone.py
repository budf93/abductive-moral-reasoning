import os
import json
empath = '/home/XXXX/LLM-project/dimacs/'
dst = '/home/XXXX/LLM-project/bbones/'
c = '/home/XXXX/LLM-project/dimacs/2neg_e9e44daa18bcdca.cnf'
# os.mkdir(dst)
# breakpoint()
for file in os.listdir(empath):
# for file in [c]:
    if file.endswith('cnf') and not os.path.exists(dst + file[:-4] + '.bbone'):
        # breakpoint()
        os.system("timeout 5000 /home/XXXX/LLM-project/cadiback/cadiback " + empath + file + " > "+ dst + file[:-4] + ".bbone")

        # os.system("timeout 5000 /home/XXXX/sat_gen/CoreDetection/HardPSGEN/src/postprocess/drat-trim/drat-trim " + empath+file +" " + empath + file[:-4] + ".drat -c " + empath+file[:-4] + '_core')
    # else:
    #     print(file)

for file in os.listdir(dst):
# c = 'neg_bd0c8ba22b3e7a16.bbone'
# for file in [c]:
# if 1 > 0:
    # file = 'neg_b66e451ee5f9acc9.cnf'
    if file.endswith('bbone'):
        
        em = empath + file[:-6] + '.maptxt'
        maptxt = open(em, 'r').read()
        maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
        print(maptxt)
        mapping = json.loads(maptxt)

        bbone= open(dst + file, 'r')
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
            print(trn)
        print(file)
        # breakpoint()
        