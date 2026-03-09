import os
import json
import pickle as pkl
import csv
import shutil
        
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


path = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core'

newpath = path + '_new'

# os.mkdir(newpath)


c = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_csvs/solver_finished.csv'


temp_outs_str = '/home/XXXX/XXXX/fs_backup_feb13/all_outs_cot_met_clutrr_rulethresh_03_cot_thresh_100_anneal_01,_dynamic_True,_sc5,llama_70B,_no_RULES_IN_PROMPT_fixed,_yes_separation,_always_YN_WITH_MAYBE,_og_prompt,_no_cot_gen,_augmented_extr,_expweight.pkl'
temp_outs = pkl.load(open(temp_outs_str, 'rb'))


# predvars = {}
# for key, value in temp_outs.items():
#     predvars[key] = []
#     for i in range(2, int(((len(value[0])-1))/3)+2, 3):
#         predvars[key].append(value[0][i])

# su = set(['SAT', 'UNSAT'])

r = csv.reader(open(c, 'r'))

# for row in r:
#     # if row[3] == 'UNSAT': breakpoint()
#     if set(row[2:]) == su:
#         # breakpoint()
#         files  = ['pos_' + row[1], 'neg_' + row[1]]

#         for file in files:
#             shutil.copyfile(path +'/'+ file, newpath + '/' + file)

#         em = path +'/' +  'pos_' + row[1].replace('.cnf', '.maptxt')

#         maptxt = open(em, 'r').read()

            
#         maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
#         # print(maptxt)
#         mapping = json.loads(maptxt)
#         inv_map = {v: k for k, v in mapping.items()}
#         skipfile=False

#         for file in files:
            
#             try:
#                 for var in predvars[row[1]]:

#                     add_clause(newpath + '/' + file)
#                     cf = open(newpath + '/' + file, 'a')

#                     nv = inv_map[var]
                

#                     cf.write('\n' + str(nv) + ' 0')
#                     cf.close()
#             except: skipfile=True
#         if skipfile:
#             for file in files:
#                 os.remove(newpath + '/' + file)
    # else: breakpoint()


nc = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_csvs/solver_finished.csv'

rnc = csv.reader(open(nc, 'r'))
nc_dict = {}
skipheader=True
for row in rnc:
    if skipheader: 
        skipheader=False
        continue

    nc_dict[row[1]] = row[2:]

c_dict = {}

skipheader=True
for row in r:
    if skipheader: 
        skipheader=False
        continue

    c_dict[row[1]] = row[2:]

same = 0
for key, value in nc_dict.items():
    if value == c_dict[key]:
        same += 1

print('sameness: ', same/len(nc_dict))