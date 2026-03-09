import numpy as np
import csv
import os
#for circuit in [ 22, 23, 24,26, 27, 28]
#for circuit in [ 23]:
    #for subdir in ["atmost2_1_", "atmost3_1_", "atmost4_3_", "equal2_", "exact2_1_", "exact2_2_", "exact3_3_", "exact4_4_", "xor2_"]:
    #for subdir in ["atmost2_1_"]:
directory = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new/'
# gen_files_path = '/home/XXXX/sat_gen/CoreDetection/HardPSGEN/formulas/PS_generated/'
log_directory = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_output/'
csv_directory = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_csvs/'
times_dict = {}
finished_dict = {}
os.mkdir(csv_directory)
    
for filename in os.listdir(directory):
    # breakpoint()
    if filename[-4:] != '.cnf':
        continue
    # print('hi')
    f = os.path.join(directory, filename)
    # checking if it is a file
    # print(f)
    g = log_directory + filename[:-4] + '.log'
    if os.path.isfile(g):
        # print('were in')     

        finished = []
        
        if 1 > 0:
            #print("isfile!")
            #print(g)
            lines = []
            start_line = 0
            line_counter = 0
            with open(g) as file:
                for line in file:
                    
                    lines.append(line.strip())
                    if "resources" in line:
                        start_line = line_counter
                        #print(start_line)
                    line_counter += 1
                
            
            if start_line == 0:
                continue
            try:
                
            
                finished = int(lines[start_line+4].split("raising signal")[1].replace(" ", "")[0:2])
            except:
                
                finished = int(lines[-1][-2:])

            else:
                print("not a file:",g)
        finished_dict[filename] = finished

dic = {}
print(finished_dict)
for file in finished_dict.keys(): 
    pol, file_base = file.split('_')
    if file_base not in dic.keys():
        dic[file_base] = ['pos', 'neg']
    if pol == 'pos':
        dic[file_base][0] = int(finished_dict[file])
    elif pol == 'neg':
        dic[file_base][1] = int(finished_dict[file])
# print(finished_dict)
# breakpoint()
delkeys = []
for key, value in dic.items():
    pos, neg = value
    if 'pos' == pos or 'neg' == pos:
        delkeys.append(key)
        continue
    elif 'neg' == neg or 'pos' == neg:
        delkeys.append(key)
        continue
for key in delkeys:
    del dic[key]
posl = []
negl = []
name = []
for key, value in dic.items():
    pos, neg = value
    if pos == 10:
        # print('hi')
        posl.append('SAT')
    elif pos == 20:
        # print('hi')
        posl.append('UNSAT')
    else:
        print('pos', key, pos)
    if neg == 10:    
        # print('hi')
        negl.append('SAT')
    elif neg == 20:
        # print('hi')
        negl.append('UNSAT')
    else:
        print('neg', key, neg)
    name.append(key)

print(len(posl), len(negl), len(name))
import pandas as pd

df = pd.DataFrame({'name':name, 'pos': posl, 'neg': negl})
df.to_csv(csv_directory + '/solver_finished' + '.csv')

# with open(csv_directory +'/solver_finished'+ '.csv', 'w') as csv_file:  
#     writer = csv.writer(csv_file)
#     for key, value in finished_dict.items():
#         writer.writerow([key, value])

