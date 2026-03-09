import os

path = '//home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new/'
output_path = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_core_new_output/'
os.mkdir(output_path)
# os.mkdir(path)
def main(file):
    os.system('/home/XXXX/XXXX/fs_backup_feb13/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + file + '> ' + file.replace('dimacs_clutrr_core_new', 'dimacs_clutrr_core_new_output')[:-4] + '.log')
    print(file.replace('dimacs_proofd5', 'dimacs_pronto_output')[:-4] + '.log')

args = []
for file in os.listdir(path):
    if file[-4:] == '.cnf':
        args.append(path + file)
# print(args)
from multiprocessing import Pool
pool = Pool(70)
pool.map(main, args)

pool.close()
pool.join()

