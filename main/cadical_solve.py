import os

path = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/'
output_path = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs_output/'
os.mkdir(output_path)
# os.mkdir(path)
def main(file):
    os.system('/mnt/c/Tugas_Akhir/ARGOS_public_anon/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + file + '> ' + file.replace('dimacs', 'dimacs_output')[:-4] + '.log')
    print(file.replace('dimacs', 'dimacs_output')[:-4] + '.log')

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

