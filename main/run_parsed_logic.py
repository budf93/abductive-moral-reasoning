
from subprocess import check_output
import subprocess
import os
import shutil
tried = 0
failed = 0
path = '/home/XXXX/XXXX/LLM-project/tmp/'
for filename in os.listdir(path):
    if open(path+filename, 'r').read().startswith('skip this one'):
        continue
    tried += 1
    try:
        output = check_output(["python", path+filename], stderr=subprocess.STDOUT, timeout=1000)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8").strip().splitlines()[-1]
        failed += 1
        print(path+filename)
        print(output)
        continue
    shutil.copyfile(path + filename, '/home/XXXX/XXXX/LLM-project/tmp_logic/' + filename)
print(tried, failed)