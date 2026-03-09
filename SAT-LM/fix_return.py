import os
import json
tmp = '/home/XXXX/SAT-LM/tmp/'
tmp_fixed = '/home/XXXX/SAT-LM/tmp_fixed/'
# os.mkdir(tmp_fixed)


for file in os.listdir(tmp):
    tf = open(tmp_fixed + file, 'w')
    writestr = ''
    # t = open()
    for line in open(tmp + file, 'r').readlines():
        if line.startswith('return'):
            continue
        writestr+= line

    tf.write(writestr)
    tf.close()