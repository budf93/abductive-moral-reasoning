import os
import shutil
import numpy as np
import time
import json

import torch, os
from torch.utils.data import DataLoader
os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
run_log_path = './run_log.txt'
run_log = open(run_log_path, 'w')
run_log.write('hello\n')
run_log.close()
unknown=False

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
cache_dir = '/media/data1/XXXX/hub/'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
# import transformers

# import urllib3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import time
import datetime
import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = {'train_file_path': './example_data', 'test_file_path': './example_data', 'save_path': './../SFT_train_res', 'engine': 'meta-llama/Llama-2-13b-chat-hf', 
        'n_rows': 20, 'max_length': 200,'temperature': 0.1, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
# args['engine'] = 'meta-llama/Meta-Llama-3-8B-Instruct'
# args['engine'] = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
args['engine'] = 'Qwen/Qwen2.5-Coder-3B-Instruct'
args = Struct(**args)

class LLM():
    def __init__(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        with no_ssl_verification():
            

            
            self.tokenizer = AutoTokenizer.from_pretrained(
                    args.engine,
                    cache_dir = cache_dir
                    )

            self.model = AutoModelForCausalLM.from_pretrained(
                    args.engine, 
                    cache_dir = cache_dir,
                    quantization_config=quant_config,
                    device_map='auto'
                    )

        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def sentence_probabilities(self, sentences):
        with torch.no_grad():
            sentence_tokens = self.tokenizer(sentences, return_tensors='pt', padding=True)
            sentence_token_ids = sentence_tokens.input_ids

            # Little hack to cut down inference time by 4-5x (leads to some imprecisions when using quantization)
            # Find the common prefix and run it through the model once, to save time
            first_different_token = (sentence_token_ids == sentence_token_ids[0, :].unsqueeze(0)).all(dim=0).long().argmin()
            common_prefix = sentence_token_ids[0, :first_different_token].unsqueeze(0)
            common_prefix_output = self.model(common_prefix, use_cache=True)
            common_prefix_key_values = tuple(tuple(tensor.expand(len(sentences), -1, -1, -1) for tensor in layer) 
                                             for layer in common_prefix_output.past_key_values)

            # Process the rest of the sentences
            rest_outputs = self.model(sentence_token_ids[:, first_different_token:], past_key_values=common_prefix_key_values)
            logits = torch.concat([common_prefix_output.logits.expand(len(sentences), -1, -1), rest_outputs.logits], dim=1)
            log_probs = logits.log_softmax(-1)
            log_probs = log_probs[:, :-1, :].gather(2, sentence_tokens.input_ids[:, 1:][:, :, None]).squeeze(-1)
            log_probs = (log_probs*sentence_tokens.attention_mask[:, 1:]).sum(-1).cpu()
        return log_probs
    def nli(self, sentences, unknown=False):
        # true_probs = self.sentence_probabilities(sentences + " True.")
        # false_probs = self.sentence_probabilities(sentences + " False.")
        # maybe_probs = self.sentence_probabilities(sentences + " Maybe.")
        if unknown:
            true_probs, maybe_probs, false_probs =  (self.sentence_probabilities([sentences + "(A)", sentences + "(B)", sentences + "(C)"]))
            return {'True': true_probs, 'Maybe': maybe_probs, 'False': false_probs}
        else:
            true_probs, false_probs =  (self.sentence_probabilities([sentences + "(A)", sentences + "(B)"]))
            return {'True': true_probs, 'False': false_probs}
    def yn(self, sentences, norm=True, relaxed=False, obvious=False, fewshot=None, maybe=False, tf = False):
        yns = []
        for sentence in sentences:
            if fewshot:
                sentence = fewshot + sentence
            
            if relaxed:
                yns.append(sentence + "Most likely")
                yns.append(sentence + "Not necessarily")
            elif obvious:
                yns.append(sentence + "obviously true.")
                yns.append(sentence + "not obviously true.")
            elif maybe:
                yns.append(sentence + "Yes")
                yns.append(sentence + "Maybe")
                yns.append(sentence + "No")
            elif tf:
                yns.append(sentence + '\\box{ True }')
                yns.append(sentence + '\\box{ False }')
            else:
                yns.append(sentence + "Yes")
                yns.append(sentence + "No")
        # if norm:
        #     norms = self.sentence_probabilities(sentences)
        probs = []
        batch_size = 256
        for i in range(0, len(yns), batch_size):
            if i+batch_size < len(yns):
                probs += list(self.sentence_probabilities(yns[i:i+batch_size]))
            else: 
                probs += list(self.sentence_probabilities(yns[i:]))
        probs=torch.tensor(probs)
        #   
        # probs = (self.sentence_probabilities(yns))
        # probs = torch.exp(probs)
        pyes = []
        pno = []
        pmaybe = []
        if maybe:
            z = 3
        else:
            z = 2
        for i in range(0,len(probs), z):
            # if yns[i] not in cache.keys():
                # yes, no = self.sentence_probabilities([yns[i], yns[i+1]])
            
            if maybe:
                
                yes, maybe, no = probs[i], probs[i+1], probs[i+2]
                
                      
            else:
                yes, no = probs[i], probs[i+1]
            if norm:
                if maybe: 
                    y,m,n = torch.tensor([yes, maybe, no]).softmax(-1)
                else:
                    y,n = torch.tensor([yes, no]).softmax(-1)
              
                # cache[yns[i]] = y
                # cache[yns[i+1]] = n
                pyes.append(y)
                pno.append(n)
                if maybe:
                    pmaybe.append(m)
            else:
                pyes.append(1-yes/(yes + no))
            # else:
            #     y, n = cache[yns[i]], cache[yns[i+1]]
            #     pyes.append(y)
                # pno.append(n)/
        # print('cache length', len(cache))
        # if maybe:
        # breakpoint()
        return torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)
    def complete(self, prompt, max_new = 20, temp = 0.1, topk=0):
        max_length = args.max_length
        encode_ids = self.tokenizer(
        prompt, 
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).input_ids.cuda()
        generated_outputs = self.model.generate(
        encode_ids, 
        max_new_tokens=max_new, 
        return_dict_in_generate=True, 
        output_scores=True,
        temperature=temp,
        top_k=topk
        )
        responses = self.tokenizer.batch_decode(
            generated_outputs.sequences,
            skip_special_tokens=True
        )
        return responses
        
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
def add_var(file):
    f = open(file, 'r')
    lines = f.readlines()
    writestr = ''
    for line in lines:
        if line.startswith('p cnf'):
            num_var, num_clause = [line.split(' ')[2], line.split(' ')[3]]
            writestr += 'p cnf ' + str(int(num_var) +1) + ' ' + str(int(num_clause) ) + '\n'
        else:
            writestr += line
    f.close()

    f = open(file, 'w')
    f.write(writestr)
    f.close()
def save_pattern(van, vap, vbn, vbp, vcn, vcp, patterns):
    mapping = {}
    if not isinstance(van, list):
        van = list(van)

    if not isinstance(vbn, list):
        vbn = list(vbn)

    if not isinstance(vcn, list):
        vcn = list(vcn)
    names = van + vbn + vcn
    i = 0
    for name in names:
        if name not in mapping.keys():
            # if i == 3:
                # breakpoint()
            mapping[name] = str(i)
            i += 1
        

    patterns.append(([mapping[name] for name in van], vap, [mapping[name] for name in vbn], vbp, [mapping[name] for name in vcn], vcp))
    return patterns


def check_pattern(van, vap, vbn, vbp, vcn, vcp, patterns):
    mapping = {}
    if not isinstance(van, list):
        van = list(van)

    if not isinstance(vbn, list):
        vbn = list(vbn)

    if not isinstance(vcn, list):
        vcn = list(vcn)

    names = van + vbn + vcn
    i = 0
    for name in names:
        if name not in mapping.keys():
            mapping[name] = str(i)
            i += 1

    pattern = ([mapping[name] for name in van], vap, [mapping[name] for name in vbn], vbp, [mapping[name] for name in vcn], vcp)

    for pat in patterns:
        if pat == pattern:
            return True
        
    return False

def search_pattern(van, vap, vbn, vbp, patterns):
    mapping = {}
    if not isinstance(van, list):
        van = list(van)

    if not isinstance(vbn, list):
        vbn = list(vbn)


    names = van + vbn
    unmapping = {}
    i = 0
    for name in names:
        if name not in mapping.keys():
            mapping[name] = str(i)
            unmapping[str(i)] = name
            i += 1
    pattern = ([mapping[name] for name in van], vap, [mapping[name] for name in vbn], vbp)

    for pat in patterns:
        ant = (pat[0], pat[1], pat[2], pat[3])
        if ant == pattern:
            # breakpoint()
            try:
                return ([unmapping[name] for name in pat[4]], pat[5])
            except:
                # breakpoint()
                continue
    return None

                
import math
def get_sol(file, lim=1000, del_sols=None):
    
    solutions = {'pos':  [], 'neg': []}
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for i in range(len(files)):
        file = files[i]
        shutil.copy(file, '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
        #   
        if not del_sols==None:
            if 'pos' in file:
                if 'neg' in file:
                    print('uh oh')
                      
                ds = del_sols['pos']
            elif 'neg' in file:
                ds = del_sols['neg']
            for sol in ds:
                add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
                cf = open('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]), 'a')
                write_str = '\n'
                for lit in sol:
                    write_str += str(-lit) + ' '
                # write_str += '0'
                cf.write(write_str)
                cf.close()
        count = 0
        while True:
            count += 1
            if count > lim:
                break
            os.system('/home/XXXX/XXXX/fs_backup_feb13/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]) + '> ' + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1])[:-4] + '.log')
            
            cf = open( '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1])[:-4] + '.log', 'r')

            lines = cf.readlines()

            el = lines[-1]
            # print(el)
            ec = el.split('exit ')[1].strip('\n')
            # lf.close()
            if ec == '20':
                break
            sl = lines[1:]
            while not sl[0].startswith('s '):
                sl = sl[1:]
            sl = sl[1:]
            solution = []
            while sl[0].startswith('v '):
                solution += list(map(int, sl[0].strip('\n').split(' ')[1:]))
                sl = sl[1:]
            if 'pos' in file:
                if 'neg' in file:
                    print('uh oh')
                      
                solutions['pos'].append(solution)
            elif 'neg' in file:
                solutions['neg'].append(solution)
            cf.close()
            #   
            add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
            cf = open('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]), 'a')
            write_str = '\n'
            for lit in solution:
                write_str += str(-lit) + ' '
            # write_str += '0'
            cf.write(write_str)
            cf.close()
            #   
        # solutions['pos'] = np.stack(solutions['pos'])
        # solutions['neg'] = np.stack(solutions['neg'])
    

    return solutions
    
def get_bb(file, del_sols=None):
    bb = {'pos':  [], 'neg': []}
    
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for i in range(len(files)):
        file = files[i]
        shutil.copy(file, '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
        if not del_sols==None:
            if 'pos' in file:
                if 'neg' in file:
                    print('uh oh')
                      
                ds = del_sols['pos']
            elif 'neg' in file:
                ds = del_sols['neg']
            for sol in ds:
                add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
                cf = open(f'/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]), 'a')
                write_str = '\n'
                for lit in sol:
                    write_str += str(-lit) + ' '
                # write_str += '0'
                cf.write(write_str)
                cf.close()
        # print('running cadical')
        os.system("timeout 5000 /home/XXXX/XXXX/fs_backup_feb13/LLM-project/cadiback/cadiback " + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]) + '> '  + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1])[:-4] + ".bbone")
        #   
        bbone= open('/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1])[:-4] + ".bbone", 'r')
        lines = bbone.readlines()
        #   
        for line in lines:
            if line.startswith('b'):
                #   
                lits = line.split(' ')[1:]
                for lit in lits:
                    lit = lit.strip()
                    if lit == '0':
                        continue
                    lit = int(lit)
                    if 'pos' in file:                                
                        if 'neg' in file:
                            print('uh oh')
                              
                        bb['pos'].append(lit)
                    elif 'neg' in file:
                            bb['neg'].append(lit)

    return bb
def next_var(bb, file, thresh=0.5 , dynamic=False, llm=None, lim=500, task='prontoqa', missed=False):
    all_probs = []
    all_trns = []
    cache = {}
    patterns = []
    trns_cache = {}
    missed_flag = None
    og = file
    #   
    vv = []
    pb = bb['pos']
    nb = bb['neg']
    sfx = ['cnf', 'mapping', 'maptxt']
    og_file = file
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    try: os.mkdir('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/workfiles_llmtresclutrr/')
    except: 1==0
    for f in files:
        for s in sfx:
            if 'pos' in f:
                shutil.copy(f[:-3] + s, '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/workfiles_llmtresclutrr/' + 'pos_tmp.' + s)
            else:
                shutil.copy(f[:-3] + s, '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/workfiles_llmtresclutrr/' + 'neg_tmp.' + s)
    file = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/workfiles_llmtresclutrr/' + 'tmp.cnf'
    
    
    em = '/'.join(file[:-4].split('/')[:-1]) + '/neg_' + file[:-4].split('/')[-1] + '.maptxt'

    maptxt = open(em, 'r').read()

          
    maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
    # print(maptxt)
    mapping = json.loads(maptxt)
    tick = 0
    set_vars = []
    set_pairs = []
    ordering = []
    names = {}
    passed = []
    failed = []
    # bb = get_bb(file)
    # nb = bb['neg']
    # pb = bb['pos']
    # jb = list(set(pb).intersection(set(nb)))

    # ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
    # for b in mapping.keys():
    #     b = int(b)
    #     phr = mapping[str(np.abs(b))]
        
                
    #     ppl = [phr.split('_')[-2], phr.split('_')[-3]]
    #     for name in ppl:
    #         if name not in names.keys():
    #             names[name] = 1
    #         else:
    #             names[name] += 1
    
    while True:
        # if not dynamic:
        #         names = {}
        #         # bb = get_bb(file)
        #         # nb = bb['neg']
        #         # pb = bb['pos']
        #         # jb = list(set(pb).intersection(set(nb)))

        #         # ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
        #         # for b in mapping.keys():
        #         #     b = int(b)
        #         #     phr = mapping[str(np.abs(b))]
                    
                            
        #         #     ppl = [phr.split('_')[-2], phr.split('_')[-3]]
        #         #     for name in ppl:
        #         #         if name not in names.keys():
        #         #             names[name] = 1
        #         #         else:
        #         #             names[name] += 1




        # uo = sorted(names, key=names.get)[::-1]
        # do = sorted(names, key=names.get)
        # uo = names
        # #   

        # # bb = get_bb(file)
        # # nb = bb['neg']
        # # pb = bb['pos']
        # jb = list(set(pb).intersection(set(nb)))

        # ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
        
        
           

        # empath = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs/'
        
        probs = torch.tensor([-10000, -100000])

        # while torch.sum(torch.where(probs > thresh, 1, 0)) == 0:
            
        p1 = -1
        p2 = -1
        breakflag=False
        good=False
        for p1 in range(len(mapping.keys())):
            # p1 += 1
            good=False
            for p2 in range(len(mapping.keys())):
                # p2 += 1

                if p1 == p2: continue
                if p1 == 0 or p2 == 0: continue
                # name1 = uo[p1]
                # name2 = do[p2]
                # if name1 == 'Richard' and name2 == 'Patricia':
                       
                # if name1 == name2:
                #     continue
                # if [name1, name2] in set_pairs or [name2, name1] in set_pairs:
                    # continue
                # name3 = None
                # n1var = 0
                # n2var = 0
                # for i in range(len(jb)):
                #     b = mapping[str(np.abs(jb[i]))]
                #     if name1 == b.split('_')[-3]:
                #         name3 = b.split('_')[-2]
                #     elif name1 == b.split('_')[-2]:
                #         name3 = b.split('_')[-3]
                #     else: continue
                #     if name2 == name3: continue
                #     for j in range(len(jb)):
                #         #   
                #         try:
                #             if name2 in mapping[str(np.abs(jb[j]))] and name3 in mapping[str(np.abs(jb[j]))]:
                #                 n1var = jb[i]
                #                 n2var = jb[j]
                #                 breakflag=True
                #                 break
                #         except:
                #             print(j)
                            
                #     if breakflag:break
                
                n1var = p1
                n2var = p2
                

        
                #   
                # print(nv)
                if n1var == 0:
                       
                    continue
                if (n1var, n2var) in passed: continue
                if (n1var, n2var) in failed: continue
                # print(name1, name2, name3)
                # print(b.split)
                # breakpoint()[]
                # print(b, mapping[str(np.abs(jb[j]))])
                # print(mapping[str(np.abs(n1var))], mapping[str(np.abs(n2var))])

                v1names = (mapping[str(np.abs(n1var))].split('_')[-3], mapping[str(np.abs(n1var))].split('_')[-2])
                v1rel = ' '.join(mapping[str(np.abs(n1var))].lower().split('_')[:-3])

                v2names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                v2rel = ' '.join(mapping[str(np.abs(n2var))].lower().split('_')[:-3])

                # question = "If " + v1names[1] + " is " + v1names[0] + '\'s '  + v1rel + " and " \
                #     + v2names[1] + " is " + v2names[0] + "\'s " + v2rel + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{"
                
                n_fs = 3
                
                fewshot = "True or False: If Frank is the son of James and Frank is the son of Martha then Martha is Jame\'s wife. Answer: \\box{ True }\n " + \
                    "True or False: If Joan is the daughter of Harry and Harry is the brother of Kevin then Kevin is Joan\'s nephew. Answer: \\box{ False }\n" + \
                    "True or False: If Robert is the father of Harriet and Samantha is the mother of Robert then Harriet is Samantha\'s granddaughter. Answer: \\box{ True }\n"


                question = "True or False: If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " then " + v2names[1] + " is " + v2names[0] + "\'s " + v2rel+ ". Answer: "
                # out = search_pattern(v1names, v1rel, v2names, v2rel, patterns)
                out = None
                if out != None:
                    rel = out[-1]
                    print('FOUND PATTERN')
                else:
                    # completion=llm.complete(fewshot+ question)[0]      
                    ent = llm.yn([fewshot + question], tf=True)[0]
                    # rel = '_'.join(completion.split('box{')[n_fs+1].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))

                    # breakpoint()
                # if rel == 'ister':
                #     breakpoint()
                # print(question)
                # print(ent)
                if ent < thresh: 
                    failed.append((n1var, n2var))
                    continue
                good=True
                passed.append((n1var, n2var))
                # nv_mapping = rel + '_' + name1 + '_' + name2 + '_'
                # patterns = save_pattern(v1names, v1rel, v2names, v2rel, [name1, name2], rel, patterns)
                # patterns = save_pattern(v2names, v2rel, v1names, v1rel, [name1, name2], rel, patterns)
                # if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                #     continue
                # nv = np.max(list(map(int, list(mapping.keys())))) + 1
                # if nv in set_vars:
                #     continue
                # newv = True
                # if nv_mapping in list(mapping.values()):
                #     for key, value in mapping.items():
                #         if value == nv_mapping:
                #             nv = int(key)
                #             newv = False
                # mapping[str(nv)] = nv_mapping
                probs = torch.tensor([10000, 100000])

                # questioni = "If " + name2 + ' is ' + name1 + '\'s ' \
                #     + rel + ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{'
                
                # questioni = "If " + name2 + ' is the '  + rel + " of " + name1 +  \
                #  ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{'
                
                # completioni = llm.complete(questioni)[0]

                # # print(completion)
                # reli = '_'.join(completioni.split('box{')[1].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip(' ').strip(' ').lower().split(' '))
                # patterns = save_pattern(v1names, v1rel, v2names, v2rel, [name2, name1], reli, patterns)
                # patterns = save_pattern(v2names, v2rel, v1names, v1rel, [name2, name1], reli, patterns)
                
                # nv_mappingi = reli + '_' + name2 + '_' + name1 + '_'
                # nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                # newvi=True
                # if nv_mappingi in list(mapping.values()):
                #     for key, value in mapping.items():
                #         if value == nv_mappingi:
                #             nvi = int(key)
                #             newvi=False
                # mapping[str(nvi)] = nv_mappingi
                # probs = torch.tensor([10000, 100000])


                

                #   
                #   
                
                # print(nv)
                # print(probs)
                #   
                # print(mapping[str(np.abs(nv))])
                #   
                # nv = 'hello'
                # try:
                #     vv += (nv, question, mapping[str(np.abs(int(nv)))])
                # except:
                #     vv += (nv, '**COULD NOT FIND A COMMON-SENSE RULE')
                #     print("**COULD NOT FIND A COMMON-SENSE RULE")
                #     #   
                #     break
                #     #   
                # print(vv)
                # # breakpoint()
                # good=True
                # if dynamic:
                #     names[name2]=np.max(list(names.values()))+1
                break
            if good:
                break
            # #   
            # if len(vv) > lim*2:
            #     vv += ["ERROR: TIME OUT"]
            #     print("ERROR TIME OUT")
            #     break
        if not good:
            # breakpoint()
            missed_flag=True
            new_sols = {'pos': [1], 'neg':[]}
            print("missed")
            # breakpoint()
            break
        tmpfiles = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
        for f in tmpfiles:
            
            add_clause(f)
            # add_clause(f)
            cf = open(f, 'a')
            
            cf.write('\n' + str(-n1var) + ' ' + str(n2var) +  ' 0')
            
            # cf.write('\n' + str(nvi) + ' 0')
            cf.close()
        di = []
        
        new_sols = get_sol(file, lim=100)
        
        # set_vars.append(nv)
        # set_vars.append(nvi)
        # set_pairs.append([name1, name2])
        #   
        #   
        if len(new_sols['pos']) == 0 or len(new_sols['neg']) == 0:
            break
    # print('done')
    #   
    return vv, new_sols, bb, missed_flag
if __name__ == '__main__':
    # noisy_data = ['clutrr33.cnf']
    noisy_data=[]
    # mistr_data = ['clutrr10.cnf']
    mistr_data = []
    c = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_new_csvs/solver_finished.csv'
    import csv
    import json
    dataset = '/home/XXXX/XXXX/fs_backup_feb13/SAT-LM/data/clutrr_test.json'
    with open(dataset, 'r') as df:
        data = json.loads(df.read())
    
    task = 'clutrr'
    missed=False
    c = open(c, 'r')
    cr = csv.reader(c)
    names = []
    all_outs = {}
    missed_list = []
    labels = {}
    for row in cr:
        if row[2] == 'SAT' and row[3] == 'SAT':
            cnf = open('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_new/neg_'+row[1]).readlines()[0].strip('\n')
            num_clause = int(cnf.split(' ')[-1])
            if row[1] in noisy_data or row[1] in mistr_data:
                continue
            # if num_clause > 500:
                # continue
            names.append(row[1])
            labels[row[1]] = data[int(row[1].split('clutrr')[1].split('.')[0])]['label']
    #   
    preds = {}
    if task == 'clutrr':
        c = open('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/clutrr_new_labels.csv', 'r')
        cr = csv.reader(c)
        for row in cr:
            cnf = open('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_new/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
            num_clause = int(cnf.split(' ')[-1])
            if row[1] in noisy_data or row[1] in mistr_data:
                continue
            # if num_clause > 500:
                # continue
            labels[row[0][:-2]+'cnf'] = row[1].lower()
    #   
    from tqdm import tqdm
    #   
    llm = LLM()
    #   
    # names = ['clutrr59.cnf']
    uhohs = []
    times = {}
    acc = 0
    counter = 0
    skip_pbar = True
    for name in (pbar := tqdm(names)):
    # name
        if not skip_pbar:
            pbar.set_description('Acc: ' + str(acc / counter) + '. Missed: ' + str(missed/counter) + ', ' + name)
        skip_pbar=False
        start_time = time.time()
        print(name)
        p = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/dimacs_clutrr_new/' + name
        # p = '/home/XXXX/XXXX/fs_backup_feb13/LLM-project/tempfiles/dimacs_test.cnf'
        # sols = get_sol(p, lim=100)
        #   
        #   
        bb = get_bb(p)

        prep_time = time.time() - start_time
        # sols = np.load(open("/home/XXXX/XXXX/fs_backup_feb13/LLM-project/tempfiles/sols.np.npy", 'rb'), allow_pickle=True)
        # bb = np.load(open("/home/XXXX/XXXX/fs_backup_feb13/LLM-project/tempfiles/bb.np.npy", 'rb'), allow_pickle=True)
        # solutions = {}
        # solutions['pos'] = sols.item().get('pos')
        # solutions['neg'] = sols.item().get('neg')

        # bones = {}
        # bones['pos'] = bb.item().get('pos')
        # bones['neg'] = bb.item().get('neg')
        
        
        # print(sols)
        # print(bb)
        # print(end_time - start_time)

        
        #   
        
        vv, solout, bbout, missed_flag = next_var(bb, p, llm=llm, task=task, missed=missed)

        end_time = time.time() - start_time - prep_time

        times[name] = {'prep_time': prep_time, 'run_time': end_time}
        #   
        # print('finished!')
        #   
        all_outs[name] = (vv, solout, bbout, missed_flag)
        if not missed_flag == None:
            missed_list.append([name, vv])
            
            
        if ((solout == None and vv == None and bbout == None) or not missed_flag==None) and missed:
            preds[name] = 'missed'            
            missed += 1
        elif len(solout['pos'])==0 and len(solout['neg']) > 0:
            if task == 'clutrr':
                preds[name] = 'false'
            else:
                preds[name] = 'true'
        
            # if preds[name] != labels[name]:
            #       
        elif len(solout['pos'])>0 and len(solout['neg']) == 0:
            if task == 'clutrr':
                preds[name] = 'true'
            else:
                preds[name] = 'false'
            # if preds[name] != labels[name]:
            #       
        else:
            print('uh oh')
            uhohs.append(name)
               
            #   
        
        print('label:', labels[name])
        print('pred:', preds[name])
        if preds[name] == labels[name].lower().strip(' '):
            acc += 1
            counter += 1
        elif preds[name] == 'missed':
            missed += 1
            counter+= 1
        else:
            counter += 1
        
        

    acc = 0
    missed = 0
    for key, value in preds.items():
        if preds[key] == labels[key].lower().strip(' '):
            acc += 1
        elif preds[key] == 'missed':
            missed += 1
        print(preds[key], labels[key])
    print('acc:', acc / len(preds))
    print('missed:', missed / len(preds))
    print('missed list:', len(missed_list))

    # for miss in missed_list:
    #     print(data[int(miss.split('clutrr')[1].split('.cnf')[0])]['missing'])
    print('llm-tres clutrr 8B')
    breakpoint()