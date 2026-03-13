import os
import shutil
import numpy as np
import time
from transformers.generation.utils import GenerationConfig

import json

import torch, os
from torch.utils.data import DataLoader
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
USER_PATH = '/home/XXXX/XXXX/fs_backup_feb13/'

os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
# os.environ['TRANSFORMERS_CACHE'] = USER_PATH + '/.cache/huggingface/hub'
cache_dir = '/media/data1/XXXX/hub/'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
# os.environ['HF_HUB_OFFLINE'] ='1'


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
        'n_rows': 20, 'max_length': 300,'temperature': 1, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
# args['engine'] = 'meta-llama/Meta-Llama-3-8B-Instruct'
# args['engine'] = 'mistralai/Mistral-7B-Instruct-v0.3'
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
                    cache_dir = cache_dir,
                    token = os.getenv("HF_TOKEN"),
                    # attn_implementation="flash_attention_2"

                    )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id            
            self.model = AutoModelForCausalLM.from_pretrained(
                    args.engine, 
                    cache_dir = cache_dir,
                    quantization_config=quant_config,
                    device_map='auto',
                    token = os.getenv("HF_TOKEN"),
                    # attn_implementation="flash_attention_2"
                    )

        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def sentence_probabilities(self, sentences):
        with torch.no_grad():
            sentence_tokens = self.tokenizer(sentences, return_tensors='pt', padding=True)
            sentence_token_ids = sentence_tokens.input_ids.cuda()

            # Little hack to cut down inference time by 4-5x (leads to some imprecisions when using quantization)
            # Find the common prefix and run it through the model once, to save time
            first_different_token = (sentence_token_ids == sentence_token_ids[0, :].unsqueeze(0)).all(dim=0).long().argmin()
            common_prefix = sentence_token_ids[0, :first_different_token].unsqueeze(0)
            common_prefix_output = self.model(common_prefix, use_cache=True)
            common_prefix_key_values = tuple(tuple(tensor.expand(len(sentences), -1, -1, -1) for tensor in layer) 
                                             for layer in common_prefix_output.past_key_values)

            # Process the rest of the sentences
            rest_outputs = self.model(sentence_token_ids[:, first_different_token:], past_key_values=common_prefix_key_values)
            logits = torch.concat([common_prefix_output.logits.expand(len(sentences), -1, -1), rest_outputs.logits], dim=1).cuda()
            log_probs = logits.log_softmax(-1)
            log_probs = log_probs[:, :-1, :].gather(2, sentence_token_ids[:, 1:][:, :, None]).squeeze(-1).cuda()
            log_probs = (log_probs*sentence_tokens.attention_mask.cuda()[:, 1:]).sum(-1).cpu()
        return log_probs
    def nli(self, sentences, unknown):
        # true_probs = self.sentence_probabilities(sentences + " True.")
        # false_probs = self.sentence_probabilities(sentences + " False.")
        # maybe_probs = self.sentence_probabilities(sentences + " Maybe.")
        if unknown:
            true_probs, maybe_probs, false_probs =  (self.sentence_probabilities([sentences + "(A)", sentences + "(B)", sentences + "(C)"]))
            return {'True': true_probs, 'Maybe': maybe_probs, 'False': false_probs}
        else:
            true_probs, false_probs =  (self.sentence_probabilities([sentences + "True", sentences + "False"]).softmax(-1))
            return {'True': true_probs, 'False': false_probs}
    def yn(self, sentences, norm=True, relaxed=False, obvious=False, fewshot=None, maybe=False):
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
        
        if maybe: return torch.stack([torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)])
        return torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)
    def complete(self, prompt, max_new = 25, temp = 1 , topk=0):
        max_length = args.max_length
        encode_ids = self.tokenizer(
        prompt, 
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1000
    ).input_ids.cuda()
        gc = GenerationConfig(temperature=1, do_sample=True, topk=0, max_length=1000, max_new_tokens=300, return_dict_in_generate=True, output_scores=True)
        generated_outputs = self.model.generate(
        encode_ids, 
        generation_config = gc
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
                continue
    return None

                
import math
def get_sol(file, lim=10000, del_sols=None, seedrun=0):
    
    solutions = {'pos':  [], 'neg': []}
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for i in range(len(files)):
        file = files[i]
        shutil.copy(file, '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]))
        #   
        if not del_sols==None:
            if 'pos' in file:
                if 'neg' in file:
                    print('l. 343 uh oh')
                      
                ds = del_sols['pos']
            elif 'neg' in file:
                ds = del_sols['neg']
            for sol in ds:
                add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/'+ str(file.split('/')[-1]))
                cf = open('/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/'+ str(file.split('/')[-1]), 'a')
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
            os.system(USER_PATH + '/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]) + '> ' + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1])[:-4] + '.log')
            
            cf = open( '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1])[:-4] + '.log', 'r')

            lines = cf.readlines()

            el = lines[-1]
            # print(el)
            try:
                ec = el.split('exit ')[1].strip('\n')
            except:
                breakpoint()
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
                    print('l. 384 uh oh')
                      
                solutions['pos'].append(solution)
            elif 'neg' in file:
                solutions['neg'].append(solution)
            cf.close()
            #   
            add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]))
            cf = open('/'.join(file.split('/')[:-2]) +'/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]), 'a')
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
    
def get_bb(file, del_sols=None, seedrun=0):
    bb = {'pos':  [], 'neg': []}
    
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for i in range(len(files)):
        file = files[i]
        # print(file, '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]))
        shutil.copy(file, '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]))
        if not del_sols==None:
            if 'pos' in file:
                if 'neg' in file:
                    print('l. 416 uh oh')
                      
                ds = del_sols['pos']
            elif 'neg' in file:
                ds = del_sols['neg']
            for sol in ds:
                add_clause('/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/'+ str(file.split('/')[-1]))
                cf = open(f'/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]), 'a')
                write_str = '\n'
                for lit in sol:
                    write_str += str(-lit) + ' '
                # write_str += '0'
                cf.write(write_str)
                cf.close()
        # print('running cadical')
        os.system("timeout 5000 " + USER_PATH + "/LLM-project/cadiback/cadiback " + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]) + '> '  + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/'+ str(file.split('/')[-1])[:-4] + ".bbone")
        #   
        bbone= open('/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1])[:-4] + ".bbone", 'r')
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
                            print('l. 447 uh oh')
                              
                        bb['pos'].append(lit)
                    elif 'neg' in file:
                            bb['neg'].append(lit)

    return bb


def cot(prob, n=5):
    # prob = folio[name]
    # few_shot = "Here are some facts and rules: \nThere are six types of wild turkeys: Eastern wild turkey, Osceola wild turkey, Gould\’s wild turkey, Merriam\’s wild turkey, Rio Grande wild turkey, and Ocellated wild turkey.\nTom is not an Eastern wild turkey.\nTom is not an Osceola wild turkey.\nTom is not a Gould's wild turkey.\nTom is neither a Merriam's wild turkey nor a Rio Grande wild turkey.\nTom is a wild turkey.\nHere are some additional facts and rules we\'ve found, written in logical form:\n[EMPTY]\n\nQuestion: Is the following statement True: \nTom is an eastern wild turkey.\nAnswer: Let\'s think step by step.\n1. Since Tom is a wild turkey, Tom is either an Eastern, Osceola, Gould\'s, Merriam\'s, Rio Grande or Ocellated wild turkey.\n2. Since Tom is not an Eastern, Osceola, Gould\'s, Merriam\'s or Rio Grande wild turkey, Tom is not an Eastern wild turkey.\nTherefore, the answer to the question is False.\n" + \
    # '\Here are some facts and rules: \nAll squares are four-sided.\nHere are some additional facts and rules we\'ve found, written in logical form: \nfour_sided(thing) implies shape(thing).\n\nQuestion: Is the following statement True: \nAll squares are shapes.\nAnswer: Let\'s think step by step.\n1. We know that all squares are four-sided.\n2. We know that all four-sided things are shapes.\n3. Since all squares are four-sided, and all four-sided things are shapes, all squares are shapes.\nTherefore, all squares are shapes.\nTherefore, the answer to the question is True. ' + \
    # "\Here are some facts and rules: \nThe state of Montana includes the cities of Butte, Helena, and Missoula.\nWhite Sulphur Springs and Butte are cities in the same state in U.S.\nA city can only be in one state in U.S. except for Bristol, Texarkana, Texhoma and Union City.\nHere are some additional facts and rules we\'ve found, written in logical form:\nNOT in_Montana(St_Pierre).\nin_Butte(city) implies NOT in_St_Pierre(city).\nin_Montana(Billings).\n\nQuestion: Is the following statement True: \nMontana is home to the city of Missoula. Answer: Let\'s think step by step. 1. We know that the state of Montana includes the cities Butte, Helena, and Missoula.\nTherefore, Missoula is in Montana.\nTherefore, the answer to the question is True.\n"
    n_fewshot = 3
    few_shot = "Prob no. 1)\nHere are some facts and rules: \nThere are six types of wild turkeys: Eastern wild turkey, Osceola wild turkey, Gould\’s wild turkey, Merriam\’s wild turkey, Rio Grande wild turkey, and Ocellated wild turkey.\nTom is not an Eastern wild turkey.\nTom is not an Osceola wild turkey.\nTom is not a Gould's wild turkey.\nTom is neither a Merriam's wild turkey nor a Rio Grande wild turkey.\nTom is a wild turkey.\nHere are some additional facts and rules we\'ve found, written in logical form:\n[EMPTY]\n\nInstruction: Determine if following statement is true:\nTom is an eastern wild turkey.\nAnswer: Let\'s think step by step.\n1. Since Tom is a wild turkey, Tom is either an Eastern, Osceola, Gould\'s, Merriam\'s, Rio Grande or Ocellated wild turkey.\n2. Since Tom is not an Eastern, Osceola, Gould\'s, Merriam\'s or Rio Grande wild turkey, Tom is not an Eastern wild turkey.\nTherefore, the answer to the question is False.\n" + \
    '"Prob no. 2)\nHere are some facts and rules: \nAll squares are four-sided.\nHere are some additional facts and rules we\'ve found, written in logical form: \nfour_sided(thing) implies shape(thing).\n\nInstruction: Determine if following statement is true:\nAll squares are shapes.\nAnswer: Let\'s think step by step.\n1. We know that all squares are four-sided.\n2. We know that all four-sided things are shapes.\n3. Since all squares are four-sided, and all four-sided things are shapes, all squares are shapes.\nTherefore, all squares are shapes.\nTherefore, the answer to the question is True. ' + \
    "Prob no. 3)\nHere are some facts and rules: \nThe state of Montana includes the cities of Butte, Helena, and Missoula.\nWhite Sulphur Springs and Butte are cities in the same state in U.S.\nA city can only be in one state in U.S. except for Bristol, Texarkana, Texhoma and Union City.\nHere are some additional facts and rules we\'ve found, written in logical form:\nNOT in_Montana(St_Pierre).\nin_Butte(city) implies NOT in_St_Pierre(city).\nin_Montana(Billings).\n\nInstruction: Determine if following statement is true:\nMontana is home to the city of Missoula. Answer: Let\'s think step by step. 1. We know that the state of Montana includes the cities Butte, Helena, and Missoula.\nTherefore, Missoula is in Montana.\nTherefore, the answer to the question is True.\n" + \
    "Prob no. 4)\n"
    # few_shot = 'Here are some facts and rules: \n Jack is a rabbit. \nAll rabbits are mamals. \nAll furry creatures are warm-blooded. \nHere are some additional facts and rules we\'ve found: \nAll mammals are furry.\n\nInstruction: Determine if following statement is true:\nJack is warm-blooded.\nAnswer: Let\'s think step by step.\n1.If Jack is a rabbit then Jack is a mammal. \n2.If Jack is a mammal then Jack is furry.\n3.If Jack is furry then Jack is warm-blooded.\nTherefore, the answer to the question is True.' + \
    # 'Here are some facts and rules: \n _18 is a natural number. \nAll naturals are integers. \nAll integers are reals. \nHere are some additional facts and rules we\'ve found: \nAll real numbers are complex numbers.\n\nInstruction: Determine if following statement is true:\n _18 is not a complex number\nAnswer: Let\'s think step by step.\n1.If _18 is natural then it is an integer. \n2.If _18 is an integer then it is a real number.\n3.If _18 is a real number then it is a complx number.\nTherefore, the answer to the question is False.'

    # if 'newrules' in prob.keys():

    #     prompt = few_shot +  'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found, written in logical form: \n' + '\n'.join(prob['newrules']) +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
    # else:
    #     prompt = few_shot + 'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found, written in logical form: \n' + '[EMPTY]' +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
    if 'newrules' in prob.keys():
        if len(prob['newrules']) != 0:
            prompt = few_shot +  'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found: \n' + '\n'.join(prob['newrules']) +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
        else:
            prompt = few_shot + 'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found: \n' + '[EMPTY]' +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
    
    else:
        prompt = few_shot + 'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found: \n' + '[EMPTY]' +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
    
    
    votes = torch.tensor([0.0,0.0])
    
    for i in range(n):

        # ans = llm.complete(prompt, max_new=300)[0]
        # ans = 'Here are some facts and rules:'.join(ans.split('Here are some facts and rules:')[:5])
        # if len(ans.split('Facts')) > 7:
        #     ans = 'Facts'.join(ans.split('Facts')[:5])
        # ans_prompt = ans + "Therefore, the answer (True/False) is "
        # nli = torch.tensor(list(llm.nli(ans_prompt, False).values()))    
        # votes[nli.argmax()] += 1

        ans = llm.complete(prompt, max_new=150)[0]

        ans = ans.split('Prob no. 5)')[0]
        # ans = 'Here are some facts and rules:'.join(ans.split('Here are some facts and rules:')[:5])
        # if len(ans.split('Facts')) > 7:
        #     ans = 'Facts'.join(ans.split('Facts')[:5])
        # ans_prompt = ans + "Therefore, the final answer (Yes/No) is: "
        # yn = llm.yn([ans_prompt]).values
        # nli = torch.tensor(yn[0], yn[2])
        # ans = 'Facts:'.join(ans.split('Facts:')[:n_fewshot + 2])

        ans_prompt = 'Here are some facts and rules:' + ans.split('Here are some facts and rules:')[n_fewshot+1] + "Therefore, the answer (True/False) is "
        # nli = torch.tensor(list(llm.nli(ans_prompt, False).values()))
        # votes[nli.argmax()] += 1.0
        # print(nli)
        # print(ans_prompt)
        # continue  
        # answer = answers[torch.argmax(nli)]
        # breakpoint()

        # ans = ans.split('***')[n_fewshot]

        # try: lines = ans.split('\n')
        # except: breakpoint()
        # i = -1
        notherefore=False
        z = ans_prompt.split('Therefore')[-2].split('\n')[0]
        if 'True' in z: 
            votes[0] += 1.0
            print(ans_prompt)
            print('[1,0]')

            continue

        elif 'False' in z: 
            votes[1] += 1.0
            print(ans_prompt)
            print('[0,1]')

            continue
        else: 
            nli = torch.tensor(list(llm.nli(ans_prompt, False).values()))
            votes += nli.softmax(-1)
            # votes[nli.softmax(-1).argmax()] += 1
            # votes +
            print(ans_prompt)
            print(nli.softmax(-1))

            continue  
    
    # votes = votes
    # print(ans.split(few_shot)[1])

    # answer = answers[torch.argmax(nli)]f
    return votes
def return_exceeded(pb, nb, mapping, llm):
    p_sum = 0
    n_sum = 0
    p_record = {}
    n_record = {}
    for p in pb:
        # try:
        #     txt = mapping[str(np.abs(p))].split('__')
        # except:
        #     breakpoint()[]
        txt = mapping[str(np.abs(p))].split('__')[:-1]

        txt = [t.replace('_', ' ') for t in txt]
        prompt = 'Is the following plausible? Answer yes or no, disregarding grammar and syntax: '
        if p > 0:
            if len(txt) == 2:
                prompt += txt[1] + ' is ' + txt[0]
            elif len(txt) == 3:
                prompt += txt[2] + ' ' + txt[1] + ' ' + txt[0]
        elif p < 0:
            if len(txt) == 2:
                prompt += txt[1] + ' is not ' + txt[0]
            elif len(txt) == 3:
                prompt += txt[2] + ' does not ' + txt[1] + ' ' + txt[0]
        prompt = prompt + '. Answer: '
        # p_sum +=  llm.yn([prompt])[0]
        p_record[prompt] = llm.yn([prompt])[0]
        p_sum += p_record[prompt] - 0.5
    for n in nb:
        txt = mapping[str(np.abs(n))].split('__')[:-1]
        txt = [t.replace('_', ' ') for t in txt]
        prompt = 'Is the following plausible? Answer yes or no, disregarding grammar and syntax: '
        if n > 0:
            if len(txt) == 2:
                prompt += txt[1] + ' is ' + txt[0]
            elif len(txt) == 3:
                prompt += txt[2] + ' ' + txt[1] + ' ' + txt[0]
        elif n < 0:
            if len(txt) == 2:
                prompt += txt[1] + ' is not ' + txt[0]
            elif len(txt) == 3:
                prompt += txt[2] + ' does not ' + txt[1] + ' ' + txt[0]
        prompt = prompt + '. Answer: '
        # n_sum +=  llm.yn([prompt])[0]
        n_record[prompt] = llm.yn([prompt])[0]
        n_sum += n_record[prompt] - 0.5
    print(n_record)
    print(p_record)
    print(n_sum, p_sum)
    # breakpoint()
    if n_sum > p_sum: return {'pos': [], 'neg': [0]}
    else: return {'pos': [0], 'neg': []}
    
def rule_check(rule, contra_thresh=0, context_thresh=0, prob='', question = '',llm=None, ablate=False):
    if ablate: 
        print('ablating rulecheck')
        return True, [0,0]
    contra_fs = 'Does the following rule seem contradictory?\nRule: blue(Allen) and happy(Allen) implies orange(Allen).\nAnswer: \\box{ Yes } this seems contradictory. If Allen is blue he is not orange.' \
        + '\nDoes the following rule seem contradictory?\nRule: NOT meet(Frank, Sam) implies strangers(Frank, Sam).\nAnswer: \\box{ No } this does not seem contradictory. If Frank and Sam have not met, they are strangers.\n'
    a = 1 - llm.yn([contra_fs + 'Does the following rule seem contradictory?\nRule: ' + rule + '\nAnswer: \\box{ '])[0]
    # a = llm.yn(['Does the following rule seem plausible?\nRule: ' + rule + '\nAnswer:'])
    # breakpoint()
    # print(r)
    b = llm.yn(['Here are some facts and rules\n' + '\n'.join(prob['context']) + '\nDoes the following new rule seem contextually relevant to the facts and rules?\nRule: ' + rule + '\nAnswer \"Yes\" or \"No\" here: '])[0]

    if a < contra_thresh:
        # breakpoint()
        print('rejected by contradiction:', rule, a)
        return False, [a,b]
    if b < context_thresh:
        # breakpoint()
        print('rejected by context:', rule, b)
        return False, [a,b]
    return True, [a,b]
def next_var(bb, file, thresh=2.0, dynamic=False, llm=None, lim=100, task='prontoqa', missed=False, prob='', question='', seedrun = 0, fixed_iter=6, cot_thresh=0.8):
    all_probs = []
    all_trns = []
    cache = {}
    patterns = []
    trns_cache = {}
    rule_scores = {}
    sc_scores = []
    ps = torch.tensor([0.0,0.0])
    missed_flag = None
    # quick_fs = 'Use your commonsense. Is the following true:\n\"red(alex)\"\nAnswer: \\box{No}, there is no reason to think Alex is red.\nUse your commonsense. Is the follwoing true:\n\"alien(martian)\"\nAnswer: \\box{Yes}, martians are aliens\n'
    quick_fs = 'Use your commonsense. Is the following true:\n\"alex is red\"\nAnswer: \\box{Maybe}, there is no way to know if Alex is red.\nUse your commonsense. Is the following true:\n\"alien is martian\"\nAnswer: \\box{Yes}, martians are aliens.Use your commonsense. Is the following true:\n\"james is girl\"\nAnswer: \\box{No}, Alex is typically a man\'s name.\n'
    # quick = llm.yn([quick_fs + 'Use your commonsense. Is the following true:\n' + question + '\nAnswer: \\box{'], maybe=True)[0]
    # print( 'Use your commonsense. Is the following true:\n' + question + '\nAnswer: \\box{', quick)
    # quickthresh=0.9
    # if quick > quickthresh: 
    #     print('quicksolving')
    #     return ['quicksolve'], {'pos':[], 'neg':[0]}, bb, None, {}
    # elif 1-quick > quickthresh: 
    #     print('quicksolving')
    #     return ['q'], {'pos':[0], 'neg':[]}, bb, None, {}
    og = file
    #   
    vv = []
    pb = bb['pos']
    nb = bb['neg']
    jb = list(set(pb).intersection(set(nb)))
    
    sfx = ['cnf', 'mapping', 'maptxt', 'arity']
    og_file = file
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for f in files:
        for s in sfx:
            if 'pos' in f:
                shutil.copy(f[:-3] + s, USER_PATH + '/LLM-project/workfiles' + str(seedrun) + '/' + 'pos_tmp.' + s)
            else:
                shutil.copy(f[:-3] + s, USER_PATH + '/LLM-project/workfiles' + str(seedrun) + '/' + 'neg_tmp.' + s)
    file = USER_PATH + '/LLM-project/workfiles' + str(seedrun) + '/' + 'tmp.cnf'
    
    
    
    em = '/'.join(file[:-4].split('/')[:-1]) + '/neg_' + file[:-4].split('/')[-1] + '.maptxt'

    maptxt = open(em, 'r').read()
    arity = '/'.join(file[:-4].split('/')[:-1]) + '/neg_' + file[:-4].split('/')[-1] + '.arity'
    arity1 = np.load(open(arity, 'rb'), allow_pickle=True).item()
    arity = {}
    for key, value in arity1.items():
        arity[key.lower()] = value
    print(arity)

        
    maptxt = maptxt.replace(" ", " \"").replace(",", "\",").replace(":", "\":").replace("{", "{\"").replace("}", "\"}")
    # print(maptxt)

    mapping = json.loads(maptxt)

    prob['newrules'] = []
    # for j in jb:
    #     split = mapping[str(np.abs(j))].split('__')
    #     if len(split) == 3:
    #         varname = split[0] + '(' + split[1] + ')'
    #     elif len(split) == 4:
    #         varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
    #     if j < 0:
    #         varname = 'NOT ' + varname
    #     prob['newrules'].append(varname)
        
    set_vars = []
    new_sols=None
    for key, value in mapping.items():
        if int(key) in np.abs(jb):
            continue
        splt = value.split('__')
        if len(splt) == 3:
            rule =  splt[0] + '(' + splt[1] + ')'
            quick = llm.yn([quick_fs + 'Use your commonsense. Is the following true:\n'  + splt[1] + ' is ' + splt[0] + '\nAnswer: \\box{ '], maybe=True)
            print('Use your commonsense. Is the following true:\n'  + splt[1] + ' is ' + splt[0] + '\nAnswer: \\box{ ', quick)
        elif len(splt) == 4:
            continue
            quick = llm.yn([quick_fs+'Use your commonsense. Is the following true:\n' + splt[0] + '(' + splt[1] + ', ' + splt[2] + ')' + '\nAnswer: \\box{ '], maybe=True)
            print('Use your commonsense. Is the following true:\n' + splt[0] + '(' + splt[1] + ', ' + splt[2] + ')' + '\nAnswer: \\{ ', quick)

        # print(value, quick)
        if quick[0] > thresh:
            tmpfiles = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
            for f in tmpfiles:
                # breakpoint()
                add_clause(f)
                cf =open(f, 'a')

                cf.write('\n' + str(key) + ' 0')
            
                cf.close()
                # breakpoint()
            print('quicksolved ' + value + ' is true')
            vv += ['quicksolved ' + value + ' is true'] 
            start = time.time()
            try:
                new_sols = get_sol(file, lim=1, seedrun=seedrun )
                b = get_bb(file, seedrun=seedrun)
            except:
                breakpoint()
            print('l. 733: took ' , time.time() - start , ' seconds to get bb and solution')

            nb = bb['neg']
            pb = bb['pos']
            set_vars.append(int(key))
            if len(new_sols['pos']) == 0 and len(new_sols['neg']) == 0:
                # breakpoint()
                ps += cot(prob)
                sc_scores.append(ps.clone())
                print('cot:', ps)
                # if torch.max(ps) >= (cot_thresh)*ps.sum():
            # return ['True', 'False'][torch.argmax(ps)]
                answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                print('quicksolve and cot')
                # sc_scores.append(ps.clone())
                return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
            if len(new_sols['pos']) == 0 or len(new_sols['neg']) == 0:
                return vv, new_sols, bb, False, rule_scores, False, sc_scores
            if 'newrules' not in prob.keys():
                prob['newrules'] = [rule]
            else:
                prob['newrules'].append(rule)
            # ps += cot(prob)
            # sc_scores.append(ps.clone())
            # answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            # #  ps += cot(prob)
            # print('cot: ', ps)
            # if torch.max(ps) >= (cot_thresh)*ps.sum():
            # # return ['True', 'False'][torch.argmax(ps)]
            #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            #     print('quicksolve and cot')
            #     # sc_scores.append(ps.clone())
            #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

            # if len(prob['newrules']) >= fixed_iter:
            #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
        elif quick[2] > thresh:
            tmpfiles = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
            for f in tmpfiles:
                # breakpoint()
                add_clause(f)
                cf = open(f, 'a')

                cf.write('\n' + str('-' + key) + ' 0')
            
                cf.close()
                # breakpoint()
            print('quicksolved ' + value + ' is false')
            vv.append(['quicksolved ' + value + ' is false'])
            start = time.time()
            try:
                new_sols = get_sol(file, lim=1, seedrun=seedrun )
                b = get_bb(file, seedrun=seedrun)
            except: breakpoint()
            print('l. 778 took ', time.time() - start , ' seconds to get bb and solution')
            nb = bb['neg']
            pb = bb['pos']
            set_vars.append(int(key))
            if len(new_sols['pos']) == 0 and len(new_sols['neg']) == 0:
                # breakpoint()
                ps += cot(prob)
                sc_scores.append(ps.clone())
                print('cot:', ps)
                # if torch.max(ps) >= (cot_thresh)*ps.sum():
            # return ['True', 'False'][torch.argmax(ps)]
                answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                print('quicksolve and cot')
                # sc_scores.append(ps.clone())
                return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
            if len(new_sols['pos']) == 0 or len(new_sols['neg']) == 0:
                # breakpoint()
                # print('')
                return vv, new_sols, bb, False, rule_scores, False, sc_scores
            if 'newrules' not in prob.keys():
                prob['newrules'] = ['NOT ' + rule]
            else:
                prob['newrules'].append('NOT ' + rule)
            # ps += cot(prob)
            # sc_scores.append(ps.clone())
            # answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            #  ps += cot(prob)
            # print('cot: ', ps)
            # if torch.max(ps) >= (cot_thresh)*ps.sum():
            # # return ['True', 'False'][torch.argmax(ps)]
            #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            #     print('quicksolve and cot')
            #     # sc_scores.append(ps.clone())
            #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

            # if len(prob['newrules']) >= fixed_iter:
            #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
    # if new_sols != None:
    ps += cot(prob)
    sc_scores.append(ps.clone())
    print('cot:', ps)
    if torch.max(ps) >= (cot_thresh)*ps.sum() or ps.sum() == 20:
            # return ['True', 'False'][torch.argmax(ps)]
                answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                print('quicksolve and cot')
                # sc_scores.append(ps.clone())
                return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores


    new_sols = None

    tick = 0
    set_pairs = []
    ordering = []
    names = {}
    bb = get_bb(file, seedrun=seedrun)
    nb = bb['neg']
    pb = bb['pos']
    jb = list(set(pb).intersection(set(nb)))
    calls = 0
    ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
    for b in ab:
    
        phr = mapping[str(np.abs(b))]
        
        pred = phr.split('__')[0].lower()
        # breakpoint()
        try:
            if arity[pred] == 1:
                ppl = [phr.split('__')[1]]
            elif arity[pred] == 2:        
                ppl = [phr.split('__')[1], phr.split('__')[2]]
        except:
            continue
        for name in ppl:
            if name not in names.keys():
                names[name] = 1
            else:
                names[name] += 1
    vv = []
    loopcount = 0
    while True:
        if loopcount > 1000: 
            ps += cot(prob)
            print('cot: ', ps)
            print('loopcount')
            # return ['True', 'False'][torch.argmax(ps)]
            answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            sc_scores.append(ps.clone())
            return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
        loopcount += 1
        if calls > lim:
            print('***LIMIT EXCEEDED***')
            # v = return_exceeded(pb, nb, mapping, llm)
            # breakpoint()
            ps += cot(prob)
            print('cot: ', ps)
            # return ['True', 'False'][torch.argmax(ps)]
            answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            sc_scores.append(ps.clone())
            return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
            # return vv + ['LIMIT EXCEEDED'], v, bb, True, rule_scores
        if not dynamic:
                names = {}
                bb = get_bb(file, seedrun=seedrun)
                nb = bb['neg']
                pb = bb['pos']
                jb = list(set(pb).intersection(set(nb)))

                ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
                for b in ab:
                
                    phr = mapping[str(np.abs(b))]
        
                    pred = phr.split('__')[0].lower()
                    try:
                        if arity[pred] == 1:
                            ppl = [phr.split('__')[1]]
                        elif arity[pred] == 2:        
                            ppl = [phr.split('__')[1], phr.split('__')[2]]
                    except:
                        continue
                
                    for name in ppl:
                        if name not in names.keys():
                            names[name] = 1
                        else:
                            names[name] += 1




        uo = sorted(names, key=names.get)[::-1]
        do = sorted(names, key=names.get)
        #   

        bb = get_bb(file, seedrun=seedrun)
        nb = bb['neg']
        pb = bb['pos']
        jb = list(set(pb).intersection(set(nb)))

        ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
        
        
           

        # empath = USER_PATH + '/LLM-project/dimacs_pronto/'
        
        probs = torch.tensor([-10000, -100000])
        # breakpoint()
        # while torch.sum(torch.where(probs > thresh, 1, 0)) == 0:
            
        p1 = -1
        p2 = -1
        breakflag=False
        good=False
        for p1 in range(len(uo)):
            # p1 += 1
            good=False
            for p2 in range(len(do)):
                # p2 += 1


                name1 = uo[p1]
                name2 = do[p2]
                # if name1 == 'Richard' and name2 == 'Patricia':
                       
                # if name1 == name2:
                #     breakpoint()
                #     continue
                # if [name1, name2] in set_pairs or [name2, name1] in set_pairs:
                #     continue
                name3 = None
                n1var = 0
                n2var = 0
                for i in range(len(jb)):
                    b = mapping[str(np.abs(jb[i]))]
                    if name1 == b.split('__')[1]:
                        # name3 = b.split('_')[-3]
                        1 > 0
                    elif len(b.split('__')) == 3:
                        if name1 == b.split('__')[2]:
                            # name3 = b.split('_')[-2]
                            1 > 0
                    else: 
                        # breakpoint()
                        continue
                    # if name2 == name3: continue
                    for j in range(len(jb)):
                        #   
                        # try:
                        if name2 in mapping[str(np.abs(jb[j]))].split('__'):
                            n1var = jb[i]
                            n2var = jb[j]
                            # breakflag=True
                            # break
                    #     except:
                    #         print(j)
                            
                    # if breakflag:break
                
                
                

        
                #   
                # print(nv) 
                            for ___ in [1]:
                                if n1var == 0:
                                    # breakpoint()
                                    continue
                                print(name1, name2)
                                print(mapping[str(np.abs(n1var))], mapping[str(np.abs(n2var))])

                                
                                pred1 = mapping[str(np.abs(n1var))].lower().split('__')[0].lower()
                                
                                try:
                                    if arity[pred1] == 1:
                                        v1names = [mapping[str(np.abs(n1var))].split('__')[1]]
                                    elif arity[pred1]== 2:
                                        v1names = [mapping[str(np.abs(n1var))].split('__')[1], mapping[str(np.abs(n1var))].split('__')[2]]
                                except:
                                    continue
                                    breakpoint()

                                pred2 = mapping[str(np.abs(n2var))].lower().split('__')[0].lower()
                                try:
                            
                                    if arity[pred2] == 1:
                                        v2names = [mapping[str(np.abs(n2var))].split('__')[1]]
                                
                                    elif arity[pred2]== 2:
                                        v2names = [mapping[str(np.abs(n2var))].split('__')[1], mapping[str(np.abs(n2var))].split('__')[2]]

                                except:
                                    continue


                                pred1str = str(pred1)
                                pred2str = str(pred2)
                                if n1var < 0:
                                    pred1str = 'NOT_' + pred1
                                if n2var < 0:
                                    pred2str = 'NOT_' + pred2
                                v1names_str = ''
                                for name in v1names:
                                    v1names_str += str(name) + ', '
                                v1names_str = v1names_str[:-2]
                                v2names_str = ''
                                for name in v2names:
                                    v2names_str += str(name) + ', '
                                v2names_str = v2names_str[:-2]

                                known_preds_str = ''
                                for key, value in arity.items():
                                    known_preds_str += "\"" + str(key) + "\"" + ', '
                                known_preds_str = known_preds_str[:-2]

                                # v1names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                                # v1rel = ' '.join(mapping[str(np.abs(n1var))].lower().split('_')[:-3])

                                # v2names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                                # v2rel = ' '.join(mapping[str(np.abs(n2var))].lower().split('_')[:-3])

                                # question = "If " + v1names[1] + " is " + v1names[0] + '\'s '  + v1rel + " and " \
                                #     + v2names[1] + " is " + v2names[0] + "\'s " + v2rel + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                                n_fs = 2
                                # few_shot = "Fill in the blank with a known predicate: NOT_earth(Marvin) implies ___(Marvin). Known predicates are: \"mars\", \"old\", \"far\", \"planet\", \"earth\", \"alien\". Answer: \\box{ alien }.\n" \
                                #     + 'Fill in the blank with a known predicate: game(Zelda) and more_than_million_sold(Zelda) implies ___(Zelda). Known predicates are: \"game\", \"more_than_million_sold\", \"top_10\", \"fun\". Answer: \\box{ top_10 }.\n' \
                                #     + 'Fill in the blank with a known predicate: mixer(Djokovic) and grand_slam(Djokovic) implies ___(Djokovic). Known predicates are: \"mixer\", \"grand_slam\", \"athlete\", \"champion\", \"man\", \"tennis_player\". Answer: \\box{ tennis_player }.\n'
                                # few_shot = "Fill in the blank with a known preidcate: If Sam is a brown and Sam is NOT a orange then Sam is a ___. Known predicates are: \"blue\", \"orange\", \"mother\". Answer: \\box{ NOT blue}\n" \
                                #     + 'Fill in the blank with a known predicate: If A is a square then A is a ___. Known predicates are: \"circle\", \"shape\", \"rectangle\". Answer: \\box{ rectangle }.\n' \
                                #     + 'Fill in the blank with a known predicate: If Martha is a car then Martha is a ___. Known predicates are: \"hot dog\", \"machine\", \"car\". Answer: \\box{ machine }.\n'
                                # few_shot = "Fill in the blank with a known preidcate: If Mary is a cat then Mary is a ___. Known predicates are: \"mammal\", \"warm-blooded\", \"reptile\". Answer: \\box{ mammal }\n" \
                                #     + 'Fill in the blank with a known predicate: If _3 is a integer  then _3 is a ___. Known predicates are: \"fraction\", \"natural\", \"prime\". Answer: \\box{ NOT fraction }.\n' \
                                few_shot = "Fill in the blank: If Mary is a cat then Mary is a ___. Answer: \\box{ mammal }\n" \
                                + 'Fill in the blank: If _3 is a integer then _3 is a ___. Answer: \\box{ real number }.\n' \
                                # if pred1str == pred2str:
                                #     if name1 != name2:
                                #         question = "Fill in the blank with a known predicate: " + pred1str + '(' + v1names_str + ') implies ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                #     elif name1 == name2:
                                #         question = "Fill in the blank with a known predicate: " + 'If ' + v1names_str + ' is a ' + pred1str + ' then ' + name1 + ' is a ___. Known predicates are: ' + known_preds_str + '. Answer: \\box{ '

                                # else: 
                                #     if name1 != name2:
                                #         question = "Fill in the blank with a known predicate: " + pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                #     elif name1==name2: 
                                #         question = "Fill in the blank with a known predicate: " + 'If ' + v1names_str + ' is a ' + pred1str + ' and ' + name2 + ' is a ' + pred2str + ' then '+ name1 + ' is a ___. Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                
                                # question = 'Fill in the blank with a known predicate: If ' + v1names_str + ' is a ' + pred1str + ' then '+  name1 + ' is a ___. Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                question = 'Fill in the blank: If ' + v1names_str + ' is a ' + pred1str + ' then '+  name1 + ' is a ___' + '. Answer: \\box{ '

                                print(question)
                                # if 'win' in pred1str or 'win' in pred2str: 
                                #     breakpoint()

                                # question = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                                #     + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                                # out = search_pattern(v1names, pred1, v2names, pred2, patterns)
                                out=None
                                if out != None:
                                    rel = out[-1]
                                    print('FOUND PATTERN')
                                else:
                                    print(question)
                                    completion=llm.complete(few_shot + question)[0] 
                                    try:
                                        print(completion.split(question)[1].split('}')[0])
                                    except:
                                        continue
                                    calls += 1     
                                    # breakpoint()
                                    if calls > lim:
                                        # v = return_exceeded(pb, nb, mapping, llm)
                                        # breakpoint()

                                        print('***LIMIT EXCEEDED***')
                                        ps += cot(prob)
                                        print('cot: ', ps)
                                        sc_scores.append(ps.clone())
                                        # return ['True', 'False'][torch.argmax(ps)]
                                        answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                                        return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
                                        # return vv + ['LIMIT EXCEEDED'],v, bb, True, rule_scores
                                    # breakpoint()
                                    


                                    
                                    try:
                                        rel = '_'.join(completion.split('box{')[1+n_fs].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))
                                    except:
                                        print('error')
                                        continue
                                    if rel == '':
                                        continue
                                    if pred1str == pred2str:
                                        if name1 != name2:
                                            rule =  pred1str + '(' + v1names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                        elif name1 == name2:
                                            rule =  'If ' + v1names_str + ' is a ' + pred1str + ' then ' + name1 + ' is a ' + rel

                                    else: 
                                        if name1 != name2:
                                            rule =  pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                        elif name1==name2: 
                                            rule = 'If ' + v1names_str + ' is a ' + pred1str + ' and ' + name2 + ' is a ' + pred2str + ' then '+ name1 + ' is a ' + rel
                                    # if name1 != name2:
                                    #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                    # elif name1==name2: 
                                    #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ')'
                                    
                                    if '(' in rel or ')' in rel or '{' in rel or '}' in rel or '\\' in rel:
                                            continue
                                negative = 1
                                if rel.startswith('NOT_'):
                                    negative=-1
                                    rel = rel.strip('NOT_')
                                elif rel.startswith('not_'):
                                    negative=-1
                                    rel=rel.strip('not_')
                                elif rel.startswith('NOT '):
                                    negative=-1
                                    rel = rel.strip('NOT ')
                                elif rel.startswith('not '):
                                    negative=-1
                                    rel=rel.strip('not ')
                                # if '47' in mapping.keys():
                                #     breakpoint()
                                check, ab = rule_check(rule, prob=prob, llm=llm)
                                rule_scores[rule] = ab
                                if ab[0]==0.0675: breakpoint()
                                if rule == 'devices(model_xx) and controlled_by(employee, google_home) implies NOT belong_to(google_home, employee)': breakpoint()
                                if check == False:
                                        # breakpoint()
                                        # if a == '0.0675'
                                        continue
                                # if ab[0] < 0.5 or ab[1] < 0.5:
                                #     breakpoint()
                                if rel in arity.keys():
                                    if arity[rel] == 1:  
                                        nv_mapping = rel + '__' + name1 + '__'
                                        patterns = save_pattern(v1names, pred1, v2names, pred2, [name1], rel, patterns)                   
                                        if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                            # breakpoint()
                                            continue
                                        nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                        if nv in set_vars:
                                            # breakpoint()
                                            continue
                                        newv = True
                                        if nv_mapping in list(mapping.values()):
                                            for key, value in mapping.items():
                                                if value == nv_mapping:
                                                    nv = int(key)
                                                    newv = False
                                        mapping[str(nv)] = nv_mapping
                                        # if '47' in mapping.keys():
                                        #     breakpoint()
                                    elif arity[rel] == 2:
                                        nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                        patterns = save_pattern(v1names, pred1, v2names, pred2, [name1, name2], rel, patterns)
                                        patterns = save_pattern(v1names, pred1, v2names, pred2, [name2, name1], rel, patterns)

                                        if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                            # breakpoint()
                                            continue
                                        nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                        if nv in set_vars:
                                            # breakpoint()
                                            continue
                                        newv = True
                                        if nv_mapping in list(mapping.values()):
                                            for key, value in mapping.items():
                                                if value == nv_mapping:
                                                    nv = int(key)
                                                    newv = False
                                        mapping[str(nv)] = nv_mapping
                                        # if '47' in mapping.keys():
                                        #     breakpoint()
                                    
                                else:
                                    if name1 != name2:
                                        arity[rel] = 2
                                        nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                    else:
                                        arity[rel] = 1
                                        nv_mapping = rel + '__' + name1 + '__'
                                    if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                        breakpoint()
                                        continue
                                    nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                    if nv in set_vars:
                                        # breakpoint()
                                        continue
                                    newv = True
                                    if nv_mapping in list(mapping.values()):
                                        for key, value in mapping.items():
                                            if value == nv_mapping:
                                                nv = int(key)
                                                newv = False
                                    mapping[str(nv)] = nv_mapping
                                    # if '47' in mapping.keys():
                                    #     breakpoint()

                                # questioni = ''


                                if arity[rel] == 2:
                                    nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                                    if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                        if nv == np.max(list(map(int, list(mapping.keys())))):
                                            del mapping[str(nv)]
                                            continue

                                        
                                    nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                    newvi = True
                                    if nvi_mapping in list(mapping.values()):
                                        for key, value in mapping.items():
                                            if value == nvi_mapping:
                                                nvi = int(key)
                                                newvi = False
                                    mapping[str(nvi)] = nvi_mapping
                                
                                # if arity[rel] == 2:
                                #     nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                                #     if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                #         if nv == np.max(list(map(int, list(mapping.keys())))):
                                #             del mapping[str(nv)]
                                #             continue

                                        
                                #     nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                #     newvi = True
                                #     if nvi_mapping in list(mapping.values()):
                                #         for key, value in mapping.items():
                                #             if value == nvi_mapping:
                                #                 nvi = int(key)
                                #                 newvi = False
                                #     mapping[str(nvi)] = nvi_mapping
                                
                                probs = torch.tensor([10000, 100000])
                                # if '47' in mapping.keys():
                                #     breakpoint()
                                # if nv == 47 or nvi == 47:
                                #     breakpoint()
                                # if nv == 48 or nv == 49:
                                #     breakpoint()
                                # questioni = "If " + name2 + ' is ' + name1 + '\'s ' \
                                #     + rel + ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                                
                                # questioni = "If " + name2 + ' is the '  + rel + " of " + name1 +  \
                                #  ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                                
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
                                probs = torch.tensor([10000, 100000])


                                

                                #   
                                #   
                                
                                # print(nv)
                                # print(probs)
                                #   
                                # print(mapping[str(np.abs(nv))])
                                #   
                                # nv = 'hello'
                                # try:
                                vv += (negative*nv, mapping[str(np.abs(int(nv)))], question + rel)
                                    # breakpoint()
                                # except:
                                #     vv += (nv, '**COULD NOT FIND A COMMON-SENSE RULE')
                                #     print("**COULD NOT FIND A COMMON-SENSE RULE")
                                #     #   
                                #     break
                                    #   
                                print(vv)
                                print(ab)
                                # if 'newrules' not in prob.keys():
                                #     prob['newrules'] = [rule]
                                # else:
                                #     prob['newrules'].append(rule)
                                split = mapping[str(np.abs(nv))].split('__')
                                if len(split) == 3:
                                    varname = split[1] + ' is a ' + split[0] 
                                    # split[0] + '(' + split[1] + ')'
                                elif len(split) == 4:
                                    varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
                                if negative < 0:
                                    varname = split[1] + ' is NOT a ' + split[0]
                                prob['newrules'].append(varname)

                                # rule = ''
                                # parts = mapping[str(np.abs(int(nv)))].split('__')
                                
                                # if negative == 1:
                                    
                                # if 'newrules' not in prob.keys():
                                #     prob['newrules'] = [' ']
                                # else:
                                #     prob['newrules'].append(rule)
                                # ps += cot(prob)
                                # sc_scores.append(ps.clone())
                                # if len(prob['newrules']) >= fixed_iter:
                                                                       
                                    
                                #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                                #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

                                ps += cot(prob)
                                sc_scores.append(ps.clone())
                                print('cot: ', ps)
                                print('cot_thresh', (cot_thresh-0.05*(len(prob['newrules'])))*ps.sum())
                                if torch.max(ps) >= (cot_thresh-0.05*(len(prob['newrules'])))*ps.sum() or ps.sum() == 20:
                                    # return ['True', 'False'][torch.argmax(ps)]
                                    print('decided with cot')
                                    answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                                    return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
                                # breakpoint()
                                good=True
                                if dynamic:
                                    names[name2]=np.max(list(names.values()))+1
                                break
                            if good:
                                # neg = 1
                                # breakpoint()
                                break
                            negative=-1
                            print('NOT', name1, name2)
                            print(mapping[str(np.abs(n1var))], mapping[str(np.abs(n2var))])

                            
                            pred1 = mapping[str(np.abs(n1var))].lower().split('__')[0].lower()
                            
                            try:
                                if arity[pred1] == 1:
                                    v1names = [mapping[str(np.abs(n1var))].split('__')[1]]
                                elif arity[pred1]== 2:
                                    v1names = [mapping[str(np.abs(n1var))].split('__')[1], mapping[str(np.abs(n1var))].split('__')[2]]
                            except:
                                continue
                                breakpoint()

                            pred2 = mapping[str(np.abs(n2var))].lower().split('__')[0].lower()
                            try:
                        
                                if arity[pred2] == 1:
                                    v2names = [mapping[str(np.abs(n2var))].split('__')[1]]
                            
                                elif arity[pred2]== 2:
                                    v2names = [mapping[str(np.abs(n2var))].split('__')[1], mapping[str(np.abs(n2var))].split('__')[2]]

                            except:
                                continue


                            pred1str = str(pred1)
                            pred2str = str(pred2)
                            if n1var < 0:
                                pred1str = 'NOT_' + pred1
                            if n2var < 0:
                                pred2str = 'NOT_' + pred2
                            v1names_str = ''
                            for name in v1names:
                                v1names_str += str(name) + ', '
                            v1names_str = v1names_str[:-2]
                            v2names_str = ''
                            for name in v2names:
                                v2names_str += str(name) + ', '
                            v2names_str = v2names_str[:-2]

                            known_preds_str = ''
                            for key, value in arity.items():
                                known_preds_str += "\"" + str(key) + "\"" + ', '
                            known_preds_str = known_preds_str[:-2]

                            # v1names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                            # v1rel = ' '.join(mapping[str(np.abs(n1var))].lower().split('_')[:-3])

                            # v2names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                            # v2rel = ' '.join(mapping[str(np.abs(n2var))].lower().split('_')[:-3])

                            # question = "If " + v1names[1] + " is " + v1names[0] + '\'s '  + v1rel + " and " \
                            #     + v2names[1] + " is " + v2names[0] + "\'s " + v2rel + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                            n_fs = 2
                            # few_shot = "Fill in the blank with a known predicate: NOT_mars(Marvin) implies NOT ___(Marvin). Known predicates are: \"mars\", \"old\", \"far\", \"planet\", \"earth\", \"alien\". Answer: \\box{ alien }.\n" \
                            #     + 'Fill in the blank with a known predicate: game(Zelda) and more_than_million_sold(Zelda) implies NOT ___(Zelda). Known predicates are: \"game\", \"more_than_million_sold\", \"top_10\", \"fun\", \"unpopular". Answer: \\box{ unpopular }.\n' \
                            #     + 'Fill in the blank with a known predicate: mixer(Djokovic) and grand_slam(Djokovic) implies NOT ___(Djokovic). Known predicates are: \"mixer\", \"grand_slam\", \"athlete\", \"champion\", \"man\", \"tennis_player\", \"bad\". Answer: \\box{ bad }.\n'
                            # few_shot = "Fill in the blank with a known preidcate: If Sam is a brown and Sam is a orange then Sam is NOT a ___. Known predicates are: \"blue\", \"orange\", \"mother\". Answer: \\box{ blue}\n" \
                            #         + 'Fill in the blank with a known predicate: If A is a square then A is NOT a ___. Known predicates are: \"circle\", \"shape\", \"rectangle\". Answer: \\box{ circle }.\n' \
                            #         + 'Fill in the blank with a known predicate: If Martha is a car then Martha is NOT a ___. Known predicates are: \"hot dog\", \"machine\", \"car\". Answer: \\box{ car }.\n'
                            # few_shot = "Fill in the blank with a known preidcate: If Mary is a cat then Mary is NOT a ___. Known predicates are: \"mammal\", \"warm-blooded\", \"reptile\". Answer: \\box{ reptile }\n" \
                            #     + 'Fill in the blank with a known predicate: If _3 is a integer then _3 is NOT a ___. Known predicates are: \"fraction\", \"natural\", \"prime\", \"real\". Answer: \\box{ NOT real }.\n' 
                            few_shot = "Fill in the blank: If Mary is a cat then Mary is NOT a ___. Answer: \\box{ reptile }\n" \
                                + 'Fill in the blank: If _3 is a integer then _3 is NOT a ___. Answer: \\box{ fraction }.\n' 
                            if pred1str == pred2str:
                                if name1 != name2:
                                    question = "Fill in the blank with a known predicate: " + pred1str + '(' + v1names_str + ') implies NOT ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                elif name1 == name2:
                                    question = "Fill in the blank with a known predicate: " + 'If ' + v1names_str + ' is a ' + pred1str + ' then '+ name1 + ' is NOT a ___. Known predicates are: ' + known_preds_str + '. Answer: \\box{ '

                            else: 
                                if name1 != name2:
                                    question = "Fill in the blank with a known predicate: " + pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies NOT ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                elif name1==name2: 
                                    question = "Fill in the blank with a known predicate: " + 'If ' + v1names_str + ' is a ' + pred1str + ' and ' + name2 + ' is a ' + pred2str + ' then '+ name1 + ' is NOT a ___. Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                            # if 'win' in pred1str or 'win' in pred2str: 
                            #         breakpoint()

                            # question = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                            #     + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                            # out = search_pattern(v1names, pred1, v2names, pred2, patterns)
                            out=None
                            print(question)
                            if out != None:
                                rel = out[-1]
                                print('FOUND PATTERN')
                            else:
                                print(question)
                                completion=llm.complete(few_shot + question)[0]  
                                try:print(completion.split(question)[1].split('}')[0])
                                except: continue
                                calls += 1    
                                # breakpoint()
                                if calls > lim:
                                    # v = return_exceeded(pb, nb, mapping, llm)
                                    # breakpoint()

                                    print('***LIMIT EXCEEDED***')
                                    ps += cot(prob)
                                    sc_scores.append(ps.clone())
                                    print('cot: ', ps)
                                    # return ['True', 'False'][torch.argmax(ps)]
                                    answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                                    return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
                                    # return vv + ['LIMIT EXCEEDED'], v, bb, True, rule_scores, False
                                # breakpoint()
                                
                                try:
                                    rel = '_'.join(completion.split('box{')[1+n_fs].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))
                                except:
                                    continue
                                if rel == '':
                                    continue
                                if pred1str == pred2str:
                                    if name1 != name2:
                                        rule =  pred1str + '(' + v1names_str + ') implies NOT ' + rel + '(' + name1 + ', ' + name2 + ')'
                                    elif name1 == name2:
                                        rule = 'If ' + v1names_str + ' is a ' + pred1str +' then '+ name1 + ' is NOT a ' + rel

                                else: 
                                    if name1 != name2:
                                        rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies NOT ' + rel + '(' + name1 + ', ' + name2 + ')'
                                    elif name1==name2: 
                                        rule ='If ' + v1names_str + ' is a ' + pred1str + ' and ' + name2 + ' is a ' + pred2str + ' then '+ name1 + ' is NOT a ' + rel
                                # if name1 != name2:
                                #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                # elif name1==name2: 
                                #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ')'
                                check, ab = rule_check(rule, prob=prob, llm=llm)
                                if ab[0]==0.0675: breakpoint()
                                if rule == 'devices(model_xx) and controlled_by(employee, google_home) implies NOT belong_to(google_home, employee)': breakpoint()
                                rule_scores[rule] = ab
                                if check == False:
                                    # breakpoint()
                                    continue
                                # if ab[0] < 0.5 or ab[1] < 0.5:
                                #     breakpoint()
                                if '(' in rel or ')' in rel or '{' in rel or '}' in rel or '\\' in rel:
                                    continue
                            # negative = 1
                            # if rel.startswith('not_'):
                            #     negative=-1
                            #     rel = rel.strip('not_')
                            # if '47' in mapping.keys():
                            #     breakpoint()
                            if rel in arity.keys():
                                if arity[rel] == 1:  
                                    nv_mapping = rel + '__' + name1 + '__'
                                    patterns = save_pattern(v1names, pred1, v2names, pred2, [name1], rel, patterns)                   
                                    if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                        # breakpoint()
                                        continue
                                    nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                    if nv in set_vars:
                                        # breakpoint()
                                        continue
                                    newv = True
                                    if nv_mapping in list(mapping.values()):
                                        for key, value in mapping.items():
                                            if value == nv_mapping:
                                                nv = int(key)
                                                newv = False
                                    mapping[str(nv)] = nv_mapping
                                    # if '47' in mapping.keys():
                                    #     breakpoint()
                                elif arity[rel] == 2:
                                    nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                    patterns = save_pattern(v1names, pred1, v2names, pred2, [name1, name2], rel, patterns)
                                    patterns = save_pattern(v1names, pred1, v2names, pred2, [name2, name1], rel, patterns)

                                    if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                        # breakpoint()
                                        continue
                                    nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                    if nv in set_vars:
                                        # breakpoint()
                                        continue
                                    newv = True
                                    if nv_mapping in list(mapping.values()):
                                        for key, value in mapping.items():
                                            if value == nv_mapping:
                                                nv = int(key)
                                                newv = False
                                    mapping[str(nv)] = nv_mapping
                                    # if '47' in mapping.keys():
                                    #     breakpoint()
                                
                            else:
                                if name1 != name2:
                                    arity[rel] = 2
                                    nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                else:
                                    arity[rel] = 1
                                    nv_mapping = rel + '__' + name1 + '__'
                                if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                    # breakpoint()
                                    continue
                                nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                if nv in set_vars:
                                    # breakpoint()
                                    continue
                                newv = True
                                if nv_mapping in list(mapping.values()):
                                    for key, value in mapping.items():
                                        if value == nv_mapping:
                                            nv = int(key)
                                            newv = False
                                mapping[str(nv)] = nv_mapping
                                # if '47' in mapping.keys():
                                #     breakpoint()

                            # questioni = ''


                            if arity[rel] == 2:
                                nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                                if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                    if nv == np.max(list(map(int, list(mapping.keys())))):
                                        del mapping[str(nv)]
                                        continue

                                    
                                nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                newvi = True
                                if nvi_mapping in list(mapping.values()):
                                    for key, value in mapping.items():
                                        if value == nvi_mapping:
                                            nvi = int(key)
                                            newvi = False
                                mapping[str(nvi)] = nvi_mapping
                            
                            # if arity[rel] == 2:
                            #     nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                            #     if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                            #         if nv == np.max(list(map(int, list(mapping.keys())))):
                            #             del mapping[str(nv)]
                            #             continue

                                    
                            #     nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                            #     newvi = True
                            #     if nvi_mapping in list(mapping.values()):
                            #         for key, value in mapping.items():
                            #             if value == nvi_mapping:
                            #                 nvi = int(key)
                            #                 newvi = False
                            #     mapping[str(nvi)] = nvi_mapping
                            
                            probs = torch.tensor([10000, 100000])
                            # if '47' in mapping.keys():
                            #     breakpoint()
                            # if nv == 47 or nvi == 47:
                            #     breakpoint()
                            # if nv == 48 or nv == 49:
                            #     breakpoint()
                            # questioni = "If " + name2 + ' is ' + name1 + '\'s ' \
                            #     + rel + ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                            
                            # questioni = "If " + name2 + ' is the '  + rel + " of " + name1 +  \
                            #  ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                            
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
                            probs = torch.tensor([10000, 100000])


                            

                            #   
                            #   
                            
                            # print(nv)
                            # print(probs)
                            #   
                            # print(mapping[str(np.abs(nv))])
                            #   
                            # nv = 'hello'
                            # try:
                            vv += (negative*nv, mapping[str(np.abs(int(nv)))], question + rel, 'calls: ' + str(calls))
                            # breakpoint()
                                # breakpoint()
                            # except:
                            #     vv += (nv, '**COULD NOT FIND A COMMON-SENSE RULE')
                            #     print("**COULD NOT FIND A COMMON-SENSE RULE")
                            #     #   
                            #     break
                                #   
                            print(vv)
                            print(ab)
                            # if 'newrules' not in prob.keys():
                            #     prob['newrules'] = [rule]
                            # else:
                            #     prob['newrules'].append(rule)
                            split = mapping[str(np.abs(nv))].split('__')
                            if len(split) == 3:
                                varname = split[1] +  ' is a ' + split[0]
                                # split[0] + '(' + split[1] + ')'
                            elif len(split) == 4:
                                varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
                            if negative < 0:
                                varname = split[1] + ' is NOT a ' + split[0]
                            prob['newrules'].append(varname)
                            # ps += cot(prob)
                            # sc_scores.append(ps.clone())
                            # if len(prob['newrules']) >= fixed_iter:
                                                    
                                
                            #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                            #     return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
                            ps += cot(prob)
                            sc_scores.append(ps.clone())
                            print('cot: ',ps)
                            print('cot_thresh', (cot_thresh-0.05*(len(prob['newrules'])-1))*ps.sum())
                            if torch.max(ps) >= (cot_thresh-0.05*(len(prob['newrules'])-1))*ps.sum() or ps.sum() == 20:
                                # return ['True', 'False'][torch.argmax(ps)]
                                answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                                print('answered with cot')

                                return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
                            # breakpoint()
                            good=True
                            if dynamic:
                                names[name2]=np.max(list(names.values()))+1
                            break

                        if good:
                            # breakpoint()
                            break
                    if good:
                        # breakpoint()
                        break
                if good:
                    break
            if good: break
            # breakpoint()
        # breakpoint()
            # if good:
            #     break
            # #   
            # if len(vv) > lim*2:
            #     vv += ["ERROR: TIME OUT"]
            #     print("ERROR TIME OUT")
            #     break
        if not good: 
            negative=-1
            for p1 in range(len(uo)):
            # p1 += 1
                good=False
                for p2 in range(len(do)):
                    # p2 += 1


                    name1 = uo[p1]
                    name2 = do[p2]
                    # if name1 == 'Richard' and name2 == 'Patricia':
                        
                    # if name1 == name2:
                    #     breakpoint()
                    #     continue
                    # if [name1, name2] in set_pairs or [name2, name1] in set_pairs:
                    #     continue
                    name3 = None
                    n1var = 0
                    n2var = 0
                    for i in range(len(jb)):
                        b = mapping[str(np.abs(jb[i]))]
                        if name1 == b.split('__')[1]:
                            # name3 = b.split('_')[-3]
                            1 > 0
                        elif len(b.split('__')) == 3:
                            if name1 == b.split('__')[2]:
                                # name3 = b.split('_')[-2]
                                1 > 0
                        else: 
                            # breakpoint()
                            continue
                        # if name2 == name3: continue
                        for j in range(len(jb)):
                            #   
                            # try:
                            if name2 in mapping[str(np.abs(jb[j]))].split('__'):
                                n1var = jb[i]
                                n2var = jb[j]
                                # breakflag=True
                                # break
                        #     except:
                        #         print(j)
                                
                        # if breakflag:break
                    
                    
                    

            
                    #   
                    # print(nv)

                                if n1var == 0:
                                    # breakpoint()
                                    continue
                                # print(name1, name2)
                                # print(mapping[str(np.abs(n1var))], mapping[str(np.abs(n2var))])

                                
                                # pred1 = mapping[str(np.abs(n1var))].lower().split('__')[0].lower()
                                
                                # try:
                                #     if arity[pred1] == 1:
                                #         v1names = [mapping[str(np.abs(n1var))].split('__')[1]]
                                #     elif arity[pred1]== 2:
                                #         v1names = [mapping[str(np.abs(n1var))].split('__')[1], mapping[str(np.abs(n1var))].split('__')[2]]
                                # except:
                                #     continue
                                #     breakpoint()

                                # pred2 = mapping[str(np.abs(n2var))].lower().split('__')[0].lower()
                                # try:
                            
                                #     if arity[pred2] == 1:
                                #         v2names = [mapping[str(np.abs(n2var))].split('__')[1]]
                                
                                #     elif arity[pred2]== 2:
                                #         v2names = [mapping[str(np.abs(n2var))].split('__')[1], mapping[str(np.abs(n2var))].split('__')[2]]

                                # except:
                                #     continue


                                # pred1str = str(pred1)
                                # pred2str = str(pred2)
                                # if n1var < 0:
                                #     pred1str = 'NOT_' + pred1
                                # if n2var < 0:
                                #     pred2str = 'NOT_' + pred2
                                # v1names_str = ''
                                # for name in v1names:
                                #     v1names_str += str(name) + ', '
                                # v1names_str = v1names_str[:-2]
                                # v2names_str = ''
                                # for name in v2names:
                                #     v2names_str += str(name) + ', '
                                # v2names_str = v2names_str[:-2]

                                # known_preds_str = ''
                                # for key, value in arity.items():
                                #     known_preds_str += "\"" + str(key) + "\"" + ', '
                                # known_preds_str = known_preds_str[:-2]

                                # # v1names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                                # # v1rel = ' '.join(mapping[str(np.abs(n1var))].lower().split('_')[:-3])

                                # # v2names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                                # # v2rel = ' '.join(mapping[str(np.abs(n2var))].lower().split('_')[:-3])

                                # # question = "If " + v1names[1] + " is " + v1names[0] + '\'s '  + v1rel + " and " \
                                # #     + v2names[1] + " is " + v2names[0] + "\'s " + v2rel + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                                # n_fs = 3
                                # few_shot = "Fill in the blank with either a new or known predicate: NOT_mars(Marvin) implies NOT ___(Marvin). Known predicates are: \"mars\", \"old\", \"far\", \"planet\", \"earth\", \"alien\". Answer: \\box{ alien }.\n" \
                                #     + 'Fill in the blank with either a new or known predicate: game(Zelda) and more_than_million_sold(Zelda) implies NOT ___(Zelda). Known predicates are: \"game\", \"more_than_million_sold\", \"top_10\", \"fun\", \"unpopular". Answer: \\box{ unpopular }.\n' \
                                #     + 'Fill in the blank with either a new or known predicate: mixer(Djokovic) and grand_slam(Djokovic) implies NOT ___(Djokovic). Known predicates are: \"mixer\", \"grand_slam\", \"athlete\", \"champion\", \"man\", \"tennis_player\", \"bad\". Answer: \\box{ bad }.\n'
                                
                                # if pred1str == pred2str:
                                #     if name1 != name2:
                                #         question = "Fill in the blank with either a new or known predicate: " + pred1str + '(' + v1names_str + ') implies NOT ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                #     elif name1 == name2:
                                #         question = "Fill in the blank with either a new or known predicate: " + pred1str + '(' + v1names_str + ') implies NOT ___(' + name1 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '

                                # else: 
                                #     if name1 != name2:
                                #         question = "Fill in the blank with either a new or known predicate: " + pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies NOT ___(' + name1 + ', ' + name2 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '
                                #     elif name1==name2: 
                                #         question = "Fill in the blank with either a new or known predicate: " + pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies NOT ___(' + name1 + '). Known predicates are: ' + known_preds_str + '. Answer: \\box{ '


                                # # question = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                                # #     + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ "
                                # # out = search_pattern(v1names, pred1, v2names, pred2, patterns)
                                # out=None
                                # if out != None:
                                #     rel = out[-1]
                                #     print('FOUND PATTERN')
                                # else:
                                #     completion=llm.complete(few_shot + question)[0]  
                                #     calls += 1    
                                #     # breakpoint()
                                #     if calls > lim:
                                #         v = return_exceeded(pb, nb, mapping, llm)
                                #         # breakpoint()

                                #         print('***LIMIT EXCEEDED***')
                                #         return vv + ['LIMIT EXCEEDED'], v, bb, True
                                #     # breakpoint()
                                    
                                #     try:
                                #         rel = '_'.join(completion.split('box{')[1+n_fs].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))
                                #     except:
                                #         continue
                                #     if rel == '':
                                #         continue
                                #     if pred1str == pred2str:
                                #         if name1 != name2:
                                #             rule =  pred1str + '(' + v1names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                #         elif name1 == name2:
                                #             rule = pred1str + '(' + v1names_str + ') implies ' + rel + '(' + name1 + ')'

                                #     else: 
                                #         if name1 != name2:
                                #             rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                #         elif name1==name2: 
                                #             rule =pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ')'
                                #     # if name1 != name2:
                                #     #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ', ' + name2 + ')'
                                #     # elif name1==name2: 
                                #     #     rule = pred1str + '(' + v1names_str + ') and ' + pred2str + '(' + v2names_str + ') implies ' + rel + '(' + name1 + ')'
                                #     if rule_check(rule,prob=prob, llm=llm) == False:
                                #         # breakpoint()
                                #         continue
                                #     if '(' in rel or ')' in rel or '{' in rel or '}' in rel or '\\' in rel:
                                #         continue
                                # # negative = 1
                                # # if rel.startswith('not_'):
                                # #     negative=-1
                                # #     rel = rel.strip('not_')
                                # # if '47' in mapping.keys():
                                # #     breakpoint()
                                # if rel in arity.keys():
                                #     if arity[rel] == 1:  
                                #         nv_mapping = rel + '__' + name1 + '__'
                                #         patterns = save_pattern(v1names, pred1, v2names, pred2, [name1], rel, patterns)                   
                                #         if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                #             # breakpoint()
                                #             continue
                                #         nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                #         if nv in set_vars:
                                #             # breakpoint()
                                #             continue
                                #         newv = True
                                #         if nv_mapping in list(mapping.values()):
                                #             for key, value in mapping.items():
                                #                 if value == nv_mapping:
                                #                     nv = int(key)
                                #                     newv = False
                                #         mapping[str(nv)] = nv_mapping
                                #         # if '47' in mapping.keys():
                                #         #     breakpoint()
                                #     elif arity[rel] == 2:
                                #         nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                #         patterns = save_pattern(v1names, pred1, v2names, pred2, [name1, name2], rel, patterns)
                                #         patterns = save_pattern(v1names, pred1, v2names, pred2, [name2, name1], rel, patterns)

                                #         if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                #             # breakpoint()
                                #             continue
                                #         nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                #         if nv in set_vars:
                                #             # breakpoint()
                                #             continue
                                #         newv = True
                                #         if nv_mapping in list(mapping.values()):
                                #             for key, value in mapping.items():
                                #                 if value == nv_mapping:
                                #                     nv = int(key)
                                #                     newv = False
                                #         mapping[str(nv)] = nv_mapping
                                #         # if '47' in mapping.keys():
                                #         #     breakpoint()
                                    
                                # else:
                                #     if name1 != name2:
                                #         arity[rel] = 2
                                #         nv_mapping = rel + '__' + name1 + '__' + name2 + '__'
                                #     else:
                                #         arity[rel] = 1
                                #         nv_mapping = rel + '__' + name1 + '__'
                                #     if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                #         # breakpoint()
                                #         continue
                                #     nv = np.max(list(map(int, list(mapping.keys())))) + 1
                                #     if nv in set_vars:
                                #         # breakpoint()
                                #         continue
                                #     newv = True
                                #     if nv_mapping in list(mapping.values()):
                                #         for key, value in mapping.items():
                                #             if value == nv_mapping:
                                #                 nv = int(key)
                                #                 newv = False
                                #     mapping[str(nv)] = nv_mapping
                                #     # if '47' in mapping.keys():
                                #     #     breakpoint()

                                # # questioni = ''


                                # if arity[rel] == 2:
                                #     nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                                #     if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                #         if nv == np.max(list(map(int, list(mapping.keys())))):
                                #             del mapping[str(nv)]
                                #             continue

                                        
                                #     nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                #     newvi = True
                                #     if nvi_mapping in list(mapping.values()):
                                #         for key, value in mapping.items():
                                #             if value == nvi_mapping:
                                #                 nvi = int(key)
                                #                 newvi = False
                                #     mapping[str(nvi)] = nvi_mapping
                                
                                # # if arity[rel] == 2:
                                # #     nvi_mapping = rel + '__' + name2 + '__' + name1 + '__'
                                # #     if nvi_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                                # #         if nv == np.max(list(map(int, list(mapping.keys())))):
                                # #             del mapping[str(nv)]
                                # #             continue

                                        
                                # #     nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                # #     newvi = True
                                # #     if nvi_mapping in list(mapping.values()):
                                # #         for key, value in mapping.items():
                                # #             if value == nvi_mapping:
                                # #                 nvi = int(key)
                                # #                 newvi = False
                                # #     mapping[str(nvi)] = nvi_mapping
                                
                                # probs = torch.tensor([10000, 100000])
                                # # if '47' in mapping.keys():
                                # #     breakpoint()
                                # # if nv == 47 or nvi == 47:
                                # #     breakpoint()
                                # # if nv == 48 or nv == 49:
                                # #     breakpoint()
                                # # questioni = "If " + name2 + ' is ' + name1 + '\'s ' \
                                # #     + rel + ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                                
                                # # questioni = "If " + name2 + ' is the '  + rel + " of " + name1 +  \
                                # #  ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank using gender-specific terms. Answer: \\box{ '
                                
                                # # completioni = llm.complete(questioni)[0]

                                # # # print(completion)
                                # # reli = '_'.join(completioni.split('box{')[1].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip(' ').strip(' ').lower().split(' '))
                                # # patterns = save_pattern(v1names, v1rel, v2names, v2rel, [name2, name1], reli, patterns)
                                # # patterns = save_pattern(v2names, v2rel, v1names, v1rel, [name2, name1], reli, patterns)
                                
                                # # nv_mappingi = reli + '_' + name2 + '_' + name1 + '_'
                                # # nvi = np.max(list(map(int, list(mapping.keys())))) + 1
                                # # newvi=True
                                # # if nv_mappingi in list(mapping.values()):
                                # #     for key, value in mapping.items():
                                # #         if value == nv_mappingi:
                                # #             nvi = int(key)
                                # #             newvi=False
                                # # mapping[str(nvi)] = nv_mappingi
                                # probs = torch.tensor([10000, 100000])


                                

                                # #   
                                # #   
                                
                                # # print(nv)
                                # # print(probs)
                                # #   
                                # # print(mapping[str(np.abs(nv))])
                                # #   
                                # # nv = 'hello'
                                # # try:
                                # vv += (nv, mapping[str(np.abs(int(nv)))], question + rel, 'calls: ' + str(calls))
                                # # breakpoint()
                                #     # breakpoint()
                                # # except:
                                # #     vv += (nv, '**COULD NOT FIND A COMMON-SENSE RULE')
                                # #     print("**COULD NOT FIND A COMMON-SENSE RULE")
                                # #     #   
                                # #     break
                                #     #   
                                # print(vv)
                                # # breakpoint()
                                # good=True
                                # if dynamic:
                                #     names[name2]=np.max(list(names.values()))+1
                                # break
                            if good:
                                # neg = 1
                                # breakpoint()
                                break
                        if good:
                            # breakpoint()
                            break
                    if good:
                        # breakpoint()
                        break
                if good:
                    # breakpoint()
                    break
            # if good:
            #     breakpoint()
            #     break
            # #   
            # if len(vv) > lim*2:
            #     vv += ["ERROR: TIME OUT"]
            #     print("ERROR TIME OUT")
            #     break
                # #   
                # if len(vv) > lim*2:
                #     vv += ["ERROR: TIME OUT"]
                #     print("ERROR TIME OUT")
                #     break
        # breakpoint()
        if not good:
            # breakpoint()
            # breakpoint()
            # missed_flag=True
            # if len(pb) > len(nb):
            #     new_sols={'pos': [1], 'neg': []}
            # else:
            #     new_sols = {'pos': [], 'neg':[1]}
            # # print("missed")
            # # breakpoint()
            # # v = return_exceeded(pb, nb, mapping, llm)
            # print('***END REACHED****')
            # return vv + ['END REACHED'], v, bb, True
            continue
            # break
        tmpfiles = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
        for f in tmpfiles:
            # breakpoint()
            add_clause(f)
            if arity[rel] ==2:
                add_clause(f)
            cf = open(f, 'a')
            if newv:
                add_var(f)
            cf.write('\n' + str(negative*nv) + ' 0')
            if arity[rel] == 2:
                if newvi:
                    add_var(f)
                cf.write('\n' + str(negative*nvi) + ' 0')
            cf.close()
        di = []
        try:
            new_sols = get_sol(file, lim=1, seedrun=seedrun)
        except:
            breakpoint()
        
        set_vars.append(nv)
        # set_vars.append(nvi)
        set_pairs.append([name1, name2])
        #   
        #   
        print('end loop new_sols:', [len(s) for s in new_sols.values()])
        if len(new_sols['pos']) == 0 or len(new_sols['neg']) == 0:
            return vv, new_sols, bb, missed_flag, rule_scores, False, sc_scores
    # print('done')
    if len(vv) == 0:
        print('***END REACHED***')
        # v = return_exceeded(pb, nb, mapping, llm)
        # breakpoint()
        ps += cot(prob)
        sc_scores.append(ps.clone())
        print('cot: ', ps)
        # return ['True', 'False'][torch.argmax(ps)]
        answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
        return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
        # return vv + ['LIMIT EXCEEDED'], v, bb, True
    breakpoint()
    return vv, new_sols, bb, missed_flag, rule_scores, False, sc_scores
if __name__ == '__main__':
    # noisy_data = ['clutrr33.cnf']
    noisy_data=[]
    seedrun = 'pronto_2'
    config = 'gc smaller cot cotthresh08 sc5 no knownpreds rulethresh00 quicksolve 20 temp1, checksol, folio fewshot'
    # config = 
    # thresh=0.8
    # dynamic=True
    # context_thresh=0.5
    # ruleethresh=0.4
    try:
        os.mkdir('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/tempfiles' + str(seedrun) + '/')
        os.mkdir('/home/XXXX/XXXX/fs_backup_feb13/LLM-project/workfiles' + str(seedrun) + '/')
    except:
        print('dir already exists')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(seedrun.split('_')[1])

    # mistr_data = ['clutrr10.cnf']
    # bad_data = ['proofd5153.cnf', 'proofd5227.cnf']
    bad_data = []
    mistr_data = []
    c = USER_PATH + '/LLM-project/dimacs_pronto_csvs/solver_finished.csv'
    import csv
    import json
    dataset = USER_PATH + '/SAT-LM/data/pronto_test.json'
    with open(dataset, 'r') as df:
        data = json.loads(df.read())
    # breakpoint()
    task = 'pronto'
    missed=False
    c = open(c, 'r')
    cr = csv.reader(c)
    names = []
    all_outs = {}
    missed_list = []
    labels = {}
    for row in cr:
        if row[2] == 'SAT' and row[3] == 'SAT':
            cnf = open(USER_PATH + '/LLM-project/dimacs_pronto/neg_'+row[1]).readlines()[0].strip('\n')
            num_clause = int(cnf.split(' ')[-1])
            if row[1] in noisy_data or row[1] in mistr_data:
                print('noisy or mistranslate')
                continue
            if task=='pronto':
                bb = get_bb(USER_PATH + '/LLM-project/dimacs_pronto/'+row[1], seedrun=seedrun)
                jb = set(bb['pos']).intersection(set(bb['neg']))
                if len(jb) == 0:
                    print("jb = 0", USER_PATH + '/SAT-LM/tmp/' + row[1][:-4] + '.py')
                    continue
            # if num_clause > 500:
                # continue
            if row[1] in bad_data:
                print('bad data')
                continue
            names.append(row[1])
            labels[row[1]] = data[int(row[1].split('proofd5')[1].split('.')[0])]['label']
    #   
    preds = {}
    if task == 'clutrr':
        c = open(USER_PATH + '/LLM-project/clutrr_labels.csv', 'r')
        cr = csv.reader(c)
        for row in cr:
            cnf = open(USER_PATH + '/LLM-project/dimacs_pronto/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
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
    # names = ['proofd537.cnf']
    uhohs = []
    times = {}
    acc = 0
    counter = 0
    total_rule_scores = {}
    hunh_list = []
    cot_list = []
    cot_acc = 0
    for name in (pbar := tqdm(names[::-1])):
    # name
        # if name != 'proofd5549.cnf':
        #     continue
        print(config)
        prob = data[int(name.split('proofd5')[1].split('.')[0])]
        start_time = time.time()
        print(name)
        p = USER_PATH + '/LLM-project/dimacs_pronto/' + name
        # p = USER_PATH + '/LLM-project/tempfiles/dimacs_test.cnf'
        # sols = get_sol(p, lim=100)
        #   
        #   
        bb = get_bb(p, seedrun=seedrun)

        prep_time = time.time() - start_time
        # sols = np.load(open("/home/XXXX/LLM-project/tempfiles/sols.np.npy", 'rb'), allow_pickle=True)
        # bb = np.load(open("/home/XXXX/LLM-project/tempfiles/bb.np.npy", 'rb'), allow_pickle=True)
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
        
        vv, solout, bbout, missed_flag, rule_scores, cot_flag, sc_scores= next_var(bb, p, llm=llm, task=task, missed=missed, prob=prob, question= data[int(name.split('proofd5')[1].split('.')[0])]['question'], seedrun=seedrun)
        for key, value in rule_scores.items():
            total_rule_scores[key] = value
        end_time = time.time() - start_time - prep_time
        if cot_flag==True:
            cot_list.append(name)
        times[name] = {'prep_time': prep_time, 'run_time': end_time}
        #   
        # print('finished!')
        #   
        all_outs[name] = (vv, solout, bbout, missed_flag, sc_scores)
        # if not missed_flag == None:
        #     missed_list.append([name, vv])
            
        if vv[-1] == 'LIMIT EXCEEDED':
            missed_list.append([name, vv])


        if ((solout == None and vv == None and bbout == None) or not missed_flag==None) and missed:
            preds[name] = 'missed'            
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
            
        elif len(solout['pos']) == 0 and len(solout['neg']) == 0:
            print('hunh')
            preds[name] = 'true'
            hunh_list.append(name)

        else:
            print('l. 1008 uh oh')
            uhohs.append(name)
            breakpoint()
            #   
        
        print('label:', labels[name])
        print('pred:', preds[name])
        if labels[name] == preds[name]:
            acc += 1
            counter += 1
            if cot_flag:
                cot_acc += 1
        else:
            counter += 1
        
        pbar.set_description('Acc: ' + str(acc / counter) + ', COT Acc: ' + str(cot_acc) + '/' + str(len(cot_list)))

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

    # for miss in missed_lis:
    #     print(data[int(miss.split('clutrr')[1].split('.cnf')[0])]['missing'])
    # print('
    print(config)

    breakpoint()