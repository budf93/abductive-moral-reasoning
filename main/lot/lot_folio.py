
import os
import shutil
import numpy as np
import time
import json
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

devices = [2]
d_str = ''
for d in devices:
    d_str += str(d) + ','
d_str = d_str[:-1]
os.environ["CUDA_VISIBLE_DEVICES"] = d_str


import torch
from torch.utils.data import DataLoader
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

args = {'train_file_path': './example_data', 'test_file_path': './example_data', 'save_path': './../SFT_train_res', 'engine': 'Qwen/Qwen2.5-VL-32B-Instruct', 
        'n_rows': 20, 'max_length': 1000,'temperature': 1, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
# args['engine'] = 'meta-llama/Meta-Llama-3-70B-Instruct'
# args['engine'] = "mistralai/Mistral-7B-Instruct-v0.3"
# args['engine'] = "Qwen/Qwen2.5-VL-32B-Instruct"
args['engine'] = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
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
            
            from tokenizers import Tokenizer
            from tokenizers.pre_tokenizers import Whitespace

            # tokenizer = Tokenizer(BPE())
            self.tokenizer = AutoTokenizer.from_pretrained(
                    args.engine,
                    cache_dir = cache_dir,
                    token = os.getenv("HF_TOKEN"),
                    # attn_implementation="flash_attention_2"

                    )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id    
            self.tokenizer.pre_tokenizer = Whitespace()
        
            self.model = AutoModelForCausalLM.from_pretrained(
                    args.engine, 
                    cache_dir = cache_dir,
                    quantization_config=quant_config,
                    device_map='auto',
                    # device_map = ['cuda:4', 'cuda:5'],
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
        max_length=2000
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


def extraction(problem):
    prompt = 'Please use uppercase English letters such as A, B, C, etc. to identify all possible propositions. Do not include negative tones such as \"not\" ' + \
    'in the propositions. For example, if the sentence is \"It is not bored,\" you should use \"A: bored\" to represent it.' + \
    '\nNext , for each proposition , use the symbol to represent its negative form. For example, the negative form of proposition A can be expressed as ¬A.' + \
    '\nNow, please carefully analyze the context and find causal relationship between propositions. A causal expression is only established when the ' + \
    'context directly supports this relationship. Use arrows (→) to indicate causal relationships, for example, \"If A, then B\", ' + \
    '\"B if A\" and \"A causes B\" etc. can be represented as A→B. \n' + \
    'Finally , output propositions and causal expressions\n'
    fewshot = 'Input:The state of Montana includes the cities of Butte, Helena, and Missoula.\nWhite Sulphur Springs and Butte are cities in the same state in U.S.\nA city can only be in one state in U.S. except for Bristol, Texarkana, Texhoma and Union City.\n' + \
             'Output:\nPropositions:\nA: in(Montana, Butte)\nB: in(Montana, Helena)\nC: in(Montana, Missoula)\nD: same_state(White_Sulphur_Springs, Butte)\nE: one_state_only(Bristol)\nF: one_state_only(Texarkana)\nG: one_state_only(Texhoma)\nH: one_state_only(Union_City)' + \
             '\nLogical Extraction of Context (purely symbolic):\n' + \
             'A\nB\nC\nD\n¬E\n¬F\n¬G\n¬H  *** DONE ***'

    # few_shot = ''

    return llm.complete('Problem no. 1): \n' + prompt + fewshot + '\nProblem no. 2): \n' +  prompt + 'Input: \n' + problem + '\n' + 'Output:\n', max_new = 300)[0].split('*** DONE ***')[1]

def extension(problem, extraction):
    
    prompt = 'Please extend this logic problem with additional rules deduced from information from the problem. '
    fewshot = '\nInput: \nA → B\nB → C\nOutput: \n¬B → ¬A\nA → C \n*** DONE ***'
    # 'Similarly, given (A → B) AND (B → C), we can deduce (A → C). \n'

    return llm.complete('Problem no. 1): \n' + prompt + fewshot + '\nProblem no. 2): \n' + prompt + '\nInput: ' + extraction.split('Logical Extraction of Context:')[-1] + '\nOutput: ', max_new = 300)[0]

def translation(problem, extraction, extension):
    prompt = 'Please use the provided propositions to translate each expression into a complete sentence. ' + \
    '¬A represents the negation of proposition A, the arrow (→) represents the causal relationship, and A→B represents if A, then B.' + \
    ' Only output the sentences in a paragraph! \n'
    fewshot = 'Input:  Here is a list of symbol translations to use when translating the expressions: \n ' + 'A: in(Montana, Butte)\nB: in(Montana, Helena)\nC: in(Montana, Missoula)\nD: same_state(White_Sulphur_Springs, Butte)\nE: one_state_only(Bristol)\nF: one_state_only(Texarkana)\nG: one_state_only(Texhoma)\nH: one_state_only(Union_City)\n' + 'Here is the list of expressions to be translated:' + '\nA\nB\nC\n' + \
            'Output: \nButte is in Montana.\nHelena is in Montana\nMissoula is in Montana\n*** DONE ***\n'
    return llm.complete('Problem no. 1: \n' + prompt +  fewshot + 'Problem no. 2: \n' + prompt + 'Input:  Here is a list of symbol translations to use when translating the expressions: \n' +  extraction.split('\nPropositions:')[1].split('Negative Propositions')[0] + '\nHere is the list of expressions to be translated:\n' + '\n'.join(extension.split('Output:')[2].split('\n*** DONE ***')[0].split('\n')[1:]) + '\nOutput:', max_new=300)

llm = LLM()

dataset = USER_PATH + '/SAT-LM/data/folio_proofd5_test.json'
with open(dataset, 'r') as df:
    data = json.loads(df.read())

import shutil

def get_bb(file, del_sols=None):
    bb = {'pos':  [], 'neg': []}
    
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for i in range(len(files)):
        file = files[i]
        shutil.copy(file, '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]))
        if not del_sols==None:
            if 'pos' in file:
                if 'neg' in file:
                    print('l. 416 uh oh')
                      
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
        os.system("timeout 5000 " + USER_PATH + "/LLM-project/cadiback/cadiback " + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1]) + '> '  + '/'.join(file.split('/')[:-2]) + '/tempfiles/' + str(file.split('/')[-1])[:-4] + ".bbone")
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
                            print('l. 447 uh oh')
                              
                        bb['pos'].append(lit)
                    elif 'neg' in file:
                            bb['neg'].append(lit)

    return bb

#     # cnf = open(USER_PATH + 'LLM-project/dimacs/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
#     # num_clause = int(cnf.split(' ')[-1])
#     # if num_clause > 500:
#     #     continue
#     labels[int(row[0][:-3].split('clutrr')[1])] = row[1].lower().strip(' ')
#     # print(row[0])
#     names.append(int(row[0].split('clutrr')[1].split('.py')[0]))
# names = np.unique(names)
# print(names)
# clutrr = {}
# for name in names:
#     clutrr[name] = (ds[name])
acc = 0
total = 0
preds = []
correct = []
import numpy as np
import time
    # noisy_data = ['clutrr33.cnf']
noisy_data=[]
seedrun = 'folio_2'
config = 'yes/no/maybe quicksolve (0.5 thresh), contra 0.3, contextthresh 0.3, dynamic false, , mistral7B, sc5, varname, cot_thresh ANNEALING, fixed_iter NOTFIXED'
# config = 
# thresh=0.8
# dynamic=True
# context_thresh=0.5
# ruleethresh=0.4
try:
    os.mkdir('/home/XXXX/XXXX/LLM-project/tempfiles' + str(seedrun) + '/')
    os.mkdir('/home/XXXX/XXXX/LLM-project/workfiles' + str(seedrun) + '/')
except:
    print('dir already exists')
os.environ["CUDA_VISIBLE_DEVICES"] = str(seedrun.split('_')[1])

# mistr_data = ['clutrr10.cnf']
# bad_data = ['proofd5153.cnf', 'proofd5227.cnf']
bad_data = []
mistr_data = []
c = USER_PATH + '/LLM-project/dimacs_csvs/solver_finished.csv'
import csv
import json

# /home/XXXX/XXXX/fs_backup_feb13/SAT-LM/data/folio_proofd5_test.json
dataset = USER_PATH + '/SAT-LM/data/folio_proofd5_test.json'
with open(dataset, 'r') as df:
    data = json.loads(df.read())
# breakpoint()
task = 'folio'
missed=False
c = open(c, 'r')
cr = csv.reader(c)
names = []
all_outs = {}
missed_list = []
labels = {}
for row in cr:
    if row[2] == 'SAT' and row[3] == 'SAT':
        cnf = open(USER_PATH + '/LLM-project/dimacs/neg_'+row[1]).readlines()[0].strip('\n')
        num_clause = int(cnf.split(' ')[-1])
        if row[1] in noisy_data or row[1] in mistr_data:
            print('noisy or mistranslate')
            continue
        if task=='folio':
            bb = get_bb(USER_PATH + '/LLM-project/dimacs/'+row[1])
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
        cnf = open(USER_PATH + '/LLM-project/dimacs/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
        num_clause = int(cnf.split(' ')[-1])
        if row[1] in noisy_data or row[1] in mistr_data:
            continue
        # if num_clause > 500:
            # continue
        labels[rw[0][:-2]+'cnf'] = row[1].lower()

from tqdm import tqdm
votes = {}
n_fewshot = 3
few_shot = "Here are some facts and rules: \nThere are six types of wild turkeys: Eastern wild turkey, Osceola wild turkey, Gould\’s wild turkey, Merriam\’s wild turkey, Rio Grande wild turkey, and Ocellated wild turkey.\nTom is not an Eastern wild turkey.\nTom is not an Osceola wild turkey.\nTom is not a Gould's wild turkey.\nTom is neither a Merriam's wild turkey nor a Rio Grande wild turkey.\nTom is a wild turkey.\nInstruction: Determine if following statement is true:\nTom is an eastern wild turkey.\nAnswer: Let\'s think step by step.\n1. Since Tom is a wild turkey, Tom is either an Eastern, Osceola, Gould\'s, Merriam\'s, Rio Grande or Ocellated wild turkey.\n2. Since Tom is not an Eastern, Osceola, Gould\'s, Merriam\'s or Rio Grande wild turkey, Tom is not an Eastern wild turkey.\nTherefore, the answer to the question is False.\n" + \
    '\nHere are some facts and rules: \nAll squares are four-sided.\nInstruction: Determine if following statement is true:\nAll squares are shapes.\nAnswer: Let\'s think step by step.\n1. We know that all squares are four-sided.\n2. We know that all four-sided things are shapes.\n3. Since all squares are four-sided, and all four-sided things are shapes, all squares are shapes.\nTherefore, all squares are shapes.\nTherefore, the answer to the question is True. ' + \
    "\nHere are some facts and rules: \nThe state of Montana includes the cities of Butte, Helena, and Missoula.\nWhite Sulphur Springs and Butte are cities in the same state in U.S.\nA city can only be in one state in U.S. except for Bristol, Texarkana, Texhoma and Union City.\n\nInstruction: Determine if following statement is true:\nMontana is home to the city of Missoula. Answer: Let\'s think step by step. 1. We know that the state of Montana includes the cities Butte, Helena, and Missoula.\nTherefore, Missoula is in Montana.\nTherefore, the answer to the question is True.\n"
    
answers = ['true', 'false']
import time
time_str = str(time.ctime()).replace(' ', '_').replace(':', '.')
skip_pbar=True
# breakpoint()
for z in range(0,20):
    total = 0
    acc = 0
    preds = []
    for name in (pbar := tqdm(names)):
        if name not in votes.keys():
            votes[name] = torch.tensor([1.0, 1.0])

        idx = int(name.split('proofd5')[1].split('.')[0])
        prob = data[idx]
        extr = extraction(('\n'.join(data[idx]['context'])))
        try: extr = extr.split('Problem no. 3)')[0]
        except: 1==0

        exten = extension(None, extr)
        try: exten = exten.split('Problem no. 3)')[0]
        except: 1==0
        newrules = translation(None, extr, exten)[0]
        # breakpoint()
        try: prob['newrules'] = newrules.split('Problem no. 2: \n')[1].split('*** DONE ***')[0].split('Problem no. 3')[0].split('Output:\n')[1]
        except:
            print('oopsie')
            breakpoint()
            newrules = ''

        # breakpoint()

        if 'newrules' in prob.keys():

            prompt = few_shot +  'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nHere are some additional facts and rules we\'ve found, written in logical form: \n' + (prob['newrules']) +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
        else:
            breakpoint()
            prompt = few_shot + 'Here are some facts and rules: \n' + '\n'.join(prob['context']) + '\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer: Let\'s think step by step.\n1. ' 
        # votes = torch.tensor([0,0])
        
        ans = llm.complete(prompt, max_new=1000)[0]

        # breakpoint()
        # print(prompt.strip(few_shot) + ans.strip(prompt))
        print(ans)

        ans = 'Here are some facts and rules:'.join(ans.split('Here are some facts and rules:')[:n_fewshot + 1])
        if len(ans.split('Facts')) > 7:
            ans = 'Facts'.join(ans.split('Facts')[:5])
        ans_prompt = ans + "Therefore, the answer (True/False) is "
        nli = torch.tensor(list(llm.nli(ans_prompt, False).values()))    
        # breakpoint()
        # votes[nli.argmax()] += 1
        # nli = torch.tensor(list(llm.nli(ans_prompt, unknown).values()))
        try:    
            answer = answers[torch.argmax(nli)]
        except: 
            breakpoint()
        try:
            votes[name] += nli
        except: breakpoint()
        s_probs, ind = torch.sort(nli)
        # answer = answers[ind[-1]]
        # if s_probs[-1] < torch.sum(s_probs[:-1]):
        #     answer=answers[1]
        if answer.strip(' ') == labels[name].strip(' '):
            acc += 1
        total += 1
        preds.append(answer)
        if not skip_pbar:
            pbar.set_description('Acc: ' + str(acc / len(preds)))
        skip_pbar=False 
        # print(answer, prob['label'])
        # if answer == 'Unknown':
        #     print('maybe')
        # if prob['answer'] == 'Unknown':
        #     print('maybe is correct')
        
        # run_log = open(run_log_path, 'a')
        # run_log.write(str(ans) + '\n')
        # run_log.write(str(nli) + '\n')
        # if answer == prob['label']:
        #     # run_log.write('Correct: ' + answer + '\n')
        #     acc += 1
        # else:
        #     run_log.write('Incorrect: (pred,real): ' + answer + ', ' + prob['label'] + '\n')
        # run_log.close()
        
    print(acc, '/', total)
    # run_log = open(run_log_path, 'a')
    # run_log.write('Few-Shot COT Accuracy PRONTO iter ' + str(z) + ': ' + str(acc) + '/' + str(total) + '\n')
    # run_log.close()
    file = open(USER_PATH + 'LLM-project/preds/LOT_folio_M7B_preds' + '_iter' + str(z) + '_' + str(devices), 'wb')
    np.save(file, preds)
    file.close()
    file = open(USER_PATH + 'LLM-project/preds/LOT_folio_M7B_votes' + '_iter' + str(z) + '_' + str(devices), 'wb')
    np.save(file, votes)
    file.close
    print(str(acc) + '/' + str(total))
        # breakpoint()
breakpoint()


