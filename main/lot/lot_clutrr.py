
import os
import shutil
import numpy as np
import time
import json
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

devices = [4]
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
        'n_rows': 20, 'max_length': 300,'temperature': 1, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
# args['engine'] = 'meta-llama/Meta-Llama-3-8B-Instruct'
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
        max_length=1000
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
    fewshot = 'Input: \n[Fran] went to the market with her daughter [Samantha]. [Deven] and his sister [Samantha] are on the same soccer team. [Deven] and his father [James] got in a fight. \n' + \
             'Output:\nPropositions:\nA: daughter(Fran, Samantha)\nB:sister(Deven, Samantha)\nC: father(Deven, James)\nLogical Extraction of Context (purely symbolic):\n' + \
             'A \nB\nC *** DONE ***'

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
    fewshot = 'Input:  Here is a list of symbol translations to use when translating the expressions: \n ' + 'A: daughter(Fran, Samantha)\nB:sister(Deven, Samantha)\nC: father(Deven, James)\n' + 'Here is the list of expressions to be translated:' + '\n ¬B → ¬A\nA → C \n' + \
            'Output: \nIf Samantha is not Deven\'s sister then Samantha is not Fran\'s daughter.\nIf Samantha is Fran\'s daughter then James is Deven\'s father.\n*** DONE ***\n'
    return llm.complete('Problem no. 1: \n' + fewshot + 'Problem no. 2: \n' + prompt + 'Input:  Here is a list of symbol translations to use when translating the expressions: \n' +  extraction.split('\nPropositions:')[1].split('Negative Propositions')[0] + '\nHere is the list of expressions to be translated:\n' + '\n'.join(extension.split('Output:')[2].split('\n*** DONE ***')[0].split('\n')[1:]) + '\nOutput:', max_new=300)

llm = LLM()


dataset = USER_PATH + '/SAT-LM/data/clutrr_test.json'
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
c = USER_PATH + '/LLM-project/dimacs_clutrr_csvs/solver_finished.csv'
import csv
import json
noisy_data = []
mistr_data = []
seedrun = 'clutrr_4'
import random
dataset = USER_PATH + '/SAT-LM/data/clutrr_test.json'
names = []
# print(len(ds))
labels = {}
c = open(USER_PATH + 'LLM-project/clutrr_labels.csv', 'r')
cr = csv.reader(c)
for row in cr:
    # cnf = open(USER_PATH + 'LLM-project/dimacs/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
    # num_clause = int(cnf.split(' ')[-1])
    # if num_clause > 500:
    #     continue
    labels[int(row[0][:-3].split('clutrr')[1])] = row[1].lower().strip(' ')
    # print(row[0])
    names.append(int(row[0].split('clutrr')[1].split('.py')[0]))
names = np.unique(names)




# print(names)
clutrr = json.load(open(USER_PATH + '/SAT-LM/data/clutrr_test.json', 'r'))
# clutrr = {}
# for name in names:
#     clutrr[name] = (ds[name])
acc = 0
total = 0
preds = []
correct = []
import numpy as np
import time
noisy_data = []
mistr_data = []
c = USER_PATH + '/LLM-project/dimacs_clutrr_csvs/solver_finished.csv'
import csv
import json
dataset = USER_PATH + '/SAT-LM/data/clutrr_test.json'
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
        try:        
            cnf = open(USER_PATH + '/LLM-project/dimacs_clutrr/neg_'+row[1]).readlines()[0].strip('\n')
        except:
            breakpoint()
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
    c = open(USER_PATH + '/LLM-project/clutrr_labels.csv', 'r')
    cr = csv.reader(c)
    for row in cr:
        if not os.path.exists(USER_PATH + '/LLM-project/dimacs_clutrr/neg_'+row[0][:-2]+'cnf'):
            continue
        cnf = open(USER_PATH + '/LLM-project/dimacs_clutrr/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
        num_clause = int(cnf.split(' ')[-1])
        if row[1] in noisy_data or row[1] in mistr_data:
            continue
        # if num_clause > 500:
            # continue
        labels[row[0][:-2]+'cnf'] = row[1].lower()


for idx in list(names):
    # try
    int_idx = int(idx.split('clutrr')[1].split('.')[0])
    n1 = clutrr[int_idx]['query'].split('(\'')[1].split('\',')[0]
    n2 = clutrr[int_idx]['query'].split(', \'')[1].split('\')')[0]
    
    # n1, n2 = clutrr[idx]['query'][0], clutrr[idx]['query'][1]
    # print(n1, n2)
    # except:
    #     print(clutrr[idx]['query'])
    relationship = clutrr[int_idx]['label']
    # print(labels[idx])
    # print(labels[idx]=='true')
    # if labels[idx].strip(' ') == 'true':
    #     clutrr[int_idx]['question'] = 'Is ' + n2 + ' ' + n1 + '\'s ' + relationship + '?'
    #     # print(clutrr[idx]['question'])
    #     # print('hihi')
    # elif labels[idx].strip(' ') == 'false':
    #     clutrr[int_idx]['question'] = 'Is ' + n2 + ' not ' + n1 + '\'s ' + relationship + '?'
    clutrr[int_idx]['question'] = 'Is ' + n2 + ' ' + n1 + '\'s ' + relationship + '?'


from tqdm import tqdm
votes = {}
n_fewshot = 4
# few_shot = "Facts: [Nancy] likes to cut the hair of her daughter [Heidi]. [Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
#             "Question: Is [Lorraine] [Nancy]\'s daughter? Answer: Let\'s think step by step. 1. [Lorraine] is [Heidi]\'s sister. 2. [Heidi] is [Nancy]\'s daughter. " + \
#             "3. [Lorraine] is [Nancy]\'s daughter. Therefore, the answer to the question is Yes. \n" + \
#             "Facts: [Dale] and his sister [Nancy] are decorating for a party. [Nancy]'s daughter [Louise] thinks the party will be fun. Question: Is [Louise] not [Dales]\'s niece? " + \
#             "Answer: Let\'s think step by step. 1. [Louise] is [Nancy]\'s daughter. 2. [Nancy] is [Dale]\'s sister. 3. [Louise] is [Dale]\'s niece. Therefore, the answer to the question is No. \n" + \
#             "Facts: [Lillian] and her sister [Nancy] are the only children in their family. [Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
#             "Question: Is [Douglas] [Nancy]\'s nephew? Answer: Let\'s think step by step. 1. [Douglas] is [Lillian]\'s son. 2. [Nancy] is [Lillian]\'s sister. " + \
#             "3. [Douglas] is [Nancy]\'s nephew. Therefore, the answer to the question is Yes. \n" + \
#             "Facts: [Ashley] liked to go to the park with her granddaughter [Charlotte]. [Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
#             "Question: Is [Ashley] not [Dale]\'s mother? Answer: Let\'s think step by step. 1. [Ashley] is [Charlotte]\'s grandmother. 2. [Charlotte] is [Dale]\'s daughter. " + \
#             "3. [Ashley] is [Dale]\'s mother. Therefore, the answer to the question is No. \n"
few_shot = "Facts: [Nancy] likes to cut the hair of her daughter [Heidi]. [Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
            "Question: Is [Lorraine] [Nancy]\'s daughter? Answer: Let\'s think step by step. 1. [Lorraine] is [Heidi]\'s sister. 2. [Heidi] is [Nancy]\'s daughter. " + \
            "3. [Lorraine] is [Nancy]\'s daughter. Therefore, the answer to the question is Yes. \n" + \
            "Facts: [Dale] and his sister [Nancy] are decorating for a party. [Nancy]'s daughter [Louise] thinks the party will be fun. Question: Is [Louise] [Dales]\'s sister? " + \
            "Answer: Let\'s think step by step. 1. [Louise] is [Nancy]\'s daughter. 2. [Nancy] is [Dale]\'s sister. 3. [Louise] is [Dale]\'s niece. Therefore, the answer to the question is No. \n" + \
            "Facts: [Lillian] and her sister [Nancy] are the only children in their family. [Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
            "Question: Is [Douglas] [Nancy]\'s nephew? Answer: Let\'s think step by step. 1. [Douglas] is [Lillian]\'s son. 2. [Nancy] is [Lillian]\'s sister. " + \
            "3. [Douglas] is [Nancy]\'s nephew. Therefore, the answer to the question is Yes. \n" + \
            "Facts: [Ashley] liked to go to the park with her granddaughter [Charlotte]. [Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
            "Question: Is [Ashley] [Dale]\'s aunt? Answer: Let\'s think step by step. 1. [Ashley] is [Charlotte]\'s grandmother. 2. [Charlotte] is [Dale]\'s daughter. " + \
            "3. [Ashley] is [Dale]\'s mother. Therefore, the answer to the question is No. \n"
answers = ['true', 'false']
import time
time_str = str(time.ctime()).replace(' ', '_').replace(':', '.')
skip_pbar=True
for z in range(8,9):
    total = 0
    acc = 0
    preds = []
    for name in (pbar := tqdm(names)):
        if name not in votes.keys():
            votes[name] = torch.tensor([1.0, 1.0])

        idx = int(name.split('clutrr')[1].split('.')[0])
        prob = clutrr[idx]
        extr = extraction((data[idx]['context']))
        exten = extension(None, extr)
        newrules = translation(None, extr, exten)[0]
        try: newrules = newrules.split('Problem no. 2: \n')[1].split('*** DONE ***')[0].split('Output: \n')[1]
        except:
            print('oopsie')
            # breakpoint()
            newrules = ''

        # breakpoint()

        prompt = few_shot + 'Facts: ' + prob['context']  + ''.join(newrules.split('\n')) + ' Question: ' + prob['question'].strip('.') + ' Answer: Let\'s think step by step.' 
        # print()
        # breakpoint()
        # print(prompt)

        ans = llm.complete(prompt, max_new=1000)[0]
        try: print('Facts: ' + prob['context']  + ''.join(newrules.split('\n')) + ' Question: ' + prob['question'].strip('.') + ' Answer: Let\'s think step by step.' + ans[len(prompt):])
        except: breakpoint()
        # if len(ans.split('Facts')) > 7:
        #     ans = 'Facts'.join(ans.split('Facts')[:5])
        ans_prompt = ans + "Therefore, the answer (True/False) is "

        ans_prompt = 'Facts:' + ans.split('Facts:')[n_fewshot+1] + " Therefore, the answer (True/False) is "
        # print(ans_prompt)
        # breakpoint()
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
        try: 
            m = ans_prompt.split('Therefore')[1]
            if 'Yes' in m: 
                nli = torch.tensor([1,0])
                # print('[1,0]')
                # print(ans_prompt)
                # continue
    
            elif 'No' in m: 
                nli = torch.tensor([0,1])
                # print('[0,1]')
                # print(ans_prompt)
                # continue
            else: 
                nli = torch.tensor(list(llm.nli(ans_prompt, False).values())).softmax(-1)
            # votes += nli.softmax(-1)
        except: 
            nli = torch.tensor(list(llm.nli(ans_prompt, False).values())).softmax(-1)
        # nli = torch.tensor(list(llm.nli(ans_prompt, unknown).values()))    
        answer = answers[torch.argmax(nli)]
        votes[name] += nli
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
    file = open(USER_PATH + 'LLM-project/preds/LOT_clutrr_8B_preds' + '_iter' + str(z) + '_' + str(devices), 'wb')
    np.save(file, preds)
    file.close()
    file = open(USER_PATH + 'LLM-project/preds/LOT_clutrr_8B_votes' + '_iter' + str(z) + '_' + str(devices), 'wb')
    np.save(file, votes)
    file.close
    print(str(acc) + '/' + str(total))
        # breakpoint()
breakpoint()

