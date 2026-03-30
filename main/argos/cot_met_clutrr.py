import os
import shutil
import numpy as np
import time
import json
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

import torch
from torch.utils.data import DataLoader
USER_PATH = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/'

os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
# os.environ['TRANSFORMERS_CACHE'] = USER_PATH + '/.cache/huggingface/hub'
# cache_dir = '/ephemeral/media/data1/XXXX/hub/'
cache_dir = os.path.join(os.getcwd(), '.cache/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
# os.environ['HF_HUB_OFFLINE'] ='1'
import pickle as pkl

# import transformers

# import urllib3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from tqdm import tqdm
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



class LLM():
    def __init__(self, args):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        with no_ssl_verification():
            self.tokenizer = AutoTokenizer.from_pretrained(
                    args.engine,
                    cache_dir=cache_dir,
                    token=os.getenv("HF_TOKEN"),
                    )
            # Llama 3.x uses a different pad token setup — set padding side to left
            # so batch inference doesn't corrupt attention on the right-hand tokens
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(
                    args.engine,
                    cache_dir=cache_dir,
                    quantization_config=quant_config,
                    device_map='auto',
                    token=os.getenv("HF_TOKEN"),
                    attn_implementation="sdpa",
                    )
            self.model.config.use_cache = True
    
    @property
    def device(self):
        """Returns the device the model is actually on, derived from its parameters."""
        return next(self.model.parameters()).device

    _kv_cache_reported = False  # print KV-cache status only once

    def sentence_probabilities(self, sentences):
        with torch.no_grad():
            sentence_tokens = self.tokenizer(sentences, return_tensors='pt', padding=True)
            sentence_token_ids = sentence_tokens.input_ids.to(self.device)
            attention_mask = sentence_tokens.attention_mask.to(self.device)

            # Optimisation: find the common prefix and run it once, then reuse the
            # KV-cache for the diverging suffixes (4-5x faster when it works).
            first_different_token = (sentence_token_ids == sentence_token_ids[0, :].unsqueeze(0)).all(dim=0).long().argmin()
            common_prefix = sentence_token_ids[0, :first_different_token].unsqueeze(0)
            common_prefix_output = self.model(common_prefix, use_cache=True)
            pkv = common_prefix_output.past_key_values

            # Try to expand the KV-cache for all sentences in the batch.
            # Handles both the old tuple-of-tuples format and the newer DynamicCache
            # used by Llama 3.x / Mistral 3.x. Falls back if anything is None.
            expanded_pkv = None
            try:
                from transformers.cache_utils import DynamicCache
                import transformers
                transformers_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])

                if isinstance(pkv, DynamicCache):
                    if transformers_version >= (4, 43):
                        # New API: key_cache / value_cache lists
                        if pkv.key_cache and all(k is not None and v is not None
                                for k, v in zip(pkv.key_cache, pkv.value_cache)):
                            new_cache = DynamicCache()
                            for k, v in zip(pkv.key_cache, pkv.value_cache):
                                new_cache.key_cache.append(k.expand(len(sentences), -1, -1, -1).contiguous())
                                new_cache.value_cache.append(v.expand(len(sentences), -1, -1, -1).contiguous())
                            expanded_pkv = new_cache
                    else:
                        # Old API: iterate over layers directly as (key, value) tuples
                        layers = list(pkv)
                        if layers and all(t is not None for layer in layers for t in layer):
                            expanded_pkv = tuple(
                                tuple(t.expand(len(sentences), -1, -1, -1).contiguous() for t in layer)
                                for layer in layers
                            )
                elif pkv is not None:
                    # Legacy pure tuple-of-tuples format
                    if all(t is not None for layer in pkv for t in layer):
                        expanded_pkv = tuple(
                            tuple(t.expand(len(sentences), -1, -1, -1).contiguous() for t in layer)
                            for layer in pkv
                        )
            except Exception as e:
                print(f'KV-cache expansion failed: {e}')
                expanded_pkv = None

            if expanded_pkv is not None:
                if not LLM._kv_cache_reported:
                    cache_type = type(pkv).__name__
                    print(f"[KV-cache] ACTIVE — using {cache_type} fast-path (prefix shared, ~4-5x faster)")
                    LLM._kv_cache_reported = True
                rest_outputs = self.model(
                    sentence_token_ids[:, first_different_token:],
                    past_key_values=expanded_pkv
                )
                logits = torch.concat(
                    [common_prefix_output.logits.expand(len(sentences), -1, -1),
                     rest_outputs.logits], dim=1
                ).to(self.device)
            else:
                if not LLM._kv_cache_reported:
                    reason = "unknown"
                    if pkv is None:
                        reason = "model returned None past_key_values"
                    else:
                        reason = f"{type(pkv).__name__} contains None tensors (sliding window / GQA)"
                    print(f"[KV-cache] DISABLED — {reason}. Using slow full forward pass.")
                    LLM._kv_cache_reported = True
                # Slow path: full forward pass for every sentence
                full_outputs = self.model(sentence_token_ids, attention_mask=attention_mask)
                logits = full_outputs.logits.to(self.device)

            log_probs = logits.log_softmax(-1)
            log_probs = log_probs[:, :-1, :].gather(2, sentence_token_ids[:, 1:][:, :, None]).squeeze(-1).to(self.device)
            log_probs = (log_probs * attention_mask[:, 1:]).sum(-1).cpu()
        return log_probs
    def nli(self, sentences, unknown):
        # true_probs = self.sentence_probabilities(sentences + " True.")
        # false_probs = self.sentence_probabilities(sentences + " False.")
        # maybe_probs = self.sentence_probabilities(sentences + " Maybe.")
        if unknown:
            true_probs, maybe_probs, false_probs =  (self.sentence_probabilities([sentences + "(A)", sentences + "(B)", sentences + "(C)"]))
            return {'True': true_probs, 'Maybe': maybe_probs, 'False': false_probs}
        else:
            true_probs, false_probs =  (self.sentence_probabilities([sentences + "(A)", sentences + "(B)"]))
            return {'True': true_probs, 'False': false_probs}
    def yn(self, sentences, norm=True, relaxed=False, obvious=False, unknown=False, fewshot=None, maybe=False):
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
            elif unknown:
                # print('unknown')
                yns.append(sentence + "True")
                yns.append(sentence + "Unknown")
                yns.append(sentence + "False")
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
        if maybe or unknown:
            z = 3
        else:
            z = 2
        for i in range(0,len(probs), z):
            # if yns[i] not in cache.keys():
                # yes, no = self.sentence_probabilities([yns[i], yns[i+1]])
            
            if maybe or unknown:
                
                yes, maybe, no = probs[i], probs[i+1], probs[i+2]
                
                      
            else:
                yes, no = probs[i], probs[i+1]
            if norm:
                if maybe or unknown: 
                    y,m,n = torch.tensor([yes, maybe, no]).softmax(-1)
                else:
                    y,n = torch.tensor([yes, no]).softmax(-1)
              
                # cache[yns[i]] = y
                # cache[yns[i+1]] = n
                pyes.append(y)
                pno.append(n)
                if maybe or unknown:
                    pmaybe.append(m)
            else:
                pyes.append(1-yes/(yes + no))
            # else:
            #     y, n = cache[yns[i]], cache[yns[i+1]]
            #     pyes.append(y)
                # pno.append(n)/
        # print('cache length', len(cache))
        # if maybe:
        
        if maybe:
            return torch.stack([torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)])
        if unknown:
            return [torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)]
        return [torch.tensor(pyes), torch.tensor(pno)]
    def complete(self, prompt, max_new=25, temp=1, topk=0, max_length=300):
        encoded = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        encode_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)
        generated_outputs = self.model.generate(
            encode_ids,
            attention_mask=attn_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=(temp > 0 and topk > 0),
            temperature=temp if (temp > 0 and topk > 0) else None,
            top_k=topk if topk > 0 else None,
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
            
            # ---------------------------------------------------------------------------------------------------------
            # SAT SOLVER EXECUTION: CaDiCaL
            # This calls the externally compiled CaDiCaL SAT solver binary via command line.
            # CaDiCaL reads the provided .cnf file and looks for a valid 1/0 assignment to variables that 
            # makes the entire boolean formula TRUE.
            # It dumps the output (satisfiable/unsatisfiable and the solution) into a .log file.
            # ---------------------------------------------------------------------------------------------------------
            log_pth = '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1])[:-4] + '.log'
            print(f"WRITING TO TEMPFILES (CaDiCaL Log): {log_pth}")
            os.system(USER_PATH + '/sat_gen/sat_tools/postprocess/cadical/build/cadical ' + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]) + '> ' + log_pth)
            
            cf = open( log_pth, 'r')

            lines = cf.readlines()

            el = lines[-1]
            print(f"EL: {el}")
            try:
                print(f"try EL: {el}")
                ec = el.split('exit ')[1].strip('\n')
            except:
                print(f"except EL: {el}")
                breakpoint()
            # lf.close()
            if ec == '20':
                print('UNSATISFIABLE, error code 20')
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
        
        # ---------------------------------------------------------------------------------------------------------
        # SAT SOLVER EXECUTION: CadiBack
        # This calls 'cadiback', a custom backbone extractor built on top of CaDiCaL.
        # It doesn't just find ONE solution; it finds the LOGICAL BACKBONE: the set of logic literals 
        # that must be assigned the exact same truth value across ALL possible valid solutions.
        # It pipes this list of certainties out to a .bbone file which the python code will read.
        # ---------------------------------------------------------------------------------------------------------
        print('running cadical')
        bbone_pth = '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/'+ str(file.split('/')[-1])[:-4] + ".bbone"
        print(f"WRITING TO TEMPFILES (CadiBack bbone): {bbone_pth}")
        os.system("timeout 5000 " + USER_PATH + "/main/cadiback/cadiback " + '/'.join(file.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + str(file.split('/')[-1]) + '> '  + bbone_pth)
        #   
        bbone= open(bbone_pth, 'r')
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

def rule_check(rule, contra_thresh=0.3, context_thresh=0.3, prob='', question = '',llm=None, ablate=False):
    if ablate: 
        print('ablating rulecheck')
        return True, [0,0]
    contra_fs = 'Does the following rule seem true?\nRule: If Nathan is the father of Billy and Billy is the brother of Mary then Mary is Nathan\'s daughter.\nAnswer: \\box{ Yes } this seems true.' \
        + '\nDoes the following rule seem true?\nRule: If Cassandra is the niece of Frank and Frank is the brother of Sam then Sam is the nephew of Cassandra. \nAnswer: \\box{ No } this does not seems true. \n'
    a = llm.yn([contra_fs + 'Does the following rule seem true?\nRule: ' + rule + '\nAnswer: \\box{ '])[0]
    # a = llm.yn(['Does the following rule seem plausible?\nRule: ' + rule + '\nAnswer:'])
    # breakpoint()
    # print(r)
    b = llm.yn(['Here are some facts and rules\n' + prob['context'] + '\nDoes the following new rule seem contextually relevant to the facts and rules?\nRule: ' + rule + '\nAnswer \"Yes\" or \"No\" here: '])[0]

    if a < contra_thresh:
        # breakpoint()
        print('rejected by contradiction:', rule, a)
        return False, [a,b]
    if b < context_thresh:
        # breakpoint()
        print('rejected by context:', rule, b)
        return False, [a,b]
    return True, [a,b]

def cot(prob, n=5, jbprompt=False):
    # prob = folio[name]
    n_fewshot = 4

    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #             "\nHere are some additional facts and rules we\'ve found:\n[Nancy] is the mother of [Lorraine]\n If [Heidi] is the sister of [Lorraine] and [Heidi] is the daughter of [Nancy] then [Nancy] is the mother of [Lorraine].\n" + \
    #             "Instruction: Determine if following statement is true: \n\"[Lorraine] is [Nancy]\'s daughter\"\nAnswer:\nLet\'s think step by step. \n1. We have already found that [Nancy] is the mother of [Lorraine].\n2. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #             "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #             "Here are some additional facts and rules we\'ve found:\n[Dale] is the uncle of [Louise]. If Nancy is the sister of Dale and Nancy is the mother of Louise then Dale is the uncle of Louise.\n" + \
    #             "Instruction: Determine if following statement is true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #             "Answer:\nLet\'s think step by step. 1. We are given that [Dale] is the uncle of [Louise].\n2.If [Dale] is the uncle of [Louise], then [Louise] is the niece of [Dale].\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #             "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #             "\nHere are some additional facts and rules we\'ve found:\n[Lillian] is the sister of [Nancy]. \nIf [Nancy] is the sister if [Lillian] then [Lillian] is the sister of [Nancy].\n" + \
    #             "Instruction: Determine if following statement is true: \n\"[Douglas] is [Nancy]\'s nephew\"\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #             "3\n. [Douglas] is [Nancy]\'s nephew. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #             "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #             "\nHere are some additional facts and rules we\'ve found:\n[Dale] is the son of [Ashley]. If [Dale] is father of [Charlotte] and [Ashley] is the grandmother of [Charlotte] then [Dale] is the son of [Ashley].\n" + \
    #             "Instruction: Determine if following statement is true: \n\"[Ashley] is not [Dale]\'s mother\"\nAnswer:\nLet\'s think step by step. \n1. We are given that [Dale] is the son of [Ashley]. \n2. If [Dale] is the son of [Ashley], then [Ashley] is the mother of [Dale]. " + \
    #             "\nTherefore, the answer to the question is No, the statement is ot true.\n***\n" + \
                
    ## no separation, no rules in prompt
    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "[Nancy] is the mother of [Lorraine].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Lorraine] is [Nancy]\'s daughter\"\nAnswer:\nLet\'s think step by step. \n1. We have already found that Nancy is the mother of Lorraine.\n2. If Nancy is the mother of Lorraine, then Lorraine is the daughter of Nancy.\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #         "[Dale] is the uncle of [Louise].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer:\nLet\'s think step by step. 1. We are given that Dale is the uncle of Louise.\n2.If Dale is the uncle of Louise, then Louise is the niece of Dale.\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "[Lillian] is the sister of [Nancy].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Douglas] is [Nancy]\'s nephew\"\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "3\n. [Douglas] is [Nancy]\'s nephew. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "[Dale] is the son of [Ashley].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Ashley] is not [Dale]\'s mother\"\nAnswer:\nLet\'s think step by step. \n1. We are given that Dale is the son of Ashley. \n2. If Dale is the son of Ashley, then Ashley is the mother of Dale. " + \
    #         "\nTherefore, the answer to the question is No, the statement is ot true.\n***\n"
    # no separation, no rules in prompt v2
    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "[Nancy] is the mother of [Lorraine].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Lorraine] is [Nancy]\'s daughter\"\nAnswer:\nLet\'s think step by step. \n1. We have already found that [Nancy] is the mother of [Lorraine].\n2. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #         "[Dale] is the uncle of [Louise].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. We are given that [Dale] is the uncle of [Louise].\n2.If [Dale] is the uncle of [Louise], then [Louise] is the niece of [Dale].\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "[Lillian] is the sister of [Nancy].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Douglas] is [Nancy]\'s nephew\"\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "\n3. [Douglas] is [Nancy]\'s nephew. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "[Dale] is the son of [Ashley].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Ashley] is not [Dale]\'s mother\"\nAnswer:\nLet\'s think step by step. \n1. We are given that [Dale] is the son of [Ashley]. \n2. If [Dale] is the son of [Ashley], then [Ashley] is the mother of [Dale]. " + \
    #         "\n3. Since [Ashley] is the mother of [Dale], the statement \"[Ashley] is not [Dale]\'s mother\" is false.\nTherefore, the answer to the question is No, the statement is not true.\n***\n" + \
    #         "Facts:\n[David] likes to go to the mall with his son [George].\n[George] and his mom [Caroline] went to the movies.\n[Francine] is the daughter of [Kevin].\n" + \
    #         "Instruction: Determine if the following statement is true: \n\"[Francine] is [David]\'s cousin\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Francine] is the daughter of [Kevin].\n2.[Kevin] has no known relation to [David] or any of his relatives.\nTherefore, we can not know if the statement is true, so the answer is Maybe.\n***\n"
    # ## no separation, no rules in prompt v2
    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "[Nancy] is the mother of [Lorraine].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Lorraine] is [Nancy]\'s daughter\"\nAnswer:\nLet\'s think step by step. \n1. We have already found that [Nancy] is the mother of [Lorraine].\n2. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #         "[Dale] is the uncle of [Louise].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. We are given that [Dale] is the uncle of [Louise].\n2.If [Dale] is the uncle of [Louise], then [Louise] is the niece of [Dale].\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "[Lillian] is the sister of [Nancy].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Douglas] is [Nancy]\'s nephew\"\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "\n3. [Douglas] is [Nancy]\'s nephew. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "[Dale] is the son of [Ashley].\n" + \
    #         "Instruction: Determine if following statement is true: \n\"[Ashley] is not [Dale]\'s mother\"\nAnswer:\nLet\'s think step by step. \n1. We are given that [Dale] is the son of [Ashley]. \n2. If [Dale] is the son of [Ashley], then [Ashley] is the mother of [Dale]. " + \
    #         "\n3. Since [Ashley] is the mother of [Dale], the statement \"[Ashley] is not [Dale]\'s mother\" is false.\nTherefore, the answer to the question is No, the statement is not true.\n***\n" + \
    #         "Facts:\n[David] likes to go to the mall with his son [George].\n[George] and his mom [Caroline] went to the movies.\n[Francine] is the daughter of [Kevin].\n" + \
    #         "Instruction: Determine if the following statement is true: \n\"[Francine] is [David]\'s cousin\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Francine] is the daughter of [Kevin].\n2.[Kevin] has no known relation to [David] or any of his relatives.\nTherefore, we can not know if the statement is true, so the answer is Unknown.\n***\n"
    
    # new few_shot with rules
    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "\nHere are some additional facts and rules we\'ve found:\nNancy is the mother of Lorraine\n If [Heidi] is the sister of [Lorraine] and [Heidi] is the daughter of [Nancy] then [Nancy] is the mother of [Lorraine].\n" + \
    #         "Question: Is the following statement true: \n\"[Lorraine] is [Nancy]\'s daughter\"\n" + \
    #         "Answer:\nLet\'s think step by step.  \n1. [Heidi] is the sister of [Lorraine]\n2. [Heidi] is the daughter of [Nancy]\n3. If [Heidi] is the sister of [Lorraine] and [Heidi] is the daughter of [Nancy] then [Nancy] is the mother of [Lorraine].\n4. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #         "Here are some additional facts and rules we\'ve found:\nDale is the uncle of Louise. If [Nancy] is the sister of [Dale] and [Nancy] is the mother of [Louise] then [Dale] is the uncle of [Louise].\n" + \
    #         "Question: Is the following statement true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer: Le\'s think step by step. \n1. [Nancy] is the sister of [Dale]. \n2. [Nancy] is the mother of [Louise]\n3.  If [Nancy] is the sister of [Dale] and [Nancy] is the mother of [Louise] then [Dale] is the uncle of [Louise].\n4.If [Dale] is the uncle of [Louise], then [Louise] is the niece of [Dale].\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "\nHere are some additional facts and rules we\'ve found:\n[Lillian] is the sister of [Nancy]. \nIf [Nancy] is the sister if [Lillian] then [Lillian] is the sister of [Nancy].\n" + \
    #         "Question: Is the following statement true: \n\"[Douglas] is [Nancy]\'s nephew\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "\n3. If [Douglas] is the son of [Lillian] and [Lillian] is the sister of [Nancy] then [Douglas] is the nephew of [Lillian]. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "\nHere are some additional facts and rules we\'ve found:\n[Dale] is the son of [Ashley]. If [Dale] is father of [Charlotte] and [Ashley] is the grandmother of [Charlotte] then [Dale] is the son of [Ashley].\n" + \
    #         "Question: Is the following statement true: \n\"[Ashley] is not [Dale]\'s mother\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Dale] is the father of [Charlotte].\n2. [Ashley] is the grandmother of [Charlotte]. \n3. If [Dale] is father of [Charlotte] and [Ashley] is the grandmother of [Charlotte] then [Dale] is the son of [Ashley].\n4. If [Dale] is the son of [Ashley], then [Ashley] is the mother of [Dale]. " + \
    #         "\nTherefore, the answer to the question is No, the statement is ot true.\n***\n"

    # new few_shot without rules
    # few_shot = "Facts:\n[Nancy] likes to cut the hair of her daughter [Heidi].\n[Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "\nHere are some additional facts we\'ve found:\n[Nancy] is the mother of [Lorraine]\n" + \
    #         "Is the following statement true?: \n\"[Lorraine] is [Nancy]\'s daughter\"\n" + \
    #         "Answer:\nLet\'s think step by step.  \n1. [Heidi] is the sister of [Lorraine]\n2. [Heidi] is the daughter of [Nancy]\n3. If [Heidi] is the sister of [Lorraine] and [Heidi] is the daughter of [Nancy] then [Nancy] is the mother of [Lorraine].\n4. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\n" + \
    #         "Therefore, the statement is \\box{True}. \n***\n" + \
    #         "Facts:\n[Dale] and his sister [Nancy] are decorating for a party.\n[Nancy]'s daughter [Louise] thinks the party will be fun.\n" + \
    #         "Here are some additional facts we\'ve found:\n[Dale] is the uncle of [Louise]\n" + \
    #         "Is the following statement true?:\n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer: Le\'s think step by step. \n1. [Nancy] is the sister of [Dale]. \n2. [Nancy] is the mother of [Louise]\n3.  If [Nancy] is the sister of [Dale] and [Nancy] is the mother of [Louise] then [Dale] is the uncle of [Louise].\n4.If [Dale] is the uncle of [Louise], then [Louise] is the niece of [Dale].\n" + \
    #         "Therefore, the statement is \\box{False}.\n***\n" + \
    #         "Facts: \n[Lillian] and her sister [Nancy] are the only children in their family. \n[Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "\nHere are some additional facts we\'ve found:\n[Lillian] is the sister of [Nancy]\n" + \
    #         "Is the following statement true?:\n\"[Douglas] is [Nancy]\'s nephew\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "\n3. If [Douglas] is the son of [Lillian] and [Lillian] is the sister of [Nancy] then [Douglas] is the nephew of [Lillian]. \n" + \
    #         "Therefore, the statement is \\box{True}. \n***\n" + \
    #         "Facts: \n[Ashley] liked to go to the park with her granddaughter [Charlotte]. \n[Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "\nHere are some additional facts we\'ve found:\n[Dale] is the son of [Ashley].\n" + \
    #         "Is the following statement true?:\n\"[Ashley] is not [Dale]\'s mother\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Dale] is the father of [Charlotte].\n2. [Ashley] is the grandmother of [Charlotte]. \n3. If [Dale] is father of [Charlotte] and [Ashley] is the grandmother of [Charlotte] then [Dale] is the son of [Ashley].\n4. If [Dale] is the son of [Ashley], then [Ashley] is the mother of [Dale]. " + \
    #         "\nTherefore, the statement is \\box{False}.\n***\n" + \
    #         "Facts:\n[David] likes to go to the mall with his son [George].\n[George] and his mom [Caroline] went to the movies.\n[Francine] is the daughter of [Kevin].\n" + \
    #         "Here are some additional facts we\'ve found:\n[Kevin] is the father of [Francine]." + \
    #         "\nInstruction: Determine if the following statement is true: \n\"[Francine] is [David]\'s cousin\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. [Francine] is the daughter of [Kevin].\n2.[Kevin] has no known relation to [David] or any of his relatives.\n" + \
    #         "Therefore, the truth of the statement is \\box{Unknown}.\n***\n"
    #og prompt
    # few_shot = "Facts: [Nancy] likes to cut the hair of her daughter [Heidi]. [Heidi]'s sister [Lorraine] went to beauty school and taught them all how to cut hair expertly. " + \
    #         "Question: Is [Lorraine] [Nancy]\'s daughter? Answer: Let\'s think step by step. 1. [Lorraine] is [Heidi]\'s sister. 2. [Heidi] is [Nancy]\'s daughter. " + \
    #         "3. [Lorraine] is [Nancy]\'s daughter. Therefore, the answer to the question is Yes. \n" + \
    #         "Facts: [Dale] and his sister [Nancy] are decorating for a party. [Nancy]'s daughter [Louise] thinks the party will be fun. Question: Is [Louise] not [Dales]\'s niece? " + \
    #         "Answer: Let\'s think step by step. 1. [Louise] is [Nancy]\'s daughter. 2. [Nancy] is [Dale]\'s sister. 3. [Louise] is [Dale]\'s niece. Therefore, the answer to the question is No. \n" + \
    #         "Facts: [Lillian] and her sister [Nancy] are the only children in their family. [Lillian]'s biggest accomplishment is raising her son [Douglas]. " + \
    #         "Question: Is [Douglas] [Nancy]\'s nephew? Answer: Let\'s think step by step. 1. [Douglas] is [Lillian]\'s son. 2. [Nancy] is [Lillian]\'s sister. " + \
    #         "3. [Douglas] is [Nancy]\'s nephew. Therefore, the answer to the question is Yes. \n" + \
    #         "Facts: [Ashley] liked to go to the park with her granddaughter [Charlotte]. [Dale], [Charlotte]'s father, like to take her to the movies instead. " + \
    #         "Question: Is [Ashley] not [Dale]\'s mother? Answer: Let\'s think step by step. 1. [Ashley] is [Charlotte]\'s grandmother. 2. [Charlotte] is [Dale]\'s daughter. " + \
    #         "3. [Ashley] is [Dale]\'s mother. Therefore, the answer to the question is No. \n"
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
    # ##jbprompt fewshot
    # few_shot = "Facts:\n[Nancy] is the daughter of [Heidi].\n[Lorraine] is the sister of [Heidi]\n." + \
    #         "\n\nHere are some additional facts and rules we\'ve found:\n[Nancy] is the mother of [Lorraine]\n If Heidi is the sister of Lorraine and Heidi is the daughter of Nancy then Nancy is the mother of Lorraine.\n" + \
    #         "Question: Is the following statement true: \n\"[Lorraine] is [Nancy]\'s daughter\"\nAnswer:\nLet\'s think step by step. \n1. We have already found that Nancy is the mother of Lorraine.\n2. If Nancy is the mother of Lorraine, then Lorraine is the daughter of Nancy.\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Nancy] is the sister of [Dale].\n[Louise] is the daughter of [Nancy].\n" + \
    #         "\nHere are some additional facts and rules we\'ve found:\nDale is the uncle of Louise. \nIf Nancy is the sister of Dale and Nancy is the mother of Louise then Dale is the uncle of Louise.\n" + \
    #         "Question: Is the following statement true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "Answer:\nLet\'s think step by step. \n1. We are given that Dale is the uncle of Louise.\n2.If Dale is the uncle of Louise, then Louise is the niece of Dale.\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Nancy] is the sister of [Lillian]. \n[Douglas] is the son of [Lillian]. " + \
    #         "\n\nHere are some additional facts and rules we\'ve found:\nLillian is the sister of Nancy. \nIf Nancy is the sister if Lillian then Lillian is the sister of Nancy.\n" + \
    #         "Question: Is the following statement true: \n\"[Douglas] is [Nancy]\'s nephew\"\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "3\n. [Douglas] is [Nancy]\'s nephew. \nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Charlotte] is the granddaughter of [Ashley].\n[Dale] is the father of [Charlotte]." + \
    #         "\n\nHere are some additional facts and rules we\'ve found:\nDale is the son of Ashley. If Dale is father of Charlotte and Ashley is the grandmother of Charlotte then Dale is the son of Ashley.\n" + \
    #         "Question: Is the following statement true: \n\"[Ashley] is not [Dale]\'s mother\"\nAnswer:\nLet\'s think step by step. \n1. We are given that Dale is the son of Ashley. \n2. If Dale is the son of Ashley, then Ashley is the mother of Dale. " + \
    #         "\nTherefore, the answer to the question is No, the statement is ot true.\n***\n"
    #jbprompt fewshot no rules
    # few_shot = "Facts:\n[Nancy] is the daughter of [Heidi]\n[Lorraine] is the sister of [Heidi]\n" + \
    #         "\nHere are some additional facts we\'ve found:\n[Nancy] is the mother of [Lorraine]\n" + \
    #         "\nQuestion: Is the following statement true: \n\"[Lorraine] is [Nancy]\'s daughter\"\n\nAnswer:\nLet\'s think step by step. \n1. We have already found that [Nancy] is the mother of Lorraine.\n2. If [Nancy] is the mother of [Lorraine], then [Lorraine] is the daughter of [Nancy].\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts:\n[Nancy] is the sister of [Dale]\n[Louise] is the daughter of [Nancy]\n" + \
    #         "\nHere are some additional facts we\'ve found:\n[Dale] is the uncle of [Louise]." + \
    #         "\n\nQuestion: Is the following statement true: \n\"[Louise] is not [Dales]\'s niece\"\n" + \
    #         "\nAnswer:\nLet\'s think step by step. \n1. We are given that [Dale] is the uncle of [Louise].\n2.If [Dale ]is the uncle of [Louise], then [Louise] is the niece of [Dale].\nTherefore, the answer is No, the statement is not true.\n***\n" + \
    #         "Facts: \n[Nancy] is the sister of [Lillian]. \n[Douglas] is the son of [Lillian]. " + \
    #         "\n\nHere are some additional facts we\'ve found:\n[Lillian] is the sister of [Nancy]. \nIf [Nancy] is the sister if [Lillian] then [Lillian] is the sister of [Nancy].\n" + \
    #         "\nQuestion: Is the following statement true: \n\"[Douglas] is [Nancy]\'s nephew\"\n\nAnswer:\nLet\'s think step by step. \n1. [Douglas] is [Lillian]\'s son. \n2. [Nancy] is [Lillian]\'s sister. " + \
    #         "\n3. [Douglas] is [Nancy]\'s nephew\nTherefore, the answer to the question is Yes, the statement is true. \n***\n" + \
    #         "Facts: \n[Charlotte] is the granddaughter of [Ashley].\n[Dale] is the father of [Charlotte]." + \
    #         "\n\nHere are some additional facts we\'ve found:\n[Dale] is the son of [Ashley]\nIf [Dale] is father of [Charlotte] and [Ashley] is the grandmother of [Charlotte] then [Dale] is the son of [Ashley].\n" + \
    #         "\nQuestion: Is the following statement true: \n\"[Ashley] is not [Dale]\'s mother\"\n\nAnswer:\nLet\'s think step by step. \n1. We are given that [Dale] is the son of [Ashley]. \n2. If [Dale] is the son of [Ashle], then [Ashley] is the mother of [Dale]. " + \
    #         "\nTherefore, the answer to the question is No, the statement is not true.\n***\n"
    system_prompt = "You are a logical reasoner. You will be given some information, and will be asked to determine if a statement is true. Answer each question by providng a detailed and careful trace of your reasoning process. If there is not enough information, the correct answer is \"Unknown\". Here are some examples: \n"
    user_prompt = "Now, please answer the following logic problem carefully, following the examples provided above: \n"
    if not jbprompt:
        if 'newrules' in prob.keys():
            # prompt =  few_shot  + 'Facts: \n' + '. '.join(prob['context'].split('. ')) + '\nHere are some additional facts we\'ve found: \n' + '.\n'.join(prob['newrules']) +  '\nQuestion: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
            prompt = few_shot +  'Facts: ' + prob['context'] + ' Here are some additional facts we\'ve found: ' + '. '.join(prob['newrules']) + '. Question: ' + prob['question'].strip('.') + ' Answer: Let\'s think step by step.' 

            # prompt = system_prompt + few_shot + user_prompt + 'Facts: \n' + '.\n'.join(prob['context'].split('. ')) + '\nHere are some additional facts we\'ve found: \n' + '.\n'.join(prob['newrules']) +  '\nIs the following statement true?: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
        else:
            prompt = few_shot + 'Facts: \n' + prob['context'] + '\n\nHere are some additional facts and rules we\'ve found: \n' + '[EMPTY]' +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
    else:
    # if not jbprompt:
    #     if 'newrules' in prob.keys():

    #         prompt = few_shot +  'Facts: \n' + '\n'.join(prob['context'].split('. ')) +  '\n' + '\n'.join(prob['newrules']) +  '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
    #     else:
    #         prompt = few_shot + 'Facts: \n' + prob['context'] + '\n\nInstruction: Determine if following statement is true: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
    # else:
        if 'newrules' in prob.keys():

            prompt = few_shot +  'Facts: \n' + '\n'.join(prob['jbprompt']) + '\n\nHere are some additional facts we\'ve found: \n' + '\n'.join(prob['newrules']) +  '\n\nQuestion: Is the following statement true: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
        else:
            prompt = few_shot + 'Facts: \n' + prob['context'] + '\n\nHere are some additional facts we\'ve found:\n' + '[EMPTY]' +  '\n\nQuestion: Is the following statement true: \n' + prob['question'].strip('.') + '\nAnswer:\nLet\'s think step by step.\n1.' 
    
    votes = torch.tensor([0.0,0.0])
    # breakpoint()
    for i in range(n):

        ans = llm.complete(prompt, max_new=1000, temp=1)[0]
        # ans = 'Here are some facts and rules:'.join(ans.split('Here are some facts and rules:')[:5])
        # if len(ans.split('Facts')) > 7:
        #     ans = 'Facts'.join(ans.split('Facts')[:5])
        # ans_prompt = ans + "Therefore, the final answer (Yes/No) is: "
        # yn = llm.yn([ans_prompt]).values
        # nli = torch.tensor(yn[0], yn[2])
        # ans = 'Facts:'.join(ans.split('Facts:')[:n_fewshot + 2])

        ans_prompt = 'Facts:' + ans.split('Facts:')[n_fewshot+1] + "Therefore, the answer (True/False) is "
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
        z = ans_prompt.split('Therefore')[1]
        if 'Yes' in z: 
            votes[0] += 1.0
            print('[1,0]')
            print(ans_prompt)
            continue

        elif 'No' in z: 
            votes[1] += 1.0
            print('[0,1]')
            print(ans_prompt)
            continue
        else: 
            print(f"nli: {llm.nli(ans_prompt, False)}")
            nli = torch.tensor(list(llm.nli(ans_prompt, False).values()))
            votes += nli.softmax(-1)
            print(nli.softmax(-1))
            print(f"ans_prompt: {ans_prompt}")
            continue  
        try:
            while 'Therefore' not in lines[i]:
                i -= 1
                if -1*i == len(lines):
                    notherefore=True
                    break
        except:
            breakpoint()

        # if 'Therefore' in lines[i]:
        try: answer = ans.split('{')[1].split('}')[0].strip(' ').lower()
        except: answer=None
        notherefore = True
        if answer == 'true':
            nli = torch.tensor([1.0,0.0])
        elif answer == 'false':
            nli = torch.tensor([0.0, 1.0])
        elif answer == 'unknown':
            nli = torch.tensor([0.5, 0.5])
        elif notherefore and 'Unknown' in lines[-1]:
            nli = torch.tensor([0.5, 0.5])
        if not notherefore:
            if 'Unknown' in lines[i]:
                nli = torch.tensor([0.5, 0.5])
                print('MAYBE!')
            elif 'Yes' in lines[i]:
                nli = torch.tensor([1.0,0.0])
            elif 'No' in lines[i]:
                nli = torch.tensor([0.0,1.0])
            # elif 
            else:
                yn = llm.yn([ans + '\n So, is the statement true? Answer: '], maybe=True)
                nli = torch.tensor([yn[0] + yn[1]/2, yn[2] + yn[1]/2])
                print('had to yn', yn)
                print(ans)
        else:
            yn = llm.yn([ans + '\n So, is the statement True, False, or Unknown? Answer: The statement is '], unknown=True)
            # yn = llm.yn([ans + "Therefore, the answer (True/False) is "])
            nli = torch.tensor([yn[0] + yn[1]/2, yn[2] + yn[1]/2])
            # nli = torch.tensor(list(llm.nli(ans + "\Therefore, the answer (True/False) is ", False).values()))
            # nli = torch.tensor([yn[0], yn[1]])
            # print('had to yn', yn)
            print(f"ans: {ans}")
            print(f"yn: {yn}")
            # print(nli)


        # votes[torch.tensor(nli).argmax()] += 1
        votes += nli

        # elif 'Therefore' in ans_prompt.split('\n')[-2]:

        #     if 'Yes' in ans_prompt.split('\n')[-2] or 'yes' in ans_prompt.split('\n')[-2]:
        #         nli = [1,0]
        #     elif 'No' in ans_prompt.split('\n')[-2] or 'no' in ans_prompt.split('\n')[-2]:
        #         nli = [0,1]
        # try:
        #     # 1[2]
        #     votes[torch.tensor(nli).argmax()] += 1
        # except: 
        #     yn = llm.yn([ans + '\n So, is the statement true? Answer: '], maybe=False)
        #     votes[torch.tensor([yn[0], yn[2]]).argmax()] += 1
        
    votes = votes
    # print(ans.split(few_shot)[1])
    # print(ans)

    # answer = answers[torch.argmax(nli)]f
    return votes, [prompt]
def next_var(bb, file, thresh=0.96 , dynamic=True, llm=None, lim=500, prob='', fixed_iter = 4, looplim = 100, call_lim = 100, cot_thresh=1.00,n_consec=5, weight=1, llmb=None, task='prontoqa', missed=False, seedrun='clutrr_1'):
    prompts = []
    all_probs = []
    cot_out = None
    all_trns = []
    rule_scores = {}
    cache = {}
    calls = 0
    patterns = []
    scs = []
    trns_cache = {}
    ps = torch.tensor([0.0, 0.0])
    loopcount=0
    missed_flag = None
    og = file
    sc_hist = []
    exp_w = torch.tensor(0)
    #   
    vv = []
    pb = bb['pos']
    nb = bb['neg']
    sfx = ['cnf', 'mapping', 'maptxt']
    og_file = file
    files = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
    for f in files:
        for s in sfx:
            if 'pos' in f:
                shutil.copy(f[:-3] + s, '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/workfiles' + str(seedrun) + '/pos_tmp.' + s)
            else:
                shutil.copy(f[:-3] + s, '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/workfiles' + str(seedrun) + '/neg_tmp.' + s)
    file = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/workfiles' + str(seedrun) +  '/tmp.cnf'
    
    
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
    bb = get_bb(file, seedrun=seedrun)
    nb = bb['neg']
    pb = bb['pos']
    jb = list(set(pb).intersection(set(nb)))

    ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
    for b in ab:
    
        phr = mapping[str(np.abs(b))]
        
                
        ppl = [phr.split('_')[-2], phr.split('_')[-3]]
        for name in ppl:
            if name not in names.keys():
                names[name] = 1
            else:
                names[name] += 1
    prob['newrules'] = []
    prob['jbprompt'] = []
    for j in jb:
        split = mapping[str(np.abs(j))].split('_')

        if len(split) == 3:
            varname = split[1] + '(' + split[0] + ')'
        elif len(split) == 4:
            # varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
            varname = '[' + split[2] + '] is the ' + split[0] + ' of [' + split[1]+']'
            if j < 0:
                varname = split[2] + ' is NOT the ' + split[0] + ' of ' + split[1]
        elif len(split) == 5:
            varname = '['+split[3] + '] is the ' + split[0] + '-' + split[1] + ' of [' + split[2] + ']'
        prob['jbprompt'].append(varname)
    while True:
        loopcount += 1
        if loopcount > looplim:
            print('***LOOP LIMIT EXCEEDED***')
            cot_out = cot(prob)                       
            ps += cot_out[0]
            prompts = cot_out[1]
            scs.append(ps.clone())
            print('cot: ', ps)
            # return ['True', 'False'][torch.argmax(ps)]
            answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            return vv + ['By COT'], answs[ps.argmax()], bb, False, rule_scores, True, scs, prompts
            # return vv + ['LIMIT EXCEEDED'], {'pos': [1], 'neg':[]}, bb, True, rule_scores, True
            # break
        if not dynamic:
                names = {}
                bb = get_bb(file, seedrun=seedrun)
                nb = bb['neg']
                pb = bb['pos']
                jb = list(set(pb).intersection(set(nb)))

                ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
                for b in ab:
                
                    phr = mapping[str(np.abs(b))]
                 
                            
                    ppl = [phr.split('_')[-2], phr.split('_')[-3]]
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
        
        
           

        # empath = '/home/XXXX/XXXX/LLM-project/dimacs/'
        
        probs = torch.tensor([-10000, -100000])

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
                       
                if name1 == name2:
                    continue
                # if [name1, name2] in set_pairs or [name2, name1] in set_pairs:
                #     continue
                name3 = None
                n1var = 0
                n2var = 0
                for i in range(len(jb)):
                    b = mapping[str(np.abs(jb[i]))]
                    if name1 == b.split('_')[-3]:
                        name3 = b.split('_')[-2]
                    elif name1 == b.split('_')[-2]:
                        name3 = b.split('_')[-3]
                    else: continue
                    if name2 == name3: continue
                    for j in range(len(jb)):
                        #   
                        try:
                            if name2 in mapping[str(np.abs(jb[j]))] and name3 in mapping[str(np.abs(jb[j]))]:
                                n1var = jb[i]
                                n2var = jb[j]
                                breakflag=True
                                break
                        except:
                            print(j)
                            
                    if breakflag:break
                
                
                

        
                #   
                # print(nv)
                if n1var == 0:
                       
                    continue
                print(name1, name2, name3)
                # print(b.split)
                # breakpoint()[]
                print(b, mapping[str(np.abs(jb[j]))])
                print(mapping[str(np.abs(n1var))], mapping[str(np.abs(n2var))])
                print(f"\\n[{'='*40}]\\n[ITERATION DETAILS]\\n[{'='*40}]")
                print(f"[1] Entity Selection:")
                print(f"    - Target Name 1 (uo[{p1}]): {name1}")
                print(f"    - Target Name 2 (do[{p2}]): {name2}")
                print(f"    - Connecting Name 3: {name3}")
                
                print(f"\\n[2] Found Connecting Backbone Literals:")
                print(f"    - Literal 1: {mapping[str(np.abs(n1var))]}")
                print(f"    - Literal 2: {mapping[str(np.abs(n2var))]}")

                v1names = (mapping[str(np.abs(n1var))].split('_')[-3], mapping[str(np.abs(n1var))].split('_')[-2])
                v1rel = ' '.join(mapping[str(np.abs(n1var))].lower().split('_')[:-3])

                v2names = (mapping[str(np.abs(n2var))].split('_')[-3], mapping[str(np.abs(n2var))].split('_')[-2])
                v2rel = ' '.join(mapping[str(np.abs(n2var))].lower().split('_')[:-3])

                # question = "If " + v1names[1] + " is " + v1names[0] + '\'s '  + v1rel + " and " \
                #     + v2names[1] + " is " + v2names[0] + "\'s " + v2rel + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank . Answer: \\box{"
                
                n_fs = 3
                fewshot_cot = "If Frank is the son of James and Frank is the son of Martha then Martha is James\'  _____. Fill in the blank .\nAnswer:\nLets think step by step: \n [...]\nTherefore, the answer is \\box{ wife }\n " + \
                    "If Joan is the daughter of Harry and Harry is the brother of Kevin then Kevin is Joan\'s  _____. Fill in the blank .\nAnswer:\nLets think step by step: \n [...]\nTherefore, the answer is \\box{ uncle }\n" + \
                    "If Robert is the father of Harriet and Samantha is the mother of Robert then Harriet is Samantha\'s  _____. Fill in the blank .\nAnswer:\nLets think step by step: \n [...]\nTherefore, the answer is \\box{ granddaughter }\n"

                fewshot = "If Frank is the son of James and Frank is the son of Martha then Martha is James\'  _____. Fill in the blank . Answer: \\box{ wife }\n " + \
                    "If Joan is the daughter of Harry and Harry is the brother of Kevin then Kevin is Joan\'s  _____. Fill in the blank . Answer: \\box{ uncle }\n" + \
                    "If Robert is the father of Harriet and Samantha is the mother of Robert then Harriet is Samantha\'s  _____. Fill in the blank . Answer: \\box{ granddaughter }\n"


                question = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                    + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank . Answer: \\box{ "
                
                question_cot = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                    + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____. Fill in the blank .\nAnswer:\nLets think step by step: \n1."
                
                # out = search_pattern(v1names, v1rel, v2names, v2rel, patterns)
                out = None
                if out != None:
                    rel = out[-1]
                    print('FOUND PATTERN')
                else:
                    # completion=llm.complete(fewshot_cot+ question_cot, max_new = 100)[0]
                    completion=llm.complete(fewshot+ question, max_new = 100)[0]

                    calls += 1
                    if calls > call_lim:
                        print('***LIMIT EXCEEDED***')                        
                        cot_out = cot(prob)
                        ps += cot_out[0]
                        prompts = cot_out[1]
                        scs.append(ps.clone())
                        print('cot: ', ps)
                        # return ['True', 'False'][torch.argmax(ps)]
                        answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                        return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, scs, prompts
                        # return vv + ['LIMIT EXCEEDED'], {'pos': [1], 'neg':[]}, bb, True 
            
                    # completion = llm.complete(completion + 'Therefore, the answer is \\box{ ')[0] 
                    try:   
                        rel = '_'.join(completion.split('box{')[n_fs+1].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))
                    except: 
                        completion = llm.complete(completion + 'Therefore, the answer to the question \"' + "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                        + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s _____." + ' is \\box{ ')[0] 
                        rel = '_'.join(completion.split('box{')[n_fs+1].split('}')[0].lower().strip(' ').strip('.').strip(' ').strip('.').strip(' ').strip(' ').lower().split(' '))

                print(f"\n[3] LLM Interrogation:")
                print(f"    - Fill-in-the-blank prompt sent: {question.strip()}")
                print(f"    - Raw output completion length: {len(completion)}")
                print(f"    - Extracted Relationship: '{rel}'")

                    # breakpoint()
                if rel == 'ister':
                    breakpoint()
                nv_mapping = rel + '_' + name1 + '_' + name2 + '_'

                if nv_mapping in [mapping[str(np.abs(int(j)))]for j in jb]:
                    continue
                rule = "If " + v1names[1] + " is the "  + v1rel +  " of " + v1names[0] + " and " \
                        + v2names[1] + " is the "+ v2rel + " of " + v2names[0]  + " then " + name2 + " is " + name1 + "\'s " + rel
                check, ab = rule_check(rule, prob=prob, llm=llm)
                rule_scores[rule] = ab
                
                print(f"\n[4] Rule Validation (Rule Check):")
                print(f"    - Proposed Rule: {rule}")
                print(f"    - Entailment Scores (ab): {ab} -> Passed Check? {check}")

                if ab[0]==0.0675: breakpoint()
                if rule == 'devices(model_xx) and controlled_by(employee, google_home) implies NOT belong_to(google_home, employee)': breakpoint()
                if check == False:
                        # breakpoint()
                        # if a == '0.0675'
                        continue
                patterns = save_pattern(v1names, v1rel, v2names, v2rel, [name1, name2], rel, patterns)
                patterns = save_pattern(v2names, v2rel, v1names, v1rel, [name1, name2], rel, patterns)
                
                nv = np.max(list(map(int, list(mapping.keys())))) + 1
                # if nv in set_vars:
                #     continue
                newv = True
                if nv_mapping in list(mapping.values()):
                    for key, value in mapping.items():
                        if value == nv_mapping:
                            nv = int(key)
                            newv = False
                mapping[str(nv)] = nv_mapping
                probs = torch.tensor([10000, 100000])

                # questioni = "If " + name2 + ' is ' + name1 + '\'s ' \
                #     + rel + ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank . Answer: \\box{'
                
                # questioni = "If " + name2 + ' is the '  + rel + " of " + name1 +  \
                #  ' then ' + name1 + ' is ' + name2 + '\'s _____. Fill in the blank . Answer: \\box{'
                
                # completioni = llm.complete(fewshot + questioni)[0]

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
                try:
                    vv += (nv, question_cot, mapping[str(np.abs(int(nv)))])
                except:
                    vv += (nv, '**COULD NOT FIND A COMMON-SENSE RULE')
                    print("**COULD NOT FIND A COMMON-SENSE RULE")
                    #   
                    break
                    #   
                print(vv)
                print(ab)
                print(prob['newrules'])
                rule = "If [" + v1names[1] + "] is the "  + v1rel +  " of [" + v1names[0] + "] and [" \
                        + v2names[1] + "] is the "+ v2rel + " of [" + v2names[0]  + "] then [" + name2 + "] is [" + name1 + "]\'s " + rel
                # if 'newrules' not in prob.keys():
                #     prob['newrules'] = [rule]
                # else:
                #     prob['newrules'].append(rule)
                split = mapping[str(np.abs(nv))].split('_')
                if len(split) == 3:
                    varname = split[1] + '(' + split[0] + ')'
                elif len(split) == 4:
                    # varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
                    varname = '[' + split[2] + '] is the ' + split[0] + ' of [' + split[1]+']'
                    if nv < 0:
                        varname = '[' + split[2] + '] is NOT the ' + split[0] + ' of [' + split[1] + ']'
                elif len(split) > 4:
                    rel_str = ''
                    for a in split[:-1]:
                        rel_str += a + '-'
                    rel_str = rel_str[:-1]
                    varname = '[' + split[3] + '] is the ' + rel_str + ' of [' + split[2] + ']'
                if 'newrules' not in prob.keys():
                    prob['newrules'] = [varname]
                else: prob['newrules'].append(varname)
                
                print(f"\\n[BACKBONE UPDATE] Inferred new variable: {nv}")
                print(f"[BACKBONE UPDATE] Text: {varname}")
                # breakpoint()
                # print('newrules')
                # split = mapping[str(np.abs(nvi))].split('_')
                # if len(split) == 3:
                #     varname = split[1] + '(' + split[0] + ')'
                # elif len(split) == 4:
                #     # varname = split[0] + '(' + split[1] + ',' + split[2] + ')'
                #     varname = split[2] + ' is the ' + split[0] + ' of ' + split[1]
                #     if nvi < 0:
                #         varname = split[2] + ' is NOT the ' + split[0] + ' of ' + split[1]
                # prob['newrules'].append(varname)
                # if len(prob['newrules']) >= fixed_iter:

                #     ps += cot(prob)
                #     print(ps)
                #     scs.append(ps.clone())
                #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                #     return vv + ['By COT'], answs[ps.argmax()], bb, False, rule_scores, True, scs
                # print('cot: ', ps)
                cot_out = cot(prob)
                ps += cot_out[0]
                scs.append(ps.clone())
                prompts += cot_out[1]
                print((cot_thresh-0.1*(len(prob['newrules'])-1))*ps.sum(), ps)
                print(prob['gt'])
                if torch.max(ps) >= (cot_thresh-0.1*(len(prob['newrules'])-1))*ps.sum():
                    # return ['True', 'False'][torch.argmax(ps)]
                    print('decided with cot')
                    # print((cot_thresh-0.05*(len(prob['newrules'])/2-1))*ps.sum(), ps)
                    answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                    return vv + ['By COT'], answs[ps.argmax()], bb, False, rule_scores, True, scs, prompts
                    sc_hist.append(int(ps.argmax()))
                
                exp_w += 1
                # else: sc_hist.append(-1)
                # print(sc_hist)
                # if sc_hist[-n_consec:] in [[1]*n_consec, [0]*n_consec]:
                #     answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]

                #     return vv + ['By COT'], answs[ps.argmax()], bb, False, rule_scores, True


                # print(completion)
                # print(completioni)
                # breakpoint()
                good=True
                if dynamic:
                    names[name2]=np.max(list(names.values()))+1
                break
            if good:
                print('good')
                break
            # #   
            # if len(vv) > lim*2:
            #     vv += ["ERROR: TIME OUT"]
            #     print("ERROR TIME OUT")
            #     break
        if not good:
            # breakpoint()
            # missed_flag=True
            # new_sols = {'pos': [1], 'neg':[]}
            # missed +=
            # print("missed")
            # breakpoint()
            # break
            print('not good')
            continue
        tmpfiles = ['/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1], '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1] ]
        for f in tmpfiles:
            
            add_clause(f)
            # add_clause(f)
            cf = open(f, 'a')
            if newv:
                add_var(f)
            cf.write('\n' + str(nv) + ' 0')
            # if newvi:
            #     add_var(f)
            # cf.write('\n' + str(nvi) + ' 0')
            cf.close()
        di = []
        
        new_sols = get_sol(file, lim=100, seedrun=seedrun)
        
        set_vars.append(nv)
        # set_vars.append(nvi)
        set_pairs.append([name1, name2])
        #   
        #   
        if len(new_sols['pos']) == 0 or len(new_sols['neg']) == 0:
            # break
            print("both pos and neg are empty")
            return vv, new_sols, bb, False, rule_scores, False, scs, prompts
    # print('done')
    print('***END REACHED***')                        
    cot_out = cot(prob)
    ps += cot_out[0]
    prompts += cot_out[1]
    scs.append(ps.clone())
    print('cot: ', ps)
    
    print("\\n[FINAL BACKBONE]")
    print(f"Positives ({len(bb['pos'])} items): {bb['pos']}")
    print(f"Negatives ({len(bb['neg'])} items): {bb['neg']}")

    # return ['True', 'False'][torch.argmax(ps)]
    answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
    return vv + ['END REACHED'], answs[ps.argmax()], bb, True, rule_scores, True, scs, prompts
    # return vv, new_sols, bb, missed_flag
if __name__ == '__main__':
    # noisy_data = ['clutrr33.cnf']
    noisy_data=[]
    # mistr_data = ['clutrr10.cnf']
    import pickle as pkl
    mistr_data = []
    seedrun = 'clutrr_2'
    # config =  "rulethresh 05, dynamic False, sc5 llama 3B,  no rules in prompt, yes solver, shuffled, new fewshot without rules, fixed_iter_4"                                                             

    config = "rulethresh 03 cot_thresh 100 anneal 01, dynamic True, sc5,llama 3B, no RULES IN PROMPT fixed, yes separation, always YN WITH MAYBE, og prompt, no cot gen, augmented extr, expweight"                                                             
    c = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs_csvs/solver_finished.csv'
    import csv
    import json
    import random
    dataset = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/SAT-LM/data/clutrr_test.json'
    with open(dataset, 'r') as df:
        data = json.loads(df.read())
    try:
        temp_dir = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/tempfiles' + str(seedrun) + '/'
        work_dir = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/workfiles' + str(seedrun) + '/'
        print(f"CREATING DIR: {temp_dir}")
        os.mkdir(temp_dir)
        print(f"CREATING DIR: {work_dir}")
        os.mkdir(work_dir)
    except:
        print('dir already exists')
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(seedrun.split('_')[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    task = 'clutrr'
    missed=False
    c = open(c, 'r')
    cr = csv.reader(c)
    names = []
    all_outs = {}
    cot_list = []
    missed_list = []
    labels = {}
    for row in cr:
        if row[2] == 'SAT' and row[3] == 'SAT':
            cnf = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/neg_'+row[1]).readlines()[0].strip('\n')
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
        c = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/clutrr_new_labels.csv', 'r')
        cr = csv.reader(c)
        for row in cr:
            if not os.path.exists('//mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/neg_'+row[0][:-2]+'cnf'):
                continue
            cnf = open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/neg_'+row[0][:-2]+'cnf').readlines()[0].strip('\n')
            num_clause = int(cnf.split(' ')[-1])
            if row[1] in noisy_data or row[1] in mistr_data:
                continue
            # if num_clause > 500:
                # continue
            labels[row[0][:-2]+'cnf'] = row[1].lower()
    #   
    from tqdm import tqdm
    args = {'train_file_path': './example_data', 'test_file_path': './example_data', 'save_path': './../SFT_train_res', 'engine': 'meta-llama/Llama-2-13b-chat-hf', 
        'n_rows': 20, 'max_length': 300,'temperature': 1, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
    # args['engine'] = 'meta-llama/Meta-Llama-3-8B-Instruct'
    args['engine'] = 'meta-llama/Llama-3.2-3B-Instruct'
    # args['engine'] = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    # args['engine'] = 'Qwen/Qwen2.5-Coder-3B-Instruct'
    args = Struct(**args)
    llm = LLM(args)


    #   
    # names = ['clutrr59.cnf']
    uhohs = []
    times = {}
    acc = 0
    counter = 0
    skip_pbar = True
    cot_acc = 0
    random.shuffle(names)
    stepcount = 0
    for name in (pbar := tqdm(names)):
    # name
        # breakpoint()
        stepcount += 1
        
        print(config)
        prob = data[int(name.split('clutrr')[1].split('.')[0])]
        n1 = prob['query'].split('(\'')[1].split('\',')[0]
        n2 = prob['query'].split(', \'')[1].split('\')')[0]
    
        # n1, n2 = clutrr[idx]['query'][0], clutrr[idx]['query'][1]
        # print(n1, n2)
        # except:
        #     print(clutrr[idx]['query'])
        relationship = prob['label']
        # print(labels[idx])
        # print(labels[idx]=='true')
        # if labels[name].strip(' ') == 'true':
        #     prob['question'] = 'Is ' + n2 + ' ' +  '' + n1 + '\'s ' + relationship + "?"
        #     # print(clutrr[idx]['question'])
        #     # print('hihi')
        # elif labels[name].strip(' ') == 'false':
        #     prob['question'] = 'Is ' + n2 + ' not ' + n1 + '\'s ' + relationship + "?"
        #     # prob['p']
        #     # breakpoint()
        prob['question'] = 'Is [' + n2 + '] ' +  '[' + n1 + ']\'s ' + relationship + "?"
        # if labels[name].strip(' ') == 'true':
        #     prob['question'] = '[' +n2 + ']' +  ' is [' + n1 + ']\'s ' + relationship
        #     # print(clutrr[idx]['question'])
        #     # print('hihi')
        # elif labels[name].strip(' ') == 'false':
        #     prob['question'] = '[' + n2 + '] is not [' + n1 + ']\'s ' + relationship 
            # prob['p']
            # breakpoint()
        if not skip_pbar:
            pbar.set_description('Acc: ' + str(acc / counter) + ', COT Acc: ' + str(cot_acc) + '/' + str(len(cot_list)))
        skip_pbar=False 
        start_time = time.time()
        print(name)
        p = '/mnt/c/Tugas_Akhir/ARGOS_public_anon/main/dimacs/' + name
        # p = '/home/XXXX/XXXX/LLM-project/tempfiles/dimacs_test.cnf'
        # sols = get_sol(p, lim=100)
        #   
        #   
        bb = get_bb(p, seedrun=seedrun)

        prep_time = time.time() - start_time
        # sols = np.load(open("/home/XXXX/XXXX/LLM-project/tempfiles/sols.np.npy", 'rb'), allow_pickle=True)
        # bb = np.load(open("/home/XXXX/XXXX/LLM-project/tempfiles/bb.np.npy", 'rb'), allow_pickle=True)
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
        
        vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts= next_var(bb, p, llm=llm, task=task, missed=missed, prob=prob, seedrun=seedrun)
        missed_flag = False
        if missed_flag:
            missed += 1
        end_time = time.time() - start_time - prep_time

        times[name] = {'prep_time': prep_time, 'run_time': end_time}
        #   
        # print('finished!')
        #   
        all_outs[name] = (vv, solout, bbout, missed_flag, cot_flag, scs, prompts)
        if cot_flag==True:
            cot_list.append(name)
        if not missed_flag == None:
            missed_list.append([name, vv])
            
            
        if ((solout == None and vv == None and bbout == None) or not missed_flag==None) and missed:
            preds[name] = 'missed'            
        elif len(solout['pos'])==0 and len(solout['neg']) > 0:
            if cot_flag == True:
                preds[name] = 'true'
            else:
                preds[name] = 'false'
        
            # if preds[name] != labels[name]:
            #       
        elif len(solout['pos'])>0 and len(solout['neg']) == 0:
            if cot_flag == True:
                preds[name] = 'false'
            else:
                preds[name] = 'true'
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
            if cot_flag:
                cot_acc += 1
        elif preds[name] == 'missed':
            missed += 1
            counter+= 1
        else:
            counter += 1
        print(f"stepcount : {stepcount}")
        if stepcount%25 == 0:
            pkl.dump(all_outs, open('/mnt/c/Tugas_Akhir/ARGOS_public_anon/all_outs_cot_met_clutrr_' + config.replace(' ', '_') + '.pkl', 'wb'))
        

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
    # print('rulethresh 05 cot_thresh 0.79, sc5 llama 3B, no jb prompt, fixed prmopt, no rules in prompt, yes solver')
    print(config)
    breakpoint()