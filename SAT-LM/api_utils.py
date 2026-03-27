import os
import itertools
import time
import openai
import warnings
import contextlib
import torch

import requests
from urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
from utils import *
from transformers import GPT2TokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
API_ERROR_IDENTIFIER = "OPENAI Error"

os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['TRANSFORMERS_CACHE'] = USER_PATH + '/.cache/huggingface/hub'
# cache_dir = '/ephemeral/media/data1/XXXX/hub/'
cache_dir = os.path.join(os.getcwd(), '.cache/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

# _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt_tok")
# _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="./hfcache")

args = {'train_file_path': './example_data', 'test_file_path': './example_data', 'save_path': './../SFT_train_res', 'engine': 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 
                'n_rows': 20, 'max_length': 1000, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}

# args['engine'] = 'meta-llama/Llama-2-13b-chat-hf'
# args['engine'] = 'meta-llama/CodeLlama-70b-Python-hf'
# args['engine'] = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
args['engine'] = 'Qwen/Qwen2.5-Coder-3B-Instruct'
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    old_merge_environment_settings = requests.Session.merge_environment_settings
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

args = Struct(**args)
with no_ssl_verification():
    _TOKENIZER = AutoTokenizer.from_pretrained(
        # 'Meta-Llama/Meta-Llama-3-70B', cache_dir = '/data/XXXX/hub/'
                        args.engine,
                        cache_dir = cache_dir,
                        # '/data/XXXX/hub/Meta-Llama-3/Meta-Llama-3-70B'
                        # '/data/mbiparva/Meta-Llama-3',
                        token = os.getenv("HF_TOKEN"),

                        )
    GPT3_LENGTH_LIMIT = 2049
    GPT_MAX_ATTEMPTS = 60
    GPT_WAITTIME = 20
    API_ERROR_IDENTIFIER = "OPENAI Error"
# 1. Update the models list to include your local options
models = ['gpt-3.5-turbo-instruct', 'code-davinci-002']
local_models = ['HuggingFaceTB/SmolLM2-1.7B-Instruct', 'meta-llama/Llama-2-13b-chat-hf', 'Qwen/Qwen2.5-Coder-3B-Instruct']
print('done api-utils')
def register_query_args(parser):
    # parser.add_argument('--engine', default='gpt-3.5-turbo-instruct', choices=models)
    parser.add_argument('--engine', default='code-davinci-002', choices=local_models)


        # "text-davinci-002", "text-davinci-003", "code-davinci-001", "code-davinci-002"])
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--do_dryrun', default=False, action='store_true')
    parser.add_argument('--force_override', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)

def register_base_args(parser):
    # standard, instruction, etc
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--eval_split', type=str, default="test")
    parser.add_argument('--slice_train', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=-1)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--slice_dev', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=-1)
    parser.add_argument('--do_print', default=False, action='store_true')
    parser.add_argument('--num_eval_samples', type=int, default=-1)
    parser.add_argument('--first_k', type=int, default=-1)
    parser.add_argument('--do_impose_prediction', default=False, action='store_true')
    register_query_args(parser)
def config_args(args):
    if args.batch_size == -1:
        args.batch_size = 1

def config_args_and_api(args):
    if args.batch_size == -1:
        args.batch_size = 1

    # Check if we are using an OpenAI model
    if args.engine in models:
        # Fix for OpenAI v1.0+: Remove or update the timeout line
        # openai.api_requestor.TIMEOUT_SECS = 60  <-- DELETE THIS LINE
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
             print("Warning: OPENAI_API_KEY not found in environment.")
             
    # Logic for SmolLM and other HuggingFace models
    elif args.engine in local_models or "HuggingFaceTB" in args.engine:
        print(f"Using local model: {args.engine}")
        # Your local model loading logic (AutoModelForCausalLM) usually goes here
    else:
        raise RuntimeError(f"Engine or Model Choice {args.engine} not supported")

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens


def gpt_safe_completion(engine, prompts, temperature, max_tokens, stop_token, logprobs=1, num_samples=1, echo=True):
    print("gpt safe completion")
    last_exc = None
    for i in range(GPT_MAX_ATTEMPTS):
        try:
            return openai.Completion.create(engine=engine, prompt=prompts,
                temperature=temperature, max_tokens=max_tokens, n=num_samples, echo=echo, stop=stop_token)
        except openai.error.RateLimitError as e:
            last_exc = e
            print("\rWARNING: OPENAI Rate Error", last_exc, end="")
            time.sleep(GPT_WAITTIME)
        except openai.error.APIError as e:
            last_exc = e
            print("\rWARNING: OPENAI API Error", last_exc)
        except openai.error.Timeout as e:
            last_exc = e
            print("\rWARNING: OPENAI Timeout Error", last_exc)
        except openai.error.APIConnectionError as e:
            last_exc = e
            print("\rWARNING: OPENAI APIConnection Error", last_exc, end="")
        except openai.error.ServiceUnavailableError as e:
            last_exc = e
            print("\rWARNING: OPENAI Service Error", last_exc, end="")
    # make a fake response
    fake_choices = [
        [{
            "text": p + " OPENAI Error - " + str(last_exc),
            "API Error": True,
        }] * num_samples
        for p in prompts
    ]
    fake_choices = itertools.chain(*fake_choices)
    resp = {
        "choices": fake_choices
    }
    return resp
def llama_local_completion(prompts, llm, max_tokens=0):
    # print('in llama local"')
    return [{'prompt': "", 'text':llm.complete(prompt, 2*max_tokens)} for prompt in prompts]
def hf_local_completion(model, tokenizer, prompt, max_tokens=512, stop_token=None):
    print("hf local completion")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False 
        )
    
    completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if stop_token and stop_token in completion:
        completion = completion.split(stop_token)[0]
        
    return completion       
def batch_query_engine(args, prompts, max_tokens, stop_token, llm):
    predictions = []
    print("batch size: ", len(prompts))
    # print(len(prompts[0]))
    if llm == None:
        resps = gpt_safe_completion(engine=args.engine, prompts=prompts, temperature=args.temperature, max_tokens=max_tokens, stop_token=stop_token, logprobs=1, num_samples=args.num_samples, echo=True)
        # resps = resps['choices'][0]['text']
        resps = resps['choices']
    else:
        print(f"Running local inference with: {args.engine}")
        resps = []
        for prompt in prompts:
            # Call our local helper using the model/tokenizer inside the llm object
            print(f"prompt for completion: {prompt}")
            res_text = hf_local_completion(llm.model, llm.tokenizer, prompt, max_tokens, stop_token)
            
            # Format to match OpenAI's response structure so rest of code doesn't break
            print(f"res_text: {res_text}")
            resps.append({
                "text": res_text,
                "finish_reason": "stop"
            })
        # return all_responses
    # breakpoint()
    # print('finished llama local')
    # resps = resps["choices"]
    # print("RESPS", resps, len(resps))
    # print("P", prompts, len(prompts))
    resps = [resps[(i * args.num_samples):(i * args.num_samples + args.num_samples)] for i in range(len(prompts))]
    # breakpoint()
    # print(resps, len(resps))
    for prompt, resp in zip(prompts, resps):
        for pred in resp:
            pred["prompt"] = prompt
            if llm != None:
                print("pred text: "+pred["text"][0])
                # pred["text"] = pred["text"][0]
            # if len(pred["text"]) > len(prompt):
            #     pred["text"] = pred["text"][len(prompt):]
            else:
                # breakpoint()
                print("pred text null")
                pred["text"] = " NULL"
            pred["completion_offset"] = len(prompt)

    return resps
[]
# args
# prompts: 2d array
# cache_filename
# for gpt, assuming apis are pretty robust
def run_completion_tasks_with_cache(args, cache_fileneme, prompts_by_examples, max_tokens, stop_token, llm):
    assert isinstance(prompts_by_examples, list) and isinstance(prompts_by_examples[0], list) and isinstance(prompts_by_examples[0][0], str)
    if max_tokens == 0:
        assert args.num_samples == 1
    shape_records = [len(x) for x in prompts_by_examples]
    data_size = sum(shape_records)

    if os.path.exists(cache_fileneme):
        print("Cached Predictions Detected:", cache_fileneme)
        if args.force_override:
            print("Force Overriding Previous Predictions")
        else:
            return read_json(cache_fileneme)

    samples = list(itertools.chain(*prompts_by_examples))
    # print(f"samples: {samples}")

    renewed_results = []
    prompt_lengths = []
    request_pool = []

    task_max_tokens = max_tokens
    for idx, prompt in enumerate(samples):
        if args.do_dryrun:
            # response = length_of_prompt(prompt, task_max_tokens)
            if llm != None:
                response = len(llm.tokenizer(return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).input_ids.cuda())
                print(f"response: {response}")
            else:
                response = length_of_prompt(prompt, task_max_tokens)
            print("-----------------------------------------")
            print(prompt)
            print("LEN", response)
            prompt_lengths.append(response)

        # add to request pool if no cached results, or error happened
        request_pool.append((idx, prompt))

    if args.do_dryrun:
        print(cache_fileneme)
        print('Total request', len(request_pool))
        print('MAX', max(prompt_lengths), 'COMP', task_max_tokens)
        return

    num_request, batch_size = len(request_pool), args.batch_size
    num_batch = (num_request + batch_size - 1) // batch_size
    # prediction loop, auto managing batching for OPT
    print("Num total request", num_request)
    for batch_idx in tqdm(range(num_batch), total=num_batch, desc="Querying"):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        reqs = request_pool[batch_start: batch_end]

        idx_lists = [x[0] for x in reqs]
        prompts = [x[1] for x in reqs]

        responses = batch_query_engine(args, prompts, task_max_tokens, stop_token, llm=llm)

        assert len(idx_lists) == len(responses)
        for i, resp in zip(idx_lists, responses):
            renewed_results.append(resp)

    print(cache_fileneme)
    # save
    # read un indexed dev
    assert len(renewed_results) == sum(shape_records)

    # group by example
    slice_start = 0
    renewed_cache = []
    for n in shape_records:
        renewed_cache.append(renewed_results[slice_start: slice_start + n])
        slice_start = slice_start + n

    print(f"WRITING TO MISC/CACHE: {cache_fileneme}")
    dump_json(renewed_cache, cache_fileneme)
    return renewed_cache



def score_of_completion(response):
    if "logprobs" not in response or response["logprobs"] is None:
        return .0, .0

    completion_offset = len(response["prompt"])
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    if completion_offset in token_offset:
        completion_start_tok_idx = token_offset.index(completion_offset)
    elif completion_offset > token_offset[-1]:
        completion_start_tok_idx = len(token_offset)
    else:
        completion_start_tok_idx = next(filter(lambda x: token_offset[x] >= completion_offset, range(len(token_offset))))

    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>", completion_start_tok_idx)
    else:
        complention_end_offset = completion_offset + len(response["text"])
        completion_end_tok_idx = next(filter(lambda x: token_offset[x + 1] >= complention_end_offset, range(len(token_offset) - 1)), len(token_offset))

    # completion_end_tok_idx = tokens.index("<|endoftext|>")
    # return len(tokens) - completion_start_tok_idx

    tok_scores = response["logprobs"]["token_logprobs"][completion_start_tok_idx:completion_end_tok_idx + 1]
    toks = response["logprobs"]["tokens"][completion_start_tok_idx:completion_end_tok_idx + 1]

    tok_scores = np.array(tok_scores)
    return tok_scores.sum(), tok_scores.mean()

def confidence_of_completion(response, answer_hint):
    completion_offset = len(response["prompt"])
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    # answer_offset = response["text"]
    lower_text =  response["text"].lower()
    lower_hint = answer_hint.lower()
    if lower_hint in lower_text:
        answer_offset = completion_offset + lower_text.index(lower_hint) + len(lower_hint)
    else:
        answer_offset = completion_offset

    if answer_offset in token_offset:
        answer_start_tok_idx = token_offset.index(answer_offset)
    elif answer_offset >= token_offset[-1]:
        return 0.
    else: 
        answer_start_tok_idx = next(filter(lambda x: token_offset[x] >= answer_offset, range(len(token_offset))))

    if "<|endoftext|>" in tokens:
        answer_end_tok_idx = tokens.index("<|endoftext|>", answer_start_tok_idx)
    elif "\n" in tokens[answer_start_tok_idx:]:
        answer_end_tok_idx = tokens.index("\n", answer_start_tok_idx)
    else:
        answer_end_tok_idx = len(tokens)
    if tokens[answer_end_tok_idx - 1].strip() == '.':
        answer_end_tok_idx = answer_end_tok_idx - 1

    # completion_end_tok_idx = tokens.index("<|endoftext|>")
    # return len(tokens) - completion_start_tok_idx

    tok_scores = response["logprobs"]["token_logprobs"][answer_start_tok_idx:answer_end_tok_idx ]
    toks = response["logprobs"]["tokens"][answer_start_tok_idx:answer_end_tok_idx ]
    tok_scores = np.array(tok_scores)
    conf = np.exp(np.sum(tok_scores))
    # print("".join(toks), conf)

    return conf
