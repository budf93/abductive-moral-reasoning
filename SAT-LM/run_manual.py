import os
import argparse
import itertools
from random import choices
import torch

from tqdm import tqdm
from math import ceil

import numpy as np

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    llama_local_completion,
    config_args_and_api,
    config_args,
    register_base_args,
    score_of_completion,
    confidence_of_completion
)

from task_helper import TaskHelper, load_train_test_set
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TRANSFORMERS_CACHE'] = USER_PATH + '/.cache/huggingface/hub'
# cache_dir = '/ephemeral/media/data1/XXXX/hub/'
cache_dir = os.path.join(os.getcwd(), '.cache/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaConfig
import argparse
from tqdm import tqdm
import time
import datetime
import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning
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


class LLM():
    def __init__(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            
        )
        self.config = LlamaConfig(max_position_embeddings=200000, hidden_size=5120)
        self.unknown=False
        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        #'engine': layoric/llama-2-13b-code-alpaca
        self.args = {'train_file_path': './example_data', 'test_file_path': './example_data', 'save_path': './../SFT_train_res', 
                'n_rows': 20, 'max_length': 1000, 'lr': 5e-05, 'weight_decay': 0.0, 'epochs': 10, 'max_grad_norm': 1.0, 'batch_size': 2, 'save_strategy': 'no', 'use_lora': True}
        # self.args['engine'] = 'meta-llama/CodeLlama-70b-Python-hf'
        # self.args['engine'] = 'meta-llama/CodeLlama-70b-Python-hf'
        # self.args['engine'] = 'meta-llama/Meta-Llama-3-70B'
        # self.args['engine'] = 'meta-llama/Llama-2-13b-chat-hf'
        # self.args['engine'] = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
        self.args['engine'] = 'Qwen/Qwen2.5-Coder-3B-Instruct'
        # model_args = {'max_length': 20000, 'hidden_size':20000}
        self.args = Struct(**self.args)
        with no_ssl_verification():
            print('l. 98')
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.engine,
                # 'Meta-Llama/Meta-Llama-3-70B', cache_dir = '/data/XXXX/hub/'
                cache_dir = cache_dir,
                token = os.getenv("HF_TOKEN"),
                # attn_implementation="flash_attention_2"
                )
            print('l. 100')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.engine,
                # 'Meta-Llama/Meta-Llama-3-70B', cache_dir = '/data/XXXX/hub/',
                    quantization_config=quant_config,
                    device_map='auto',
                    cache_dir = cache_dir,
                    token = os.getenv("HF_TOKEN"),
                    # attn_implementation="flash_attention_2" # perlu inst flash attn 2
                    attn_implementation="sdpa"

                    # , **{"config": self.config}
                    )
        print(self.model)
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
    def complete(self, prompt, max_tokens):
        # print(prompt)
        # prompt = ['hello, how are ']
        # print(prompt)
        # print('encoding')
        encode_ids = self.tokenizer(
        prompt, 
        return_tensors='pt',
        padding=False,
        truncation=False,
        
    ).input_ids.cuda()
        # print('len tokenized', len(encode_ids))
        # print('generating')
        generated_outputs = self.model.generate(
        encode_ids,  
        max_new_tokens=max_tokens,
        return_dict_in_generate=True, 
        output_scores=True,
        )
        # print('len generated', len(generated_outputs))
        # print(generated_outputs)
        # print('decoding')
        responses = self.tokenizer.batch_decode(
            generated_outputs.sequences,
            skip_special_tokens=False
        )
        # print(responses)
        print(prompt)
        print(responses)
        return responses
        
        
def get_eval_split_abbrev(args):
    return args.eval_split

def run_evaluation(args, test_data, responses, print_perplexity=False, return_verbose=False):
    evaluator = get_task_evaluator(args.task)

    prompting_style = args.style_template
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)

    max_sample_num = max([len(x) for x in responses]) if responses else 0
    num_eval_samples = args.num_eval_samples if args.num_eval_samples > 0 else max_sample_num
    if args.first_k > 0:
        test_data = test_data[:args.first_k]
        responses = responses[:args.first_k]

    predictions = [
        [Prediction(x["text"], x["prompt"], *score_of_completion(x)) for x in completions[:num_eval_samples]] for completions in responses
    ]

    print("\\n=== PREDICTIONS ===")
    for i, preds_list in enumerate(predictions):
        for j, pred in enumerate(preds_list):
            print(f"--- Example {i}, Sample {j} ---")
            print(pred.completion)
            print("-" * 50)
    if args.do_print:
        TaskEvaluator.do_printing = True
    if args.do_impose_prediction:
        TaskEvaluator.do_impose_prediction = True

    sums = np.array([[x.logprob for x in preds] for preds in predictions])
    norms = np.array([[x.norm_logprob for x in preds] for preds in predictions])
    avg_sum = sums.mean(axis=1).mean(axis=0)
    avg_norm = norms.mean(axis=1).mean(axis=0)

    if print_perplexity:
        print("AVG Logprob: {:.4f}".format(avg_sum))
        print("AVG Norm Logprob: {:.4f}".format(avg_norm))
    filenames = [str(args.task) + str(i) for i in range(len(predictions))]
    print(evaluator.evaluate)
    eval_results = evaluator.evaluate(predictions, test_data, prompting_style, train_sep=task_helper.get_train_sep(), return_verbose=return_verbose, filenames=filenames)
    eval_results["avg_logprob"] = sums.mean(axis=1).mean(axis=0)
    eval_results["avg_normlogprob"] = norms.mean(axis=1).mean(axis=0)
    if return_verbose:
        confidences = [
            [confidence_of_completion(x, evaluator.ANSWER_HINT) for x in completions[:num_eval_samples]] for completions in responses
        ]
        avg_conf = np.array(confidences).mean(axis=1).mean(axis=0)
        eval_results["avg_confidence"] = avg_conf

    return eval_results


def register_manual_args(parser):
    parser.add_argument('--manual_prompt_id', type=str, default=None, required=True)
    parser.add_argument('--style_template', type=str, default="default")

def manual_query_result_filename_func(args):
    # Sanitize engine name to avoid "/" being treated as path separator
    safe_engine = args.engine.replace("/", "_")
    
    return "misc/{}--eng{}--{}{}-{}--manual{}--numsamp{}--temp{}--sty{}--predictions.json".format(
        args.task,
        safe_engine,
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        args.manual_prompt_id,
        args.num_samples,
        args.temperature,
        args.style_template
    )

def read_manual_prompt(task, prompt_id, style_template):
    # prompt_lines = read_jsonline(f'manual_prompts/{task}.jsonline')
    prompt_lines = read_jsonline('manual_prompts/' + str(task) + '.jsonline')

    d = dict([(x["id"], x) for x in prompt_lines])
    # print(d)
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]

def predict_framework(args, llm):
    print("train test split")
    train_data, test_data = load_train_test_set(args)
    test_data = test_data
    print("task helper")
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)
    print("read manual prompt")
    base_manual_prompt = read_manual_prompt(args.task, args.manual_prompt_id, args.style_template)
    prompts_to_complete = []   
    for test_ex in test_data:
        test_part = task_helper.prompt_func(test_ex, [])
        print(f"base manual prompt {base_manual_prompt}")
        print(f"test part {test_part}")
        prompts_to_complete.append(
            [base_manual_prompt + task_helper.get_train_sep() + test_part]
        )
    print("prompts to complete")

    task_max_tokens = task_helper.get_completion_length()
    task_stop_token = task_helper.get_train_sep()
    cache_filename = manual_query_result_filename_func(args)
    print(f"cache filename: {cache_filename}")
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token, llm=llm)
    # print(f"responses1: {responses}")
    # responses = llama_local_completion()
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    print(f"eval")
    eval_results = run_evaluation(args, test_data, responses)

def eval_framework(args):
    _, test_data = load_train_test_set(args)
    responses = read_json(manual_query_result_filename_func(args))
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    eval_results = run_evaluation(args, test_data, responses)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_manual_args(parser)
    llm = LLM()
    # llm = Nonedtm
    args = parser.parse_args()
    assert args.task is not None
    assert args.manual_prompt_id is not None

    # config_args_and_api(args)
    config_args(args)
    if args.run_prediction:
        predict_framework(args, llm)
    else:
        eval_framework(args)

if __name__=="__main__":
   


    main()
