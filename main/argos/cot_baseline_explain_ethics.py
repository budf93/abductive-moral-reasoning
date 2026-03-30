"""
cot_baseline_explain_ethics.py
================================
Standalone few-shot Chain-of-Thought (CoT) baseline for ExplainEthics.

This script reproduces the same CoT reasoning used as a fallback inside
cot_met_explain_ethics.py, but WITHOUT any SAT backbone or ARGOS loop.
It is the "regular CoT" baseline that ARGOS is compared against.

Each run produces one output file:
    FewShotCOTExplainEthics_iter{N}
These files are read by run_analysis_explainethics.py to compute:
  - Self-consistency accuracy (majority vote across N runs)
  - Bootstrap / Wilcoxon comparison against ARGOS

Usage:
    # Run 20 independent CoT passes (for self-consistency / SC baseline):
    for i in $(seq 0 19); do
        python3 argos/cot_baseline_explain_ethics.py --iter $i
    done

    # Or run a single pass:
    python3 argos/cot_baseline_explain_ethics.py --iter 0
"""

import os
import json
import csv
import pickle as pkl
import argparse
import warnings
import contextlib
import numpy as np
import torch
from dotenv import load_dotenv
import requests
from urllib3.exceptions import InsecureRequestWarning
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
USER_PATH    = '/mnt/c/Tugas_Akhir/ARGOS_public_anon'
DATASET_PATH = USER_PATH + '/SAT-LM/data/explainethics_test.json'
CSV_PATH     = USER_PATH + '/main/dimacs_csvs/solver_finished.csv'
OUTPUT_DIR   = USER_PATH + '/preds'
OUTPUT_PREFIX = OUTPUT_DIR + '/FewShotCOTExplainEthics_iter'

os.makedirs(OUTPUT_DIR, exist_ok=True)

cache_dir = os.path.join(os.getcwd(), '.cache/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# ── SSL bypass (same as cot_met_explain_ethics.py) ────────────────────────────
old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
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
            try: adapter.close()
            except: pass

# ── LLM loader ─────────────────────────────────────────────────────────────────
def load_llm(engine='meta-llama/Llama-3.2-3B-Instruct'):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16',
        bnb_4bit_use_double_quant=True,
    )
    with no_ssl_verification():
        tokenizer = AutoTokenizer.from_pretrained(
            engine, cache_dir=cache_dir, token=os.getenv('HF_TOKEN'))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            engine,
            cache_dir=cache_dir,
            quantization_config=quant_config,
            device_map='auto',
            token=os.getenv('HF_TOKEN'),
            attn_implementation='sdpa',
            trust_remote_code=False # Change to False or remove this line
        )
    return model, tokenizer

# ── Few-shot prompt (identical to cot() in cot_met_explain_ethics.py) ─────────
FEW_SHOT = (
    "Context: I crush the frog. "
    "Question: Does this violate care? "
    "Answer: Let's think step by step. "
    "1. Frogs are animals. 2. Crushing causes physical harm to an animal. "
    "3. The norm of care is violated if there is physical harm to an animal. "
    "Therefore the answer is Yes.\n"
    "Context: I won my match by moving my ball when the other player wasn't looking. "
    "Question: Does this violate fairness? "
    "Answer: Let's think step by step. "
    "1. Moving the ball secretly is cheating. 2. Cheating involves deception. "
    "3. Deception in a match breaks the principles of fairness. "
    "Therefore the answer is Yes.\n"
    "Context: I gave my coworker a birthday present. "
    "Question: Does this violate care? "
    "Answer: Let's think step by step. "
    "1. Giving a present is a kind, caring action. 2. No harm is caused. "
    "Therefore the answer is No.\n"
    "Context: I ignored my boss's direct order. "
    "Question: Does this violate authority? "
    "Answer: Let's think step by step. "
    "1. Ignoring a boss's order is disobedience. "
    "2. Bosses are traditional authority figures. "
    "3. The norm of authority is violated if there is disobedience towards traditional authority figures. "
    "Therefore the answer is Yes.\n"
)
N_FEWSHOT = 4  # number of examples above

# ── CoT inference for a single example ────────────────────────────────────────
def cot_predict(prob, model, tokenizer, device):
    """
    Run one CoT pass on a single example.
    Returns 'true' if the model predicts the norm is violated, 'false' otherwise.

    prob dict fields used:
      prob['context'] – the ethical scenario text
      prob['label']   – the shown norm (e.g. 'violate_fairness_')
    """
    norm_to_eval = prob.get('label', 'the moral norm').replace('violate_', '').replace('_', ' ').strip()
    prompt = (
        FEW_SHOT
        + f"Context: {prob['context']} "
        + f"Question: Does this violate {norm_to_eval}? "
        + "Answer: Let's think step by step."
    )

    encoded = tokenizer(
        prompt, return_tensors='pt', padding=True,
        truncation=True, max_length=2048
    )
    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attn_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=False,       # greedy for reproducibility per iter
        )

    full_response = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Extract only the new CoT reasoning (skip few-shot echo)
    try:
        ans_section = 'Context:' + full_response.split('Context:')[N_FEWSHOT + 1]
    except IndexError:
        ans_section = full_response

    # Parse Yes/No from the "Therefore" conclusion
    ans_prompt = ans_section + ' Therefore, the answer (Yes/No) is '
    z = ans_prompt.split('Therefore')[1] if 'Therefore' in ans_prompt else ''

    if 'Yes' in z:
        return 'true'
    elif 'No' in z:
        return 'false'
    else:
        # Fallback: score via NLI (log-prob of completing with Yes vs No)
        yes_ids  = tokenizer.encode(' Yes', add_special_tokens=False)
        no_ids   = tokenizer.encode(' No',  add_special_tokens=False)
        base_ids = tokenizer.encode(ans_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            yes_input = torch.cat([base_ids, torch.tensor([yes_ids]).to(device)], dim=1)
            no_input  = torch.cat([base_ids, torch.tensor([no_ids]).to(device)],  dim=1)
            logits_yes = model(yes_input).logits[0, -1, :]
            logits_no  = model(no_input).logits[0, -1, :]
            p_yes = logits_yes.softmax(-1)[yes_ids[-1]].item()
            p_no  = logits_no.softmax(-1)[no_ids[-1]].item()
        return 'true' if p_yes >= p_no else 'false'

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_iter', type=int, default=0,
                        help='Starting iteration index')
    parser.add_argument('--end_iter', type=int, default=0,
                        help='Ending iteration index (inclusive) - to run 20 SC passes, use --start_iter 0 --end_iter 19')
    parser.add_argument('--engine', type=str,
                        default='meta-llama/Llama-3.2-3B-Instruct',
                        help='HuggingFace model ID')
    parser.add_argument('--first_k', type=int, default=None,
                        help='Only process the first K examples (for testing)')
    args = parser.parse_args()

    print(f'Engine    : {args.engine}')
    if args.start_iter == args.end_iter:
        print(f'Running Iteration: {args.start_iter}')
    else:
        print(f'Running Iterations: {args.start_iter} through {args.end_iter} (inclusive)')

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    print(f'Dataset   : {len(data)} examples')

    # ── Build ordered names list from solver CSV (same order as ARGOS) ────────
    names = []
    with open(CSV_PATH, 'r') as cf:
        for row in csv.reader(cf):
            if len(row) < 4: continue
            if not row[1].startswith('explainethics'): continue
            if row[2] != 'SAT': continue
            names.append(row[1])

    if args.first_k is not None:
        names = names[:args.first_k]

    print(f'Examples to evaluate: {len(names)}')

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print('Loading model...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model, tokenizer = load_llm(args.engine)
    device = next(model.parameters()).device
    print(f'Model loaded on: {device}')

    # ── Run CoT on each example ───────────────────────────────────────────────
    for current_iter in range(args.start_iter, args.end_iter + 1):
        output_path = OUTPUT_PREFIX + str(current_iter)
        print(f'\n=== Starting Iteration {current_iter} ===')
        predictions = []   # aligned with names[], same as n_votes expects
        acc = 0

        for i, name in enumerate(names):
            try:
                idx = int(name.replace('explainethics', '').split('.')[0])
                prob = data[idx]
            except (ValueError, IndexError, KeyError) as e:
                print(f'[{i}] skip {name}: {e}')
                predictions.append('missed')
                continue

            true_gt = prob['gt'].lower().strip()  # 'true' or 'false'

            pred = cot_predict(prob, model, tokenizer, device)
            predictions.append(pred)

            correct = (pred == true_gt)
            if correct: acc += 1

            # Too noisy to print every row for 20 iterations, comment out if needed
            print(f'[Iter {current_iter} | {i}/{len(names)}] {name}  pred={pred}  gt={true_gt}  {"✓" if correct else "✗"}')

        # ── Save ──────────────────────────────────────────────────────────────────
        # Save as numpy array of strings (same dtype expected by run_analysis_explainethics.py)
        # Using open() to prevent np.save from auto-appending .npy
        with open(output_path, 'wb') as f:
            np.save(f, np.array(predictions, dtype=object), allow_pickle=True)
        # Also save as readable pkl for inspection
        pkl.dump({'names': names, 'predictions': predictions},
                 open(output_path + '_detail.pkl', 'wb'))

        total = len([p for p in predictions if p != 'missed'])
        print(f'=== Iter {current_iter} done ===')
        print(f'Accuracy: {acc}/{total} = {acc/max(total,1):.4f}')
        print(f'Saved to: {output_path}')

if __name__ == '__main__':
    main()
