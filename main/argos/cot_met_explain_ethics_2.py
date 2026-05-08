"""
cot_met_explain_ethics_2.py
============================
ExplainEthics-FOL adaptation of cot_met_pronto.py.

Counterpart to cot_met_explain_ethics.py (which uses the z3/implies format).
This file consumes DIMACS/arity files produced by explain_ethics_to_sat_2.py
(FOL ForAll/Implies format, named explainethics2{idx}.cnf).

Key differences from cot_met_explain_ethics.py
-----------------------------------------------
- Variable naming: pred__entity__ (double-underscore, arity-1, like pronto)
- Fill-in-the-blank predicate completion (like pronto) instead of pairwise norm scoring
- Reads .arity files to know predicate arities
- Searches backbone for unknown pred→pred implications using LLM completion
- Labels CSV: explain_ethics_labels_2.csv
"""

import os, shutil, json, time, csv
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import warnings, contextlib, requests
from urllib3.exceptions import InsecureRequestWarning
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

USER_PATH = '/mnt/c/Tugas_Akhir/ARGOS_public_anon'
cache_dir  = os.path.join(USER_PATH, 'main/.cache/huggingface/hub')
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME']            = cache_dir
os.environ['CURL_CA_BUNDLE']     = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

_MORAL_NORMS = [
    'violate_care', 'violate_fairness', 'violate_loyalty',
    'violate_authority', 'violate_sanctity', 'violate_liberty',
]

# ---------------------------------------------------------------------------
# SSL bypass
# ---------------------------------------------------------------------------
_old_merge = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened = set()
    def merge(self, url, proxies, stream, verify, cert):
        opened.add(self.get_adapter(url))
        s = _old_merge(self, url, proxies, stream, verify, cert)
        s['verify'] = False
        return s
    requests.Session.merge_environment_settings = merge
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = _old_merge
        for a in opened:
            try: a.close()
            except: pass

# ---------------------------------------------------------------------------
# LLM wrapper (copied from cot_met_explain_ethics.py)
# ---------------------------------------------------------------------------
class Struct:
    def __init__(self, **e): self.__dict__.update(e)

class LLM:
    _kv_reported = False

    def __init__(self):
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='bfloat16', bnb_4bit_use_double_quant=True)
        engine = 'Qwen/Qwen2.5-Coder-3B-Instruct'
        with no_ssl_verification():
            self.tokenizer = AutoTokenizer.from_pretrained(engine, cache_dir=cache_dir,
                                                           token=os.getenv('HF_TOKEN'))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
            self.model = AutoModelForCausalLM.from_pretrained(
                engine, cache_dir=cache_dir, quantization_config=qc,
                device_map='auto', token=os.getenv('HF_TOKEN'), attn_implementation='sdpa')

    @property
    def device(self): return next(self.model.parameters()).device

    def sentence_probabilities(self, sentences):
        with torch.no_grad():
            toks = self.tokenizer(sentences, return_tensors='pt', padding=True)
            ids  = toks.input_ids.to(self.device)
            mask = toks.attention_mask.to(self.device)
            out  = self.model(ids, attention_mask=mask)
            lp   = out.logits.log_softmax(-1)
            lp   = lp[:, :-1, :].gather(2, ids[:, 1:, None]).squeeze(-1)
            lp   = (lp * mask[:, 1:]).sum(-1).cpu()
        return lp

    def yn(self, sentences, norm=True, maybe=False):
        yns = []
        for s in sentences:
            if maybe: yns += [s+'Yes', s+'Maybe', s+'No']
            else:     yns += [s+'Yes', s+'No']
        probs = list(self.sentence_probabilities(yns))
        probs = torch.tensor(probs)
        pyes, pno, pmaybe = [], [], []
        z = 3 if maybe else 2
        for i in range(0, len(probs), z):
            if maybe:
                y, m, n = torch.tensor([probs[i], probs[i+1], probs[i+2]]).softmax(-1)
                pyes.append(y); pmaybe.append(m); pno.append(n)
            else:
                y, n = torch.tensor([probs[i], probs[i+1]]).softmax(-1)
                pyes.append(y); pno.append(n)
        if maybe: return [torch.tensor(pyes), torch.tensor(pmaybe), torch.tensor(pno)]
        return [torch.tensor(pyes), torch.tensor(pno)]

    def complete(self, prompt, max_new=25):
        ids = self.tokenizer(prompt, return_tensors='pt', padding=True,
                             truncation=True, max_length=1000).input_ids.to(self.device)
        out = self.model.generate(ids, max_new_tokens=max_new,
                                  return_dict_in_generate=True, output_scores=True)
        return self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# DIMACS helpers
# ---------------------------------------------------------------------------
def add_clause(f):
    lines = open(f).readlines()
    w = ''
    for l in lines:
        if l.startswith('p cnf'):
            nv, nc = l.split()[2], l.split()[3]
            w += f'p cnf {nv} {int(nc)+1}\n'
        else: w += l
    open(f, 'w').write(w)

def add_var(f):
    lines = open(f).readlines()
    w = ''
    for l in lines:
        if l.startswith('p cnf'):
            nv, nc = l.split()[2], l.split()[3]
            w += f'p cnf {int(nv)+1} {nc}\n'
        else: w += l
    open(f, 'w').write(w)

# ---------------------------------------------------------------------------
# Backbone extraction (same as cot_met_explain_ethics.py / cot_met_pronto.py)
# ---------------------------------------------------------------------------
def get_bb(file, del_sols=None, seedrun=0):
    bb = {'pos': [], 'neg': []}
    files = [
        '/'.join(file.split('/')[:-1]) + '/pos_' + file.split('/')[-1],
        '/'.join(file.split('/')[:-1]) + '/neg_' + file.split('/')[-1],
    ]
    for cur in files:
        dst = '/'.join(cur.split('/')[:-2]) + '/tempfiles' + str(seedrun) + '/' + cur.split('/')[-1]
        shutil.copy(cur, dst)
        if del_sols:
            polarity = 'pos' if ('pos' in cur and 'neg' not in cur) else 'neg'
            for sol in del_sols[polarity]:
                add_clause(dst)
                cf = open(dst, 'a')
                cf.write('\n' + ' '.join(str(-l) for l in sol))
                cf.close()
        bbone_pth = dst[:-4] + '.bbone'
        os.system(f'timeout 5000 {USER_PATH}/main/cadiback/cadiback {dst} > {bbone_pth}')
        try:
            for line in open(bbone_pth):
                if line.startswith('b'):
                    for lit in line.split()[1:]:
                        if lit == '0': continue
                        lit = int(lit)
                        if 'pos' in cur and 'neg' not in cur: bb['pos'].append(lit)
                        elif 'neg' in cur: bb['neg'].append(lit)
        except: pass
    return bb

# ---------------------------------------------------------------------------
# rule_check (ethics-flavoured, same as cot_met_explain_ethics.py)
# ---------------------------------------------------------------------------
def rule_check(rule, prob='', llm=None):
    contra_fs = (
        'Does the following ethical rule seem contradictory?\n'
        'Rule: tearing_down_decorations implies emotional_harm.\n'
        'Answer: \\box{ No } not contradictory.\n'
        'Does the following ethical rule seem contradictory?\n'
        'Rule: helping_a_friend implies violate_loyalty.\n'
        'Answer: \\box{ Yes } contradictory.\n'
    )
    a = 1 - llm.yn([contra_fs + 'Does the following ethical rule seem contradictory?\nRule: ' + rule + '\nAnswer: \\box{ '])[0]
    ctx = prob.get('context', '') if isinstance(prob, dict) else str(prob)
    b = llm.yn(['Context: ' + ctx + '\nIs this rule contextually relevant?\nRule: ' + rule + '\nAnswer \"Yes\" or \"No\": '])[0]
    if float(a) < 0: return False, [a, b]
    if float(b) < 0: return False, [a, b]
    return True, [a, b]

# ---------------------------------------------------------------------------
# CoT fallback (ethics-specific, same as cot_met_explain_ethics.py)
# ---------------------------------------------------------------------------
def cot(prob, n=5, llm=None):
    norms = ', '.join(_MORAL_NORMS)
    few_shot = (
        f'Possible norm violations: {norms}.\n'
        'Context: I crush the frog. Question: Does this violate violate_care? '
        "Answer: Let's think step by step. 1. Frogs are animals. 2. Crushing causes physical harm. "
        'Therefore the answer is violate_care.\n'
        'Context: I won my match by moving the ball when no one looked. Question: Does this violate violate_fairness? '
        "Answer: Let's think step by step. 1. Moving the ball secretly is cheating. 2. Cheating breaks fairness. "
        'Therefore the answer is violate_fairness.\n'
        'Context: I ignored my boss\'s order. Question: Does this violate violate_authority? '
        "Answer: Let's think step by step. 1. Ignoring a boss is disobedience. 2. Bosses are authority figures. "
        'Therefore the answer is violate_authority.\n'
    )
    prompt = (
        few_shot
        + f"Context: {prob['context']} Question: Does this violate {prob.get('label','')}? "
        + "Answer: Let's think step by step."
    )
    gold = prob.get('gold_foundation', '').lower().strip().replace('-', '_')
    votes = torch.tensor([0.0, 0.0])
    for _ in range(n):
        ans = llm.complete(prompt, max_new=300)[0]
        try: section = 'Context:' + ans.split('Context:')[4]
        except: section = ans
        z = section.split('Therefore')[-1] if 'Therefore' in section else ''
        predicted = next((nm for nm in _MORAL_NORMS if nm in z.lower()), None)
        if predicted is None:
            predicted = next((nm for nm in _MORAL_NORMS if nm in section.lower()), None)
        if predicted is not None:
            if predicted == gold: votes[0] += 1.0
            else: votes[1] += 1.0
        else:
            try:
                nli = torch.tensor(list(llm.yn([section + ' Therefore, the answer (Yes/No) is '])))
                votes += torch.stack(nli).squeeze(-1)
            except: pass
    return votes

# ---------------------------------------------------------------------------
# next_var — pronto-style fill-in-the-blank predicate search for ExplainEthics
# ---------------------------------------------------------------------------
def next_var(bb, file, llm=None, lim=100, prob='', seedrun=0, cot_thresh=0.8):
    """
    Search the backbone for underdetermined literals (jb = pos ∩ neg) and ask
    the LLM to fill in a moral implication connecting two backbone predicates.
    Mirrors cot_met_pronto.next_var() but uses ethics-specific prompts and
    pred__entity__ variable naming.
    """
    vv = []
    rule_scores = {}
    sc_scores = []
    ps = torch.tensor([0.0, 0.0])
    missed_flag = None
    calls = 0
    set_vars = []
    set_pairs = []
    patterns = []

    # --- copy workfiles ---
    sfx = ['cnf', 'mapping', 'maptxt', 'arity']
    for q in ['pos', 'neg']:
        src_base = '/'.join(file.split('/')[:-1]) + '/' + q + '_' + file.split('/')[-1][:-4]
        dst_base = f'{USER_PATH}/main/workfiles{seedrun}/{q}_tmp'
        for s in sfx:
            try: shutil.copy(src_base + '.' + s, dst_base + '.' + s)
            except Exception as e: print(f'[next_var] copy {s} failed: {e}')

    wf = f'{USER_PATH}/main/workfiles{seedrun}/tmp.cnf'

    # --- load arity + mapping ---
    arity_pth = f'{USER_PATH}/main/workfiles{seedrun}/neg_tmp.arity'
    try:
        arity1 = np.load(open(arity_pth, 'rb'), allow_pickle=True).item()
        arity = {k.lower(): v for k, v in arity1.items()}
    except Exception as e:
        print(f'[next_var] arity load failed: {e}')
        arity = {}
    print('[next_var] arity:', arity)

    maptxt_pth = f'{USER_PATH}/main/workfiles{seedrun}/neg_tmp.maptxt'
    try:
        maptxt = open(maptxt_pth).read()
        maptxt = (maptxt.replace(' ', ' "').replace(',', '",').replace(':', '":')
                  .replace('{', '{"').replace('}', '"}'))
        mapping = json.loads(maptxt)
    except Exception as e:
        print(f'[next_var] mapping load failed: {e}')
        ps += cot(prob, llm=llm)
        sc_scores.append(ps.clone())
        answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
        return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

    # --- few-shot prompt for fill-in-the-blank ---
    few_shot_pos = (
        'Fill in the blank: If I is covering_up_truth then I is a ___. Answer: \\box{ deception }\n'
        'Fill in the blank: If I is physical_harm then I is a ___. Answer: \\box{ violate_care }\n'
    )
    few_shot_neg = (
        'Fill in the blank: If I is helping_friend then I is NOT a ___. Answer: \\box{ violate_loyalty }\n'
        'Fill in the blank: If I is kind_act then I is NOT a ___. Answer: \\box{ emotional_harm }\n'
    )
    n_fs = 2

    loopcount = 0
    while True:
        if loopcount > 1000:
            ps += cot(prob, llm=llm)
            sc_scores.append(ps.clone())
            answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores
        loopcount += 1

        if calls > lim:
            print('***LIMIT EXCEEDED***')
            ps += cot(prob, llm=llm)
            sc_scores.append(ps.clone())
            answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
            return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

        # refresh backbone
        bb = get_bb(wf, seedrun=seedrun)
        nb = bb['neg']; pb = bb['pos']
        jb = list(set(pb).intersection(set(nb)))

        # build name frequency table from all backbone literals
        names = {}
        ab = list(set(np.abs(pb)).union(set(np.abs(nb))))
        for b in ab:
            phr = mapping.get(str(int(np.abs(b))), '')
            parts = phr.split('__')
            if len(parts) >= 2:
                for ent in parts[1:]:
                    ent = ent.strip('_')
                    if ent:
                        names[ent] = names.get(ent, 0) + 1

        uo = sorted(names, key=names.get)[::-1]
        do = sorted(names, key=names.get)

        good = False
        for negative in [1, -1]:
            for p1 in range(len(uo)):
                for p2 in range(len(do)):
                    name1 = uo[p1]
                    name2 = do[p2]

                    # find n1var / n2var from jb
                    n1var, n2var = 0, 0
                    for i in range(len(jb)):
                        phr_i = mapping.get(str(int(np.abs(jb[i]))), '')
                        if name1 in phr_i.split('__'):
                            for j in range(len(jb)):
                                phr_j = mapping.get(str(int(np.abs(jb[j]))), '')
                                if name2 in phr_j.split('__'):
                                    n1var = jb[i]; n2var = jb[j]

                    if n1var == 0:
                        continue

                    pred1 = mapping.get(str(int(np.abs(n1var))), '').split('__')[0].lower()
                    pred1str = ('NOT_' + pred1) if n1var < 0 else pred1

                    # build question
                    if negative == 1:
                        question = f'Fill in the blank: If {name1} is {pred1str} then {name1} is a ___. Answer: \\box{{ '
                        fs = few_shot_pos
                    else:
                        question = f'Fill in the blank: If {name1} is {pred1str} then {name1} is NOT a ___. Answer: \\box{{ '
                        fs = few_shot_neg

                    print(question)
                    completion = llm.complete(fs + question, max_new=25)[0]
                    calls += 1
                    try:
                        rel = '_'.join(
                            completion.split('box{')[1 + n_fs].split('}')[0]
                            .lower().strip(' .\n').split()
                        )
                    except Exception:
                        continue
                    if not rel or '(' in rel or ')' in rel or '\\' in rel:
                        continue

                    # strip NOT_ prefix if present
                    if rel.startswith('not_'):
                        negative = -1
                        rel = rel[4:]
                    elif rel.startswith('not '):
                        negative = -1
                        rel = rel[4:]

                    rule_txt = (f'If {name1} is {pred1str} then {name1} is '
                                + ('NOT ' if negative < 0 else '') + rel)
                    check, ab_scores = rule_check(rule_txt, prob=prob, llm=llm)
                    rule_scores[rule_txt] = ab_scores
                    if not check:
                        continue

                    # register new variable
                    arity[rel] = 1
                    nv_mapping = rel + '__' + name1 + '__'
                    if nv_mapping in [mapping.get(str(int(np.abs(j))), '') for j in jb]:
                        continue
                    nv = np.max(list(map(int, mapping.keys()))) + 1
                    newv = True
                    if nv_mapping in mapping.values():
                        for k, v in mapping.items():
                            if v == nv_mapping:
                                nv = int(k); newv = False
                    mapping[str(nv)] = nv_mapping

                    # update DIMACS files
                    tmpfiles = [
                        '/'.join(wf.split('/')[:-1]) + '/pos_' + wf.split('/')[-1],
                        '/'.join(wf.split('/')[:-1]) + '/neg_' + wf.split('/')[-1],
                    ]
                    for f in tmpfiles:
                        add_clause(f)
                        if newv: add_var(f)
                        open(f, 'a').write('\n' + str(negative * nv) + ' 0')

                    vv += [negative * nv, nv_mapping, rule_txt]
                    varname = (name1 + ' is NOT a ' + rel) if negative < 0 else (name1 + ' is a ' + rel)
                    if 'newrules' not in prob: prob['newrules'] = [varname]
                    else: prob['newrules'].append(varname)

                    ps += cot(prob, llm=llm)
                    sc_scores.append(ps.clone())
                    print('cot:', ps)
                    thresh_now = (cot_thresh - 0.05 * len(prob.get('newrules', []))) * ps.sum()
                    if torch.max(ps) >= thresh_now or ps.sum() == 20:
                        answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
                        return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores

                    good = True
                    break
                if good: break
            if good: break

        if not good:
            continue  # retry backbone

        try:
            new_sols_check = {'pos': get_bb(wf, seedrun=seedrun)['pos'],
                              'neg': get_bb(wf, seedrun=seedrun)['neg']}
        except Exception:
            new_sols_check = {'pos': [1], 'neg': [1]}

        set_vars.append(nv)
        set_pairs.append([name1, name2])
        if len(new_sols_check['pos']) == 0 or len(new_sols_check['neg']) == 0:
            return vv, new_sols_check, bb, missed_flag, rule_scores, False, sc_scores

    ps += cot(prob, llm=llm)
    sc_scores.append(ps.clone())
    answs = [{'pos': [], 'neg': [0]}, {'pos': [0], 'neg': []}]
    return vv + ['By COT'], answs[ps.argmax()], bb, True, rule_scores, True, sc_scores


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    seedrun = 'ethics2_0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.makedirs(f'{USER_PATH}/main/tempfiles{seedrun}', exist_ok=True)
    os.makedirs(f'{USER_PATH}/main/workfiles{seedrun}', exist_ok=True)

    DIMACS_DIR   = f'{USER_PATH}/main/dimacs'
    DATASET_PATH = f'{USER_PATH}/SAT-LM/data/explainethics_test.json'
    LABELS_CSV   = f'{USER_PATH}/main/explain_ethics_labels_2.csv'
    SOLVER_CSV   = f'{USER_PATH}/main/dimacs_csvs/solver_finished.csv'
    OUTPUT_PKL   = f'{USER_PATH}/main/analysis/analysis_outputs/explain_ethics_2_outputs.pkl'
    PREMISES_DIR = f'{USER_PATH}/main/analysis/analysis_outputs/final_premises_2'
    os.makedirs(PREMISES_DIR, exist_ok=True)

    with open(DATASET_PATH) as f:
        data = json.load(f)

    labels = {}
    names  = []
    with open(SOLVER_CSV) as f:
        for row in csv.reader(f):
            # Only process FOL files named explainethics2{idx}.cnf
            if not row[1].startswith('explainethics2'):
                continue
            if row[2] == 'SAT' and row[3] == 'SAT':
                idx = int(row[1].replace('explainethics2', '').split('.')[0])
                labels[row[1]] = data[idx].get('gt', 'true')
                names.append(row[1])
            elif row[2] == 'UNSAT' or row[3] == 'UNSAT':
                idx = int(row[1].replace('explainethics2', '').split('.')[0])
                labels[row[1]] = data[idx].get('gt', 'true')

    llm = LLM()
    preds = {}
    all_outs = {}
    acc = counter = 0

    for name in tqdm(names):
        print('\n' + '='*50)
        print(f'Processing: {name}')
        idx  = int(name.replace('explainethics2', '').split('.')[0])
        prob = data[idx]
        prob['newrules'] = []

        p    = os.path.join(DIMACS_DIR, name)
        bb   = get_bb(p, seedrun=seedrun)
        jb   = list(set(bb['pos']).intersection(set(bb['neg'])))

        # Log backbone
        try:
            maptxt = open(os.path.join(DIMACS_DIR, 'neg_' + name[:-4] + '.maptxt')).read()
            maptxt = (maptxt.replace(' ', ' "').replace(',', '",').replace(':', '":')
                      .replace('{', '{"').replace('}', '"}'))
            mapping_log = json.loads(maptxt)
            print('Backbone +:', [mapping_log.get(str(abs(b)), b) for b in bb['pos']])
            print('Backbone -:', [mapping_log.get(str(abs(b)), b) for b in bb['neg']])
        except Exception: pass

        if len(jb) == 0:
            print('jb=0, skipping')
            continue

        vv, solout, bbout, missed_flag, rule_scores, cot_flag, sc_scores = next_var(
            bb, p, llm=llm, prob=prob, seedrun=seedrun
        )
        all_outs[name] = (vv, solout, bbout, missed_flag, sc_scores)

        # Save premises
        try:
            with open(os.path.join(PREMISES_DIR, name[:-4] + '_premises.txt'), 'w') as pf:
                pf.write(f'POSITIVE BACKBONE:\n')
                for b in bbout['pos']: pf.write(f'  + {b}\n')
                pf.write(f'NEGATIVE BACKBONE:\n')
                for b in bbout['neg']: pf.write(f'  - {b}\n')
                pf.write(f'RULES ADDED:\n')
                for r in vv: pf.write(f'  {r}\n')
        except Exception as e:
            print(f'[premises save] {e}')

        # Determine prediction
        if solout is None or (len(solout.get('pos', [])) == 0 and len(solout.get('neg', [])) == 0):
            preds[name] = 'true'
        elif len(solout.get('pos', [])) == 0:
            preds[name] = 'true'
        elif len(solout.get('neg', [])) == 0:
            preds[name] = 'false'
        else:
            preds[name] = 'true'

        label = labels.get(name, 'true')
        print(f'label={label}  pred={preds[name]}')
        if label == preds[name]:
            acc += 1
        counter += 1
        if counter > 0:
            print(f'Accuracy so far: {acc}/{counter} = {acc/counter:.3f}')

    print(f'\nFinal Accuracy: {acc}/{counter}')
    import pickle
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_outs, f)
    print(f'Saved outputs to {OUTPUT_PKL}')
