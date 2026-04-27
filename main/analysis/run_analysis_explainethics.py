"""
ExplainEthics Analysis Script
==============================
Analyse results produced by cot_met_explain_ethics.py.
Run interactively:  python3 -i run_analysis_explainethics.py

Statistical methods:
  1. Self-consistency majority vote  (CoT iter files → sc_acc)
  2. Bootstrap resampling            (sklearn, n=86)
  3. Wilcoxon signed-rank test       (scipy, ARGOS > SC)
  4. t-interval 95% CI               (scipy.stats.t.interval)
  5. Per-norm confusion matrix       (predicted norm vs gold_foundation)
  6. Iteration-trajectory flip analysis (scs confidence over iterations)
  7. 2D histogram  (iteration × confidence)
  8. Normalised stacked histogram    (proportion per outcome per iteration)
  + ExplainEthics-specific:
  9. Per-norm accuracy (by gold_foundation)
  10. Decoy resistance analysis

all_outs[name] structure (key = 'explainethicsN.cnf'):
  [0] vv          – whether prediction matched gold_foundation
  [1] solout      – {'pos': [...], 'neg': [...]}
  [2] bbout       – backbone after last iter
  [3] missed_flag – True if skipped
  [4] rule_scores – rule → score
  [5] cot_flag    – True if resolved by CoT
  [6] scs         – list of probability tensors over iterations
  [7] prompts     – prompts used

preds semantics (NEW):
  preds[name] = 'violate_care' / 'violate_fairness' / ... / 'unknown' / 'missed'
  labels[name] = data[i]['gold_foundation']  (the true violated norm)
  Correct when preds[name] == labels[name]
"""

import pickle as pkl
import json
import csv
import os
import glob
import numpy as np
import torch
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.size'] = 12

# ── Paths & Logging ──────────────────────────────────────────────────────────────
BASE_PATH    = '/mnt/c/Tugas_Akhir/ARGOS_public_anon'

import sys
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()

# Duplicate all printed output into a text report in the outputs directory
ANALYSIS_OUT_DIR = BASE_PATH + '/main/analysis/analysis_outputs'
os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
sys.stdout = Logger(ANALYSIS_OUT_DIR + '/explainethics_analysis_report.txt')
DATASET_PATH = BASE_PATH + '/SAT-LM/data/explainethics_test.json'
DIMACS_DIR   = BASE_PATH + '/main/dimacs'
CSV_PATH     = BASE_PATH + '/main/dimacs_csvs/solver_finished.csv'
LABELS_CSV   = BASE_PATH + '/main/explain_ethics_labels.csv'
# Optional: directory of CoT baseline iter files for SC comparison
COT_ITER_PREFIX = BASE_PATH + '/preds/FewShotCOTExplainEthics_iter'

# ── Step 1: Find pkl files ─────────────────────────────────────────────────────
pkl_files = sorted(glob.glob(BASE_PATH + '/all_outs_cot_met_explain_ethics*.pkl'))
print('Available ExplainEthics pkl files:')
for f in pkl_files:
    print(' ', f)

# ── Step 2: Load pkl ───────────────────────────────────────────────────────────
if pkl_files:
    PKL_PATH = pkl_files[-1]
    outs = pkl.load(open(PKL_PATH, 'rb'))
    print(f'\nLoaded {len(outs)} examples from {os.path.basename(PKL_PATH)}')
else:
    print('No pkl files found. Run cot_met_explain_ethics.py first.')
    outs = {}

# ── Step 3: Load dataset ───────────────────────────────────────────────────────
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)
print(f'Dataset: {len(data)} ExplainEthics examples')

# ── Step 4: Build labels ──────────────────────────────────────────────────────
# labels[name] = gold_foundation norm string (e.g. 'violate_fairness')
labels = {}
names  = []

for name in outs.keys():
    try:
        idx = int(name.replace('explainethics', '').split('.')[0])
        labels[name] = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        pass
print(f'Derived {len(labels)} labels from dataset gold_foundation field')

# Build names list (ordered same as data index)
for name in sorted(outs.keys(), key=lambda n: int(n.replace('explainethics','').split('.')[0])
                   if n.replace('explainethics','').split('.')[0].isdigit() else 9999):
    names.append(name)
name_idx = {name: i for i, name in enumerate(names)}

# ── Step 5: Build preds ────────────────────────────────────────────────────────
# preds[name] = 'violate_{norm}' | 'unknown' | 'missed'
preds      = {}
cot_count            = 0   # resolved by CoT fallback (cot_flag=True)
presolve_count       = 0   # pre-solved by SAT before backbone loop (initial pos=SAT, neg=UNSAT)
sat_added_premise_count = 0  # resolved by SAT backbone loop (premise was added)
miss_count           = 0

# Sentinel used by cot_met_explain_ethics.py for pre-solved entries (no LLM needed):
# all_outs[row[1]] = (vv, {'pos': ['dummy'], 'neg': []}, None, False, {}, False, [], [])
_PRESOLVE_SENTINEL = 'dummy'

IMAS_DIR = BASE_PATH + '/main/dimacs/'
_MORAL_NORMS = [
    'violate_care', 'violate_fairness', 'violate_loyalty',
    'violate_authority', 'violate_sanctity', 'violate_liberty',
]

import re

for name, value in outs.items():
    vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts = value
    if missed_flag:
        preds[name] = 'missed'; miss_count += 1; continue

    sat_norm = None
    maptxt_pth = IMAS_DIR + 'pos_' + name[:-4] + '.maptxt'
    try:
        maptxt_content = open(maptxt_pth, 'r').read()
        match = re.search(r'violate_[a-z]+_?', maptxt_content)
        if match:
            sat_norm = match.group(0).rstrip('_')
    except Exception:
        pass

    if cot_flag:
        cot_count += 1
        # Check if prob stored _predicted_norm (majority CoT vote)
        # scs[-1][0] > scs[-1][1] means majority voted for gold norm
        gold = labels.get(name, '')
        if scs:
            last = scs[-1]
            # votes[0]=correct-norm votes, votes[1]=wrong-norm votes
            preds[name] = gold if last[0] > last[1] else 'unknown'
        else:
            preds[name] = 'unknown'
    else:
        # Distinguish pre-solved (sentinel) from backbone-loop resolved
        is_presolve = (
            solout is not None
            and solout.get('pos') == [_PRESOLVE_SENTINEL]
            and solout.get('neg') == []
            and bbout is None
            and not scs
        )
        if is_presolve:
            presolve_count += 1
        else:
            sat_added_premise_count += 1

        # SAT backbone resolved: read norm from maptxt
        if solout and len(solout.get('neg', [])) == 0 and len(solout.get('pos', [])) > 0:
            preds[name] = sat_norm if sat_norm else 'unknown'
        elif solout and len(solout.get('pos', [])) == 0:
            preds[name] = 'unknown'  # neg UNSAT without tracked norm
        else:
            preds[name] = sat_norm if sat_norm else 'unknown'

sat_count = presolve_count + sat_added_premise_count

print(f'\nTotal preds: {len(preds)}')
print(f'  Pre-solved by SAT (initial UNSAT, no LLM needed) : {presolve_count}')
print(f'  SAT backbone with added premise (backbone loop)  : {sat_added_premise_count}')
print(f'  CoT fallback                                     : {cot_count}')
print(f'  Missed                                           : {miss_count}')
print(f'  [Total SAT-resolved = {sat_count}]')


# ── Step 6: Accuracy ──────────────────────────────────────────────────────────
# preds[name] = predicted norm string; labels[name] = gold_foundation norm string
acc            = 0
missed         = 0
correct_by_cot = 0
correct_by_sat = 0
ous_pred = {}

for name, pred in preds.items():
    try:
        idx      = int(name.replace('explainethics', '').split('.')[0])
        gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        gold_norm = labels.get(name, '')

    if pred == 'missed':
        missed += 1; continue

    correct = (pred == gold_norm)
    ous_pred[name] = correct
    if correct:
        acc += 1
        _, _, _, _, _, cot_flag, _, _ = outs[name]
        if cot_flag: correct_by_cot += 1
        else:        correct_by_sat  += 1

# Alias for downstream compatibility
outs_pred = ous_pred

total    = max(len(preds) - missed, 1)
accuracy = acc / total

# Per-norm confusion breakdown
norm_tp = {n: 0 for n in _MORAL_NORMS}
norm_fp = {n: 0 for n in _MORAL_NORMS}
norm_fn = {n: 0 for n in _MORAL_NORMS}

for name, pred in preds.items():
    if pred == 'missed': continue
    try:
        idx       = int(name.replace('explainethics', '').split('.')[0])
        gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        gold_norm = labels.get(name, '')
    if pred == gold_norm and pred in norm_tp:
        norm_tp[pred]   += 1
    else:
        if gold_norm in norm_fn: norm_fn[gold_norm] += 1
        if pred in norm_fp:      norm_fp[pred]      += 1

precision_micro = sum(norm_tp.values()) / max(sum(norm_tp.values()) + sum(norm_fp.values()), 1)
recall_micro    = sum(norm_tp.values()) / max(sum(norm_tp.values()) + sum(norm_fn.values()), 1)
f1_score        = 2 * precision_micro * recall_micro / max(precision_micro + recall_micro, 1e-9)
precision = precision_micro
recall    = recall_micro

print(f'\nOverall accuracy : {accuracy:.4f}  ({acc}/{total})')
print(f'Correct via SAT  : {correct_by_sat}')
print(f'Correct via CoT  : {correct_by_cot}')
print(f'Missed/skipped   : {missed}')
print(f'\nMicro-averaged standard metrics (over {len(_MORAL_NORMS)} norm classes):')
print(f'  Accuracy : {accuracy:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall   : {recall:.4f}')
print(f'  F1 Score : {f1_score:.4f}')

# ── Step 7: Per-prediction breakdown ──────────────────────────────────────────
print(f'\n{"Name":<30} {"Pred norm":<25} {"Gold norm":<25} {"Shown label":<25} {"OK?"}')
print('-' * 115)
for name, pred in preds.items():
    try:
        idx        = int(name.replace('explainethics', '').split('.')[0])
        gold_norm  = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
        shown_norm = data[idx].get('label', '??')
    except Exception:
        gold_norm = shown_norm = '??'
    ok = '✓' if pred == gold_norm else '✗'
    print(f'{name:<30} {pred:<25} {gold_norm:<25} {shown_norm:<25} {ok}')

# ── Step 8: CoT vs SAT breakdown ──────────────────────────────────────────────
cot_c = cot_t = sat_c = sat_t = 0
for name, pred in preds.items():
    if pred == 'missed': continue
    try:
        idx       = int(name.replace('explainethics', '').split('.')[0])
        gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        continue
    _, _, _, _, _, cot_flag, _, _ = outs[name]
    correct = (pred == gold_norm)
    if cot_flag: cot_t += 1; cot_c += correct
    else:        sat_t += 1; sat_c += correct

print(f'\nSAT backbone accuracy: {sat_c}/{sat_t} = {sat_c/max(sat_t,1):.3f}')
print(f'CoT fallback accuracy: {cot_c}/{cot_t} = {cot_c/max(cot_t,1):.3f}')

# ── Step 9: Per-norm accuracy (ExplainEthics-specific) ─────────────────────────
norm_correct = defaultdict(int)
norm_total   = defaultdict(int)
for name, pred in preds.items():
    if pred == 'missed': continue
    try:
        idx       = int(name.replace('explainethics', '').split('.')[0])
        gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        continue
    norm_total[gold_norm] += 1
    if pred == gold_norm:
        norm_correct[gold_norm] += 1

print('\nPer-norm accuracy (by gold_foundation):')
for norm in sorted(norm_total.keys()):
    c = norm_correct[norm]; t = norm_total[norm]
    print(f'  {norm:<25} {c}/{t} = {c/max(t,1):.3f}')
    print(f'    TP={norm_tp.get(norm,0)}  FP={norm_fp.get(norm,0)}  FN={norm_fn.get(norm,0)}')

# ── Step 10: Decoy resistance (ExplainEthics-specific) ────────────────────────
# Decoy: shown label != gold_foundation → model should predict gold_foundation, not shown label
decoy_r_c = decoy_r_t = true_r_c = true_r_t = 0
for name, pred in preds.items():
    if pred == 'missed': continue
    try:
        idx        = int(name.replace('explainethics', '').split('.')[0])
        gold_norm  = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
        shown_norm = data[idx].get('label', '').replace('-', '_').lower().strip()
    except Exception:
        continue
    is_decoy = (shown_norm != gold_norm)
    if is_decoy:
        decoy_r_t += 1
        if pred == gold_norm: decoy_r_c += 1
    else:
        true_r_t += 1
        if pred == gold_norm: true_r_c += 1

print(f'\nDecoy resistance  (shown≠gold, pred==gold): {decoy_r_c}/{decoy_r_t} = {decoy_r_c/max(decoy_r_t,1):.3f}')
print(f'True-label recall (shown==gold, pred==gold): {true_r_c}/{true_r_t} = {true_r_c/max(true_r_t,1):.3f}')

# ── Step 11: Self-consistency baseline (CoT iter files) ───────────────────────
# Note: CoT iter files store norm strings if generated with the new pipeline.
# If they store 'true'/'false', skip SC comparison (legacy format).
cot_pred_list = []
cot_accs      = []
n_votes       = [0] * len(names)
sc_votes_norm = [defaultdict(int) for _ in names]  # per-name per-norm vote count
sc_votes_total= [0] * len(names)

for i in range(20):
    pth = COT_ITER_PREFIX + str(i)
    if not os.path.exists(pth):
        break
    cot = np.load(open(pth, 'rb'), allow_pickle=True)
    cot_acc  = 0
    cot_list = []
    for j, name in enumerate(names):
        if j >= len(cot): break
        try:
            idx       = int(name.replace('explainethics', '').split('.')[0])
            gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
        except Exception:
            continue
        pred_val = str(cot[j]).lower().strip().replace('-', '_')
        correct = (pred_val == gold_norm)
        cot_acc += correct
        cot_list.append(int(correct))
        n_votes[j] += int(correct)
        sc_votes_norm[j][pred_val] += 1
        sc_votes_total[j] += 1
    print(f'CoT iter {i}: acc={cot_acc}')
    cot_accs.append(cot_acc)
    cot_pred_list.append(cot_list)

if cot_pred_list:
    sc_correct = 0
    for j, name in enumerate(names):
        if j >= len(sc_votes_total) or sc_votes_total[j] == 0: continue
        try:
            idx       = int(name.replace('explainethics', '').split('.')[0])
            gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
        except Exception:
            continue
        # Majority-voted norm
        sc_pred = max(sc_votes_norm[j], key=sc_votes_norm[j].get)
        if sc_pred == gold_norm: sc_correct += 1

    sc_acc_final = sc_correct / max(len(names), 1)
    sc_prec = sc_acc_final  # micro precision == accuracy for exact-match multi-class
    sc_rec  = sc_acc_final
    sc_f1   = sc_acc_final

    print(f'\n=== ARGOS vs Baseline (CoT SC) Comparison ===')
    print(f'Metric    | ARGOS     | CoT SC')
    print(f'-----------------------------------')
    print(f'Accuracy  | {accuracy:.4f}    | {sc_acc_final:.4f}')
    print(f'Precision | {precision:.4f}    | {sc_prec:.4f}')
    print(f'Recall    | {recall:.4f}    | {sc_rec:.4f}')
    print(f'F1 Score  | {f1_score:.4f}    | {sc_f1:.4f}')
    print(f'-----------------------------------')
    print(f'Mean single CoT acc: {np.mean(cot_accs)/max(len(names),1):.3f}')
else:
    print('\n[skip] No CoT iter files found — SC comparison skipped.')

# ── Step 12: Bootstrap resampling + Wilcoxon + t-CI ──────────────────────────
from sklearn.utils import resample
from scipy.stats  import wilcoxon, t as t_dist

outs_pred_val = np.array(list(outs_pred.values()), dtype=float)
N_BOOT        = max(len(outs_pred), 1)
BS_N          = min(86, N_BOOT)

bs_outs_acc = [np.sum(resample(outs_pred_val, n_samples=BS_N)) / BS_N
               for _ in range(N_BOOT)]

bs_sc_acc = []
if cot_pred_list:
    for _ in range(N_BOOT):
        bs_sample = resample(n_votes[:len(cot_pred_list[0])], n_samples=BS_N)
        bs_sc_acc.append(
            np.sum(np.where(np.array(bs_sample) >= np.ceil(len(cot_pred_list)/2 + 0.5),
                            1, 0)) / BS_N)

print(f'\nBootstrap ARGOS mean acc : {np.mean(bs_outs_acc):.4f}')
if bs_sc_acc:
    print(f'Bootstrap SC mean acc    : {np.mean(bs_sc_acc):.4f}')
    stat, p = wilcoxon(np.array(bs_outs_acc) - np.array(bs_sc_acc), alternative='greater')
    print(f'\nWilcoxon signed-rank (ARGOS > SC): stat={stat:.4f}  p={p:.4f}',
          '  SIGNIFICANT' if p < 0.05 else '  not significant', '(α=0.05)')

d_arr = np.array(bs_outs_acc)
ci = t_dist.interval(0.95, df=len(d_arr)-1,
                     loc=np.mean(d_arr),
                     scale=np.std(d_arr, ddof=1)/np.sqrt(len(d_arr)))
print(f'95% CI on ARGOS accuracy: ({ci[0]:.4f}, {ci[1]:.4f})')

# ── Step 13: Iteration-trajectory analysis ────────────────────────────────────
cs          = ['r', 'g', 'b', 'orange']
plot_labels = ['unflipped-wrong', 'unflipped-correct',
               'flipped correct', 'flipped incorrect']
scs_all  = []
flag_all = []
lens_all = []
first_good_flip = []
first_bad_flip  = []

for name in list(outs.keys()):
    vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts = outs[name]
    if not scs or missed_flag:
        continue
    try:
        idx       = int(name.replace('explainethics', '').split('.')[0])
        gold_norm = data[idx]['gold_foundation'].replace('-', '_').lower().strip()
    except Exception:
        continue

    mat = torch.stack(scs) / torch.stack(scs).sum(1).reshape(-1, 1)
    mat = mat[:, 0]  # p(correct norm) = votes[0] / total votes

    # Prepend SC vote if available
    j = name_idx.get(name, None)
    if j is not None and j < len(n_votes) and cot_pred_list:
        n_sc = len(cot_pred_list)
        sc_p = n_votes[j] / n_sc  # fraction of SC iters that voted the gold norm
        mat  = torch.cat([torch.tensor([sc_p]), mat])

    lens_all.append(len(mat) - 1)

    start_correct = (mat[0] > 0.5)
    end_correct   = (mat[-1] > 0.5)

    if   start_correct and end_correct:       flag_all.append(1)
    elif not start_correct and not end_correct: flag_all.append(0)
    elif not start_correct and end_correct:
        flag_all.append(2)
        for z in range(len(mat)):
            # mat[z] = p(correct norm); good flip = model crosses into correct territory
            if mat[z] > 0.5:
                first_good_flip.append(z); break
    else:
        flag_all.append(3)
        for z in range(len(mat)):
            # bad flip = model crosses out of correct territory
            if mat[z] < 0.5:
                first_bad_flip.append(z); break

    scs_all.append(mat.clone())

flag_counts = np.unique(flag_all, return_counts=True) if flag_all else ([], [])
print(f'\nTrajectory flags:')
for fi, fc in zip(flag_counts[0], flag_counts[1]):
    print(f'  {plot_labels[int(fi)]}: {fc}')

# ── Step 14: Confidence trajectory plot ───────────────────────────────────────
if scs_all:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i in range(len(scs_all)):
        ax1.plot(scs_all[i].numpy(), c=cs[int(flag_all[i])], alpha=0.4)
    ax1.plot([1.0 - k*0.1 for k in range(6)], '--', c='black')
    ax1.plot([0.0 + k*0.1 for k in range(6)], '--', c='black')
    ax1.set_title(f'p(norm violated) confidence over iterations (n={len(scs_all)})')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('p(True/Violated)')
    patches = [mpatches.Patch(color=cs[int(fi)], label=f'{plot_labels[int(fi)]} (n={fc})')
               for fi, fc in zip(flag_counts[0], flag_counts[1])]
    ax1.legend(handles=patches)
    fig1.savefig(ANALYSIS_OUT_DIR + '/explainethics_trajectories.pdf', bbox_inches='tight')
    plt.show()

# ── Step 15: First-flip histograms ────────────────────────────────────────────
if first_good_flip:
    fig3, ax3 = plt.subplots()
    ax3.hist(first_good_flip, label='First Good Flip')
    ax3.set_title('Iteration of First Good Flip'); ax3.set_xlabel('Iteration')
    fig3.savefig(ANALYSIS_OUT_DIR + '/explainethics_good_flips.pdf', bbox_inches='tight')
if first_bad_flip:
    fig4, ax4 = plt.subplots()
    ax4.hist(first_bad_flip, label='First Bad Flip')
    ax4.set_title('Iteration of First Bad Flip'); ax4.set_xlabel('Iteration')
    fig4.savefig(ANALYSIS_OUT_DIR + '/explainethics_bad_flips.pdf', bbox_inches='tight')
if first_good_flip or first_bad_flip:
    plt.show()

# ── Step 16: Normalised stacked histogram ─────────────────────────────────────
if scs_all:
    lvsf   = [[], [], [], []]
    totals = []
    for i in range(len(scs_all)):
        l = len(scs_all[i]) - 1
        lvsf[flag_all[i]].append(l)
        totals.append(l)

    bins       = np.array([1,2,3,4,5,6,7])
    counts_all = np.vstack([np.histogram(lvsf[j], bins=bins)[0] for j in range(4)])
    col_sums   = counts_all.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    norm_counts = counts_all / col_sums
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width       = np.diff(bins)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    bottom = np.zeros_like(bin_centers, dtype=float)
    for j in range(4):
        ax5.bar(bin_centers, norm_counts[j], width=width, bottom=bottom,
                color=cs[j], label=plot_labels[j], edgecolor='black')
        bottom += norm_counts[j]
    ax5.set_ylabel('Proportion'); ax5.set_xlabel('# ARGOS Iterations Before Exit')
    ax5.set_xticks(bins[:-1])
    for sp in ['top', 'right', 'left']: ax5.spines[sp].set_visible(False)
    ax5.set_yticks([])
    ____, totalcounts = np.unique(totals, return_counts=True)
    ax_top = fig5.add_axes([0.125, 0.85, 0.775, 0.2])
    ax_top.hist(totals, bins=bins, alpha=0.6)
    ax_top.set_ylabel('Total count')
    for sp in ['top', 'right', 'bottom', 'left']: ax_top.spines[sp].set_visible(False)
    ax_top.set_yticks([]); ax_top.set_xticks([])
    patches = [mpatches.Patch(color=cs[int(fi)], label=f'{plot_labels[int(fi)]} (n={fc})')
               for fi, fc in zip(flag_counts[0], flag_counts[1])]
    ax_top.legend(handles=patches, bbox_to_anchor=(0.3, 0.5))
    fig5.savefig(ANALYSIS_OUT_DIR + '/explainethics_lenhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 17: 2D histogram (iteration × confidence) ────────────────────────────
if scs_all:
    from matplotlib.colors import LightSource
    fx_by_flag = [[], [], [], []]
    fy_by_flag = [[], [], [], []]
    for j in range(len(scs_all)):
        for i_step in range(len(scs_all[j])):
            fx_by_flag[flag_all[j]].append(i_step)
            fy_by_flag[flag_all[j]].append(float(scs_all[j][i_step]))

    fhist = [np.histogram2d(fx_by_flag[i], fy_by_flag[i],
                            bins=[[2,3,4,5,6,7],[0,0.2,0.4,0.6,0.8,1]])[0]
             for i in range(4)]

    xedges = [2,3,4,5,6,7]; yedges = [0,0.2,0.4,0.6,0.8,1]
    xpos_m, ypos_m = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    xpos_m = xpos_m.ravel(); ypos_m = ypos_m.ravel() * 100

    fig6 = plt.figure()
    ax6  = fig6.add_subplot(projection='3d')
    ax6.view_init(elev=40, azim=320, roll=0)
    for j in range(len(xpos_m)):
        cumhist = 0
        for i in [3, 0, 2, 1]:
            dz = fhist[i].ravel()[j]
            ax6.bar3d(xpos_m[j], ypos_m[j], cumhist, 0.5, 10, dz,
                      zorder=0, color=cs[i], lightsource=LightSource(azdeg=190))
            cumhist += dz
    ax6.set_xlabel('Iteration Number'); ax6.set_ylabel('confidence')
    ax6.set_title('Histogram of Confidences as ARGOS Iterates (ExplainEthics)')
    ax6.set_yticklabels([f'{i}%' for i in [-100,-60,-20,20,60,100]])
    ax6.set_ylim(0, 101)
    patches = [mpatches.Patch(color=cs[i], label=plot_labels[i]) for i in range(4)]
    ax6.legend(handles=patches, bbox_to_anchor=(1.3, 1))
    fig6.savefig(ANALYSIS_OUT_DIR + '/explainethics_threedhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 18: Summary bar chart ────────────────────────────────────────────────
categories  = ['Correct (SAT)', 'Correct (CoT)', 'Incorrect', 'Missed']
bar_values  = [correct_by_sat, correct_by_cot, total - acc, missed]
bar_colors  = ['#2ecc71', '#27ae60', '#e74c3c', '#95a5a6']

fig7, ax7 = plt.subplots(figsize=(8, 4))
bars = ax7.bar(categories, bar_values, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, bar_values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             str(val), ha='center', va='bottom', fontweight='bold')
ax7.set_title('ExplainEthics Pipeline Results')
ax7.set_ylabel('Count')
fig7.tight_layout()
fig7.savefig(ANALYSIS_OUT_DIR + '/explainethics_results.png', dpi=150)
plt.show()

print('\n[Done] Available: preds, labels, outs, data, names, scs_all, flag_all, bs_outs_acc')
