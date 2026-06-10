"""
ProntoQA Analysis Script
========================
Equivalent analysis to run_analysis_explainethics.py but for ProntoQA.
Run interactively:  python3 -i run_analysis_pronto.py

Statistical methods:
  1. Self-consistency majority vote  (CoT iter files → sc_acc)
  2. Bootstrap resampling            (sklearn, n=86)
  3. Wilcoxon signed-rank test       (scipy, ARGOS > SC)
  4. t-interval 95% CI               (scipy.stats.t.interval)
  5. Confusion matrix TP/TN/FP/FN
  6. Iteration-trajectory flip analysis (scs confidence over iterations)
  7. 2D histogram  (iteration × confidence)
  8. Normalised stacked histogram    (proportion per outcome per iteration)

all_outs[name] structure (key = 'prontoN.cnf'):
  [0] vv          – inferred variable tuples
  [1] solout      – {'pos': [...], 'neg': [...]}
  [2] bbout       – backbone after last iter
  [3] missed_flag – True if skipped
  [4] sc_scores   – list of probability tensors over iterations (equivalent to scs)

Note: ProntoQA uses a binary (true/false) label scheme.
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
DATASET_PATH = BASE_PATH + '/SAT-LM/data/pronto_test.json'
DIMACS_DIR   = BASE_PATH + '/main/dimacs'
CSV_PATH     = BASE_PATH + '/main/dimacs_csvs/solver_finished.csv'
LABELS_CSV   = BASE_PATH + '/main/pronto_labels.csv'
COT_ITER_PREFIX = BASE_PATH + '/preds/FewShotCOTProNToQA_iter'

import sys
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush(); self.log.flush()

ANALYSIS_OUT_DIR = BASE_PATH + '/main/analysis/analysis_outputs'
os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
sys.stdout = Logger(ANALYSIS_OUT_DIR + '/pronto_analysis_report.txt')

# ── Step 1: Find pkl files ─────────────────────────────────────────────────────
pkl_files = sorted(glob.glob(BASE_PATH + '/all_outs_cot_met_pronto*.pkl'))
print('Available ProntoQA pkl files:')
for f in pkl_files:
    print(' ', f)

# ── Step 2: Load pkl ───────────────────────────────────────────────────────────
if pkl_files:
    PKL_PATH = pkl_files[-1]
    outs = pkl.load(open(PKL_PATH, 'rb'))
    print(f'\nLoaded {len(outs)} examples from {os.path.basename(PKL_PATH)}')
else:
    print('No pkl files found. Run cot_met_pronto.py first.')
    outs = {}

# ── Step 3: Load dataset ───────────────────────────────────────────────────────
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)
print(f'Dataset: {len(data)} ProntoQA examples')

# ── Step 4: Load solver CSV → names / labels ──────────────────────────────────
names  = []
labels = {}

with open(CSV_PATH, 'r') as cf:
    cr = csv.reader(cf)
    for row in cr:
        if len(row) < 4: continue
        if not row[1].startswith('pronto'): continue
        if row[2] == 'SAT' and row[3] == 'SAT':
            names.append(row[1])
            try:
                idx = int(row[1].replace('pronto', '').split('.')[0])
                labels[row[1]] = data[idx]['label'].lower().strip()
            except Exception:
                pass

print(f'Names from CSV: {len(names)}, Labels: {len(labels)}')

if os.path.exists(LABELS_CSV):
    with open(LABELS_CSV, 'r') as lf:
        for row in csv.reader(lf):
            if len(row) < 2: continue
            cnf = row[0][:-2] + 'cnf'
            neg_path = os.path.join(DIMACS_DIR, 'neg_' + cnf)
            if not os.path.exists(neg_path): continue
            if cnf not in names: continue
            labels[cnf] = row[1].lower().strip()
    print(f'Labels after CSV override: {len(labels)}')

name_idx = {name: i for i, name in enumerate(names)}

# ── Step 5: Build preds ────────────────────────────────────────────────────────
# NOTE: ProntoQA all_outs has 5 fields: (vv, solout, bbout, missed_flag, sc_scores)
# sc_scores plays the role of scs in CLUTRR/ExplainEthics.
preds                   = {}
cot_count               = 0
presolve_count          = 0
sat_added_premise_count = 0
miss_count              = 0
_PRESOLVE_SENTINEL      = 'dummy'

for name, value in outs.items():
    # Unpack — handle both 5-field (pronto) and 8-field (clutrr-style) tuples
    if len(value) == 5:
        vv, solout, bbout, missed_flag, scs = value
        cot_flag = False  # pronto resolves via backbone or misses; no dedicated cot_flag
    else:
        vv, solout, bbout, missed_flag, _, cot_flag, scs, _ = value

    if missed_flag:
        preds[name] = 'missed'; miss_count += 1; continue

    if cot_flag:
        cot_count += 1
        preds[name] = ('true' if scs[-1].argmax() == 0 else 'false') if scs else 'missed'
    else:
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

        if   len(solout.get('neg', [])) == 0: preds[name] = 'false'
        elif len(solout.get('pos', [])) == 0: preds[name] = 'true'
        else:                                  preds[name] = 'missed'

sat_count = presolve_count + sat_added_premise_count

print(f'\nTotal preds: {len(preds)}')
print(f'  Pre-solved by SAT (initial UNSAT, no LLM needed) : {presolve_count}')
print(f'  SAT backbone with added premise (backbone loop)  : {sat_added_premise_count}')
print(f'  CoT fallback                                     : {cot_count}')
print(f'  Missed                                           : {miss_count}')
print(f'  [Total SAT-resolved = {sat_count}]')

# ── Step 6: Accuracy + confusion matrix ───────────────────────────────────────
acc            = 0
missed         = 0
correct_by_cot = 0
correct_by_sat = 0
true_pos = false_pos = true_neg = false_neg = 0
n_true = n_false = 0
outs_pred = {}

for name, pred in preds.items():
    label = labels.get(name, '').lower().strip()
    if label == 'true':  n_true  += 1
    if label == 'false': n_false += 1
    if pred == 'missed':
        missed += 1; continue
    correct = (pred == label)
    outs_pred[name] = correct
    if correct:
        acc += 1
        entry = outs[name]
        cf = entry[5] if len(entry) == 8 else False
        if cf: correct_by_cot += 1
        else:  correct_by_sat  += 1
        if label == 'true':  true_pos  += 1
        else:                true_neg  += 1
    else:
        if label == 'true':  false_neg += 1
        else:                false_pos += 1

total = max(len(preds) - missed, 1)
print(f'\nOverall accuracy : {acc/total:.4f}  ({acc}/{total})')
print(f'Correct via SAT  : {correct_by_sat}')
print(f'Correct via CoT  : {correct_by_cot}')
print(f'Missed/skipped   : {missed}')
print(f'\nConfusion matrix:')
print(f'  TP={true_pos}  FP={false_pos}  TN={true_neg}  FN={false_neg}')
print(f'  n_true={n_true}  n_false={n_false}')

precision = true_pos / max(true_pos + false_pos, 1)
recall    = true_pos / max(true_pos + false_neg, 1)
f1_score  = 2 * precision * recall / max(precision + recall, 1e-9)
print(f'\nPrecision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1 Score : {f1_score:.4f}')

# ── Step 7: Per-prediction detail (first 30) ──────────────────────────────────
print(f'\n{"Name":<30} {"Pred":<8} {"Label":<8} {"OK?"}')
print('-' * 55)
for name, pred in list(preds.items())[:30]:
    label = labels.get(name, '??').strip()
    ok = '✓' if pred == label else '✗'
    print(f'{name:<30} {pred:<8} {label:<8} {ok}')

# ── Step 8: CoT vs SAT breakdown ──────────────────────────────────────────────
cot_c = cot_t = sat_c = sat_t = 0
for name, pred in preds.items():
    if pred == 'missed': continue
    label = labels.get(name, '').lower().strip()
    entry = outs[name]
    cf = entry[5] if len(entry) == 8 else False
    correct = (pred == label)
    if cf: cot_t += 1; cot_c += correct
    else:  sat_t += 1; sat_c += correct

print(f'\nSAT backbone accuracy: {sat_c}/{sat_t} = {sat_c/max(sat_t,1):.3f}')
print(f'CoT fallback accuracy: {cot_c}/{cot_t} = {cot_c/max(cot_t,1):.3f}')

# ── Step 9: Self-consistency baseline (CoT iter files) ────────────────────────
cot_pred_list = []
cot_accs      = []
n_votes       = [0] * len(names)

for i in range(20):
    pth = COT_ITER_PREFIX + str(i)
    if not os.path.exists(pth):
        break
    cot = np.load(open(pth, 'rb'), allow_pickle=True)
    cot_acc  = 0
    cot_list = []
    for j, name in enumerate(names):
        if j >= len(cot): break
        correct = (cot[j] == labels.get(name, '').strip())
        cot_acc  += correct
        cot_list.append(int(correct))
        n_votes[j] += int(correct)
    print(f'CoT iter {i}: acc={cot_acc}')
    cot_accs.append(cot_acc)
    cot_pred_list.append(cot_list)

if cot_pred_list:
    sc_acc = np.sum(np.where(np.array(n_votes[:len(cot_pred_list[0])]) >=
                             np.ceil(len(cot_pred_list)/2 + 0.5), 1, 0))
    sc_acc_final = sc_acc / max(len(names), 1)
    print(f'\n=== ARGOS vs Baseline (CoT SC) Comparison ===')
    print(f'Metric    | ARGOS     | CoT SC')
    print(f'-----------------------------------')
    print(f'Accuracy  | {acc/total:.4f}    | {sc_acc_final:.4f}')
    print(f'Precision | {precision:.4f}    | {sc_acc_final:.4f}')
    print(f'Recall    | {recall:.4f}    | {sc_acc_final:.4f}')
    print(f'F1 Score  | {f1_score:.4f}    | {sc_acc_final:.4f}')
    print(f'-----------------------------------')
    print(f'Mean single CoT acc: {np.mean(cot_accs)/max(len(names),1):.3f}')
else:
    print('\n[skip] No CoT iter files found — SC comparison skipped.')

# ── Step 10: Bootstrap resampling + Wilcoxon + t-CI ──────────────────────────
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

# ── Step 11: Iteration-trajectory analysis ────────────────────────────────────
cs          = ['r', 'g', 'b', 'orange']
plot_labels = ['unflipped-wrong', 'unflipped-correct',
               'flipped correct', 'flipped incorrect']
scs_all  = []
flag_all = []
lens_all = []
first_good_flip = []
first_bad_flip  = []

for name in list(outs.keys()):
    entry = outs[name]
    scs = entry[4] if len(entry) == 5 else entry[6]
    missed_flag = entry[3]
    if not scs or missed_flag:
        continue
    label = labels.get(name, '').lower().strip()

    mat = torch.stack(scs) / torch.stack(scs).sum(1).reshape(-1, 1)
    mat = mat[:, 0]

    j = name_idx.get(name, None)
    if j is not None and j < len(n_votes) and cot_pred_list:
        n_sc = len(cot_pred_list)
        sc_p = n_votes[j] / n_sc if label == 'true' else 1 - n_votes[j] / n_sc
        mat  = torch.cat([torch.tensor([sc_p]), mat])

    lens_all.append(len(mat) - 1)

    if label == 'true':
        start_correct = (mat[0] > 0.5)
        end_correct   = (mat[-1] > 0.5)
    else:
        start_correct = (mat[0] < 0.5)
        end_correct   = (mat[-1] < 0.5)

    if start_correct and end_correct:       flag_all.append(1)
    elif not start_correct and not end_correct: flag_all.append(0)
    elif not start_correct and end_correct:
        flag_all.append(2)
        for z in range(len(mat)):
            cross = (mat[z] > 0.5) if label == 'true' else (mat[z] < 0.5)
            if cross: first_good_flip.append(z); break
    else:
        flag_all.append(3)
        for z in range(len(mat)):
            cross = (mat[z] < 0.5) if label == 'true' else (mat[z] > 0.5)
            if cross: first_bad_flip.append(z); break

    scs_all.append(mat.clone())

flag_counts = np.unique(flag_all, return_counts=True) if flag_all else ([], [])
print(f'\nTrajectory flags:')
for fi, fc in zip(flag_counts[0], flag_counts[1]):
    print(f'  {plot_labels[int(fi)]}: {fc}')

# ── Step 12: Confidence trajectory plot ───────────────────────────────────────
if scs_all:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i in range(len(scs_all)):
        ax1.plot(scs_all[i].numpy(), c=cs[int(flag_all[i])], alpha=0.4)
    ax1.axhline(y=0.5, linestyle='--', c='black', linewidth=1.2, label='Decision boundary (0.5)')
    ax1.set_title(f'"True" confidence over iterations (ProntoQA, n={len(scs_all)})')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('p(True)')
    ax1.set_ylim(-0.05, 1.05)
    patches = [mpatches.Patch(color=cs[int(fi)], label=f'{plot_labels[int(fi)]} (n={fc})')
               for fi, fc in zip(flag_counts[0], flag_counts[1])]
    ax1.legend(handles=patches)
    fig1.savefig(ANALYSIS_OUT_DIR + '/pronto_trajectories.pdf', bbox_inches='tight')
    plt.show()

# ── Step 13: First-flip histograms ────────────────────────────────────────────
if first_good_flip:
    fig3, ax3 = plt.subplots()
    ax3.hist(first_good_flip, label='First Good Flip')
    ax3.set_title('Iteration of First Good Flip (ProntoQA)'); ax3.set_xlabel('Iteration')
    fig3.savefig(ANALYSIS_OUT_DIR + '/pronto_good_flips.pdf', bbox_inches='tight')
if first_bad_flip:
    fig4, ax4 = plt.subplots()
    ax4.hist(first_bad_flip, label='First Bad Flip')
    ax4.set_title('Iteration of First Bad Flip (ProntoQA)'); ax4.set_xlabel('Iteration')
    fig4.savefig(ANALYSIS_OUT_DIR + '/pronto_bad_flips.pdf', bbox_inches='tight')
if first_good_flip or first_bad_flip:
    plt.show()

# ── Step 14: Normalised stacked histogram ─────────────────────────────────────
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
    ax_top = fig5.add_axes([0.125, 0.85, 0.775, 0.2])
    ax_top.hist(totals, bins=bins, alpha=0.6)
    ax_top.set_ylabel('Total count')
    for sp in ['top', 'right', 'bottom', 'left']: ax_top.spines[sp].set_visible(False)
    ax_top.set_yticks([]); ax_top.set_xticks([])
    patches = [mpatches.Patch(color=cs[int(fi)], label=f'{plot_labels[int(fi)]} (n={fc})')
               for fi, fc in zip(flag_counts[0], flag_counts[1])]
    ax_top.legend(handles=patches, bbox_to_anchor=(0.3, 0.5))
    fig5.savefig(ANALYSIS_OUT_DIR + '/pronto_lenhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 15: 2D histogram (iteration × confidence) ────────────────────────────
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
    ax6.set_xlabel('Iteration Number'); ax6.set_ylabel('Confidence')
    ax6.set_title('Histogram of Confidences as ARGOS Iterates (ProntoQA)')
    ax6.set_yticklabels([f'{int(i*100)}%' for i in yedges])
    ax6.set_ylim(0, 101)
    patches = [mpatches.Patch(color=cs[i], label=plot_labels[i]) for i in range(4)]
    ax6.legend(handles=patches, bbox_to_anchor=(1.3, 1))
    fig6.savefig(ANALYSIS_OUT_DIR + '/pronto_threedhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 16: Summary bar chart ────────────────────────────────────────────────
categories  = ['Correct (SAT)', 'Correct (CoT)', 'Incorrect', 'Missed']
bar_values  = [correct_by_sat, correct_by_cot, total - acc, missed]
bar_colors  = ['#2ecc71', '#27ae60', '#e74c3c', '#95a5a6']

fig7, ax7 = plt.subplots(figsize=(8, 4))
bars = ax7.bar(categories, bar_values, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, bar_values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             str(val), ha='center', va='bottom', fontweight='bold')
ax7.set_title('ProntoQA Pipeline Results')
ax7.set_ylabel('Count')
fig7.tight_layout()
fig7.savefig(ANALYSIS_OUT_DIR + '/pronto_results.png', dpi=150)
plt.show()

# ── Step 17: Resolution pathway pie chart ─────────────────────────────────────
pie_labels = ['Pre-solved by SAT', 'SAT backbone (loop)', 'CoT fallback', 'Missed']
pie_values = [presolve_count, sat_added_premise_count, cot_count, miss_count]
pie_colors = ['#3498db', '#2980b9', '#e67e22', '#95a5a6']
pie_values_nonzero = [(v, l, c) for v, l, c in zip(pie_values, pie_labels, pie_colors) if v > 0]
if pie_values_nonzero:
    pv, pl, pc = zip(*pie_values_nonzero)
    fig8, ax8 = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax8.pie(
        pv, labels=pl, colors=pc, autopct='%1.1f%%',
        startangle=140, pctdistance=0.8
    )
    for t in autotexts: t.set_fontsize(9)
    ax8.set_title('Resolution Pathway Distribution (ProntoQA)')
    fig8.savefig(ANALYSIS_OUT_DIR + '/pronto_resolution_pie.png', dpi=150, bbox_inches='tight')
    plt.show()

print('\n[Done] Available: preds, labels, outs, data, names, scs_all, flag_all, bs_outs_acc')
