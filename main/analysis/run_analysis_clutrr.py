"""
CLUTRR Analysis Script
======================
Equivalent of analysis_clutrr.ipynb — corrected local paths + full statistical pipeline.
Run interactively:  python3 -i run_analysis_clutrr.py

Statistical methods used (matching analysis_clutrr.ipynb):
  1. Self-consistency majority vote  (20 CoT iters → sc_acc)
  2. Bootstrap resampling            (sklearn, n=86)
  3. Wilcoxon signed-rank test       (scipy, ARGOS > SC)
  4. t-interval 95% CI               (scipy.stats.t.interval)
  5. Confusion matrix TP/TN/FP/FN
  6. Iteration-trajectory flip analysis (scs confidence over iterations)
  7. 2D histogram  (iteration × confidence)
  8. Normalised stacked histogram    (proportion per outcome per iteration)

all_outs[name] structure:
  [0] vv          – inferred variable tuples
  [1] solout      – {'pos': [...], 'neg': [...]}
  [2] bbout       – backbone after last iter
  [3] missed_flag – True if skipped
  [4] rule_scores – rule → score
  [5] cot_flag    – True if resolved by CoT
  [6] scs         – list of probability tensors over iterations
  [7] prompts     – prompts used
"""

import pickle as pkl
import json
import csv
import os
import glob
import copy
import numpy as np
import torch
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.size'] = 12

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH    = '/mnt/c/Tugas_Akhir/ARGOS_public_anon'
DATASET_PATH = BASE_PATH + '/SAT-LM/data/clutrr_test.json'
DIMACS_DIR   = BASE_PATH + '/main/dimacs'
CSV_PATH     = BASE_PATH + '/main/dimacs_csvs/solver_finished.csv'
LABELS_CSV   = BASE_PATH + '/main/clutrr_labels.csv'
# Optional: directory of CoT baseline iter files (FewShotCOTCLUTRR_iter0 … iter19)
COT_ITER_PREFIX = BASE_PATH + '/preds/FewShotCOTCLUTRR_iter'

# ── Step 1: Find pkl files ─────────────────────────────────────────────────────
pkl_files = sorted(glob.glob(BASE_PATH + '/all_outs_cot_met_clutrr*.pkl'))
print('Available CLUTRR pkl files:')
for f in pkl_files:
    print(' ', f)

# ── Step 2: Load pkl ───────────────────────────────────────────────────────────
if pkl_files:
    PKL_PATH = pkl_files[-1]  # newest; change index if needed
    outs = pkl.load(open(PKL_PATH, 'rb'))
    print(f'\nLoaded {len(outs)} examples from {os.path.basename(PKL_PATH)}')
else:
    print('No pkl files found. Run cot_met_clutrr.py first.')
    outs = {}

# ── Step 3: Load dataset ───────────────────────────────────────────────────────
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)
print(f'Dataset: {len(data)} CLUTRR examples')

# ── Step 4: Load solver CSV → names / labels ──────────────────────────────────
noisy_data = []
mistr_data = []
names      = []
labels     = {}

with open(CSV_PATH, 'r') as cf:
    cr = csv.reader(cf)
    for row in cr:
        if len(row) < 4: continue
        if row[1] in noisy_data or row[1] in mistr_data: continue
        if not row[1].startswith('clutrr'): continue
        if row[2] == 'SAT' and row[3] == 'SAT':
            names.append(row[1])
            try:
                idx = int(row[1].replace('clutrr', '').split('.')[0])
                labels[row[1]] = data[idx]['label']
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

# name → position index (needed for SC vote lookup)
name_idx = {name: i for i, name in enumerate(names)}

# ── Step 5: Build preds ────────────────────────────────────────────────────────
preds      = {}
cot_count  = 0
sat_count  = 0
miss_count = 0

for name, value in outs.items():
    vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts = value
    if missed_flag:
        preds[name] = 'missed'; miss_count += 1; continue
    if cot_flag:
        cot_count += 1
        preds[name] = ('true' if scs[-1].argmax() == 0 else 'false') if scs else 'missed'
    else:
        sat_count += 1
        if   len(solout['neg']) == 0: preds[name] = 'false'
        elif len(solout['pos']) == 0: preds[name] = 'true'
        else:                         preds[name] = 'missed'

print(f'\nTotal preds: {len(preds)}')
print(f'  SAT backbone: {sat_count}')
print(f'  CoT fallback: {cot_count}')
print(f'  Missed:       {miss_count}')

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
        _, _, _, _, _, cot_flag, _, _ = outs[name]
        if cot_flag: correct_by_cot += 1
        else:        correct_by_sat  += 1
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

# ── Step 7: Per-prediction detail (first 30) ───────────────────────────────────
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
    _, _, _, _, _, cot_flag, _, _ = outs[name]
    correct = (pred == label)
    if cot_flag: cot_t += 1; cot_c += correct
    else:        sat_t += 1; sat_c += correct

print(f'\nSAT backbone accuracy: {sat_c}/{sat_t} = {sat_c/max(sat_t,1):.3f}')
print(f'CoT fallback accuracy: {cot_c}/{cot_t} = {cot_c/max(cot_t,1):.3f}')

# ── Step 9: Self-consistency baseline (CoT iter files) ────────────────────────
#   Needs:  {COT_ITER_PREFIX}0 … {COT_ITER_PREFIX}19
#   Each file is a numpy array of predictions aligned to `names`
cot_pred_list = []
cot_accs      = []
n_votes       = [0] * len(names)

for i in range(20):
    pth = COT_ITER_PREFIX + str(i)
    if not os.path.exists(pth):
        break
    cot = np.load(open(pth, 'rb'), allow_pickle=True)
    cot_acc   = 0
    cot_preds = {}
    cot_list  = []
    for j, name in enumerate(names):
        if j >= len(cot): break
        correct = (cot[j] == labels.get(name, '').strip())
        cot_acc  += correct
        cot_preds[name] = correct
        cot_list.append(int(correct))
        n_votes[j] += int(correct)
    print(f'CoT iter {i}: acc={cot_acc}')
    cot_accs.append(cot_acc)
    cot_pred_list.append(cot_list)

if cot_pred_list:
    sc_acc = np.sum(np.where(np.array(n_votes[:len(cot_pred_list[0])]) >=
                             np.ceil(len(cot_pred_list)/2 + 0.5), 1, 0))
    print(f'\nSelf-consistency acc: {sc_acc}/{len(names)} = {sc_acc/max(len(names),1):.3f}')
    print(f'Mean CoT acc:          {np.mean(cot_accs)/max(len(names),1):.3f}')
else:
    print('\n[skip] No CoT iter files found — SC comparison skipped.')
    n_votes = [0] * len(names)

# ── Step 10: Bootstrap resampling + Wilcoxon + t-CI ──────────────────────────
from sklearn.utils import resample
from scipy.stats  import wilcoxon, t as t_dist

outs_pred_val = np.array(list(outs_pred.values()), dtype=float)
N_BOOT        = len(outs_pred)
BS_N          = min(86, N_BOOT)

bs_outs_acc = []
for _ in range(N_BOOT):
    bs_outs_acc.append(np.sum(resample(outs_pred_val, n_samples=BS_N)) / BS_N)

bs_sc_acc = []
if cot_pred_list:
    for _ in range(N_BOOT):
        bs_sample = resample(n_votes[:len(cot_pred_list[0])], n_samples=BS_N)
        bs_sc_acc.append(
            np.sum(np.where(np.array(bs_sample) >= np.ceil(len(cot_pred_list)/2 + 0.5),
                            1, 0)) / BS_N
        )

print(f'\nBootstrap ARGOS mean acc : {np.mean(bs_outs_acc):.4f}')
if bs_sc_acc:
    print(f'Bootstrap SC mean acc    : {np.mean(bs_sc_acc):.4f}')
    stat, p = wilcoxon(np.array(bs_outs_acc) - np.array(bs_sc_acc), alternative='greater')
    print(f'\nWilcoxon signed-rank test (ARGOS > SC):')
    print(f'  statistic={stat:.4f}  p={p:.4f}  {"SIGNIFICANT" if p<0.05 else "not significant"} (α=0.05)')

# 95% t-interval on ARGOS bootstrap
d_arr = np.array(bs_outs_acc)
ci = t_dist.interval(0.95, df=len(d_arr)-1,
                     loc=np.mean(d_arr),
                     scale=np.std(d_arr, ddof=1)/np.sqrt(len(d_arr)))
print(f'\n95% CI on ARGOS accuracy: ({ci[0]:.4f}, {ci[1]:.4f})')

# ── Step 11: Iteration-trajectory analysis ────────────────────────────────────
#   For each example, track scs (confidence in "true") over backbone iterations.
#   Label each trajectory as one of 4 flags:
#     0 = unflipped-wrong   (stayed wrong side of 0.5)
#     1 = unflipped-correct (stayed right side of 0.5)
#     2 = flipped correct   (crossed 0.5 from wrong → right)
#     3 = flipped incorrect (crossed 0.5 from right → wrong)
scs_all  = []
flag_all = []
lens_all = []
first_good_flip = []
first_bad_flip  = []

cs           = ['r', 'g', 'b', 'orange']
plot_labels  = ['unflipped-wrong', 'unflipped-correct',
                'flipped correct', 'flipped incorrect']

for name in list(outs.keys()):
    vv, solout, bbout, missed_flag, rule_scores, cot_flag, scs, prompts = outs[name]
    if not scs or missed_flag:
        continue
    label = labels.get(name, '').lower().strip()

    # Stack scs tensors → normalise to p(true)
    mat = torch.stack(scs) / torch.stack(scs).sum(1).reshape(-1, 1)
    mat = mat[:, 0]  # p(true) column

    # Prepend SC vote as step 0 (if available)
    j = name_idx.get(name, None)
    if j is not None and j < len(n_votes) and cot_pred_list:
        n_sc = len(cot_pred_list)
        sc_p = n_votes[j] / n_sc if label == 'true' else 1 - n_votes[j] / n_sc
        mat  = torch.cat([torch.tensor([sc_p]), mat])

    lens_all.append(len(mat) - 1)

    # Classify trajectory
    if label == 'true':
        start_correct = (mat[0] > 0.5)
        end_correct   = (mat[-1] > 0.5)
    else:
        start_correct = (mat[0] < 0.5)
        end_correct   = (mat[-1] < 0.5)

    if start_correct and end_correct:
        flag_all.append(1)
    elif not start_correct and not end_correct:
        flag_all.append(0)
    elif not start_correct and end_correct:
        flag_all.append(2)
        threshold = 0.5 if label == 'true' else 0.5
        for z in range(len(mat)):
            cross = (mat[z] > 0.5) if label == 'true' else (mat[z] < 0.5)
            if cross:
                first_good_flip.append(z); break
    else:  # start correct, end wrong
        flag_all.append(3)
        for z in range(len(mat)):
            cross = (mat[z] < 0.5) if label == 'true' else (mat[z] > 0.5)
            if cross:
                first_bad_flip.append(z); break

    scs_all.append(mat.clone())

flag_counts = np.unique(flag_all, return_counts=True)
print(f'\nTrajectory flags:')
for fi, fc in zip(flag_counts[0], flag_counts[1]):
    print(f'  {plot_labels[fi]}: {fc}')

# ── Step 12: Confidence trajectory plot ───────────────────────────────────────
if scs_all:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i in range(len(scs_all)):
        col = cs[int(flag_all[i])]
        ax1.plot(scs_all[i].numpy(), c=col, alpha=0.4,
                 label=plot_labels[int(flag_all[i])])
    # 0.5 reference lines
    line_down = [1.0 - k * 0.1 for k in range(6)]
    line_up   = [0.0 + k * 0.1 for k in range(6)]
    ax1.plot(line_down, '--', c='black')
    ax1.plot(line_up,   '--', c='black')
    ax1.set_title(f'"True" classification confidence over iterations (n={len(scs_all)})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('p(True)')
    patches = [mpatches.Patch(color=cs[i], label=f'{plot_labels[i]} (n={fc})')
               for i, fc in zip(flag_counts[0], flag_counts[1])]
    ax1.legend(handles=patches)
    fig1.savefig(BASE_PATH + '/main/analysis/clutrr_trajectories.pdf', bbox_inches='tight')
    plt.show()

# ── Step 13: First-flip histograms ────────────────────────────────────────────
if first_good_flip or first_bad_flip:
    fig3, ax3 = plt.subplots()
    ax3.hist(first_good_flip, label='First Good Flip Iteration')
    ax3.set_title('Iteration of Good Flip')
    ax3.set_xlabel('Iteration')
    fig3.savefig(BASE_PATH + '/main/analysis/clutrr_good_flips.pdf', bbox_inches='tight')

    fig4, ax4 = plt.subplots()
    ax4.hist(first_bad_flip, label='First Bad Flip Iteration')
    ax4.set_title('Iteration of Bad Flip')
    ax4.set_xlabel('Iteration')
    fig4.savefig(BASE_PATH + '/main/analysis/clutrr_bad_flips.pdf', bbox_inches='tight')
    plt.show()

# ── Step 14: Normalised stacked histogram (outcome proportion per iter count) ──
if scs_all:
    lvsf   = [[], [], [], []]
    totals = []
    for i in range(len(scs_all)):
        l = len(scs_all[i]) - 1
        lvsf[flag_all[i]].append(l)
        totals.append(l)

    bins       = np.array([1, 2, 3, 4, 5, 6, 7])
    counts_all = np.vstack([np.histogram(lvsf[j], bins=bins)[0] for j in range(4)])
    col_sums   = counts_all.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    norm_counts = counts_all / col_sums
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width       = np.diff(bins)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    bottom    = np.zeros_like(bin_centers, dtype=float)
    for j in range(4):
        ax5.bar(bin_centers, norm_counts[j], width=width, bottom=bottom,
                color=cs[j], label=plot_labels[j], edgecolor='black')
        bottom += norm_counts[j]

    ax5.set_ylabel('Proportion')
    ax5.set_xlabel('# ARGOS Iterations Before Exit')
    ax5.set_xticks(bins[:-1])
    for spine in ['top', 'right', 'left']:
        ax5.spines[spine].set_visible(False)
    ax5.set_yticks([])

    # Total count sub-axis
    ____, totalcounts = np.unique(totals, return_counts=True)
    ax_top = fig5.add_axes([0.125, 0.85, 0.775, 0.2])
    ax_top.hist(totals, bins=bins, alpha=0.6)
    ax_top.set_ylabel('Total count')
    for sp in ['top', 'right', 'bottom', 'left']:
        ax_top.spines[sp].set_visible(False)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    patches = [mpatches.Patch(color=cs[i], label=f'{plot_labels[i]} (n={fc})')
               for i, fc in zip(flag_counts[0], flag_counts[1])]
    ax_top.legend(handles=patches, bbox_to_anchor=(0.3, 0.5))
    fig5.savefig(BASE_PATH + '/main/analysis/clutrr_lenhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 15: 2D histogram (iteration × confidence) ────────────────────────────
if scs_all:
    fx_by_flag = [[], [], [], []]
    fy_by_flag = [[], [], [], []]
    for j in range(len(scs_all)):
        s = scs_all[j]
        for i_step in range(len(s)):
            fx_by_flag[flag_all[j]].append(i_step)
            fy_by_flag[flag_all[j]].append(float(s[i_step]))

    from matplotlib.colors import LightSource
    fhist = [np.histogram2d(fx_by_flag[i], fy_by_flag[i],
                            bins=[[2,3,4,5,6,7],[0,0.2,0.4,0.6,0.8,1]])[0]
             for i in range(4)]

    fig6 = plt.figure()
    ax6  = fig6.add_subplot(projection='3d')
    ax6.view_init(elev=40, azim=320, roll=0)
    xedges = [2,3,4,5,6,7]
    yedges = [0, 0.2, 0.4, 0.6, 0.8, 1]
    xpos_m, ypos_m = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    xpos_m = xpos_m.ravel()
    ypos_m = ypos_m.ravel() * 100

    for j in range(len(xpos_m)):
        cumhist = 0
        for i in [3, 0, 2, 1]:
            dz = fhist[i].ravel()[j]
            ax6.bar3d(xpos_m[j], ypos_m[j], cumhist, 0.5, 10, dz,
                      zorder=0, color=cs[i], lightsource=LightSource(azdeg=190))
            cumhist += dz

    ax6.set_xlabel('Iteration Number')
    ax6.set_ylabel('confidence')
    ax6.set_title('Histogram of Confidences as ARGOS Iterates')
    ax6.set_yticklabels([f'{i}%' for i in [-100, -60, -20, 20, 60, 100]])
    ax6.set_ylim(0, 101)
    patches = [mpatches.Patch(color=cs[i], label=plot_labels[i]) for i in range(4)]
    ax6.legend(handles=patches, bbox_to_anchor=(1.3, 1))
    fig6.savefig(BASE_PATH + '/main/analysis/clutrr_threedhist.pdf', bbox_inches='tight')
    plt.show()

# ── Step 16: Summary bar chart ────────────────────────────────────────────────
categories = ['Correct (SAT)', 'Correct (CoT)', 'Incorrect', 'Missed']
values     = [correct_by_sat, correct_by_cot, total - acc, missed]
bar_colors = ['#2ecc71', '#27ae60', '#e74c3c', '#95a5a6']

fig7, ax7 = plt.subplots(figsize=(8, 4))
bars = ax7.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             str(val), ha='center', va='bottom', fontweight='bold')
ax7.set_title('CLUTRR Pipeline Results')
ax7.set_ylabel('Count')
fig7.tight_layout()
fig7.savefig(BASE_PATH + '/main/analysis/clutrr_results.png', dpi=150)
plt.show()

print('\n[Done] Available: preds, labels, outs, data, names, scs_all, flag_all, bs_outs_acc, bs_sc_acc')
