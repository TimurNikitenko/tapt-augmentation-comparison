import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

def load_metrics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Замени пути на свои
base_data = load_metrics('experiment_results/experiment_metrics_tapt_False_aug_False.json')
tapt_data = load_metrics('experiment_results/experiment_metrics_tapt_True_aug_False.json')
aug_data = load_metrics('experiment_results/experiment_metrics_tapt_False_aug_True.json')
tapt_aug_data = load_metrics('experiment_results/experiment_metrics_tapt_True_aug_True.json')


classes = ['single', 'listing', 'junk', 'macro avg']
datasets = [
    ('Gold Standard Evaluation Set', 'golden_classification_report'), 
]

def get_padded_matrix(data, metric_name, max_epochs=10):
    """
    Извлекает метрику для всех сидов и выравнивает длину массива.
    Если модель остановилась раньше (early stopping), последнее значение тянется до конца.
    """
    runs = data['runs']
    matrix = np.zeros((len(runs), max_epochs))
    
    for i, run in enumerate(runs):
        curve = run['learning_curve']
        for epoch_idx in range(max_epochs):
            if epoch_idx < len(curve):
                matrix[i, epoch_idx] = curve[epoch_idx][metric_name]
            else:
                matrix[i, epoch_idx] = curve[-1][metric_name]
    return matrix

def plot_learning_curves(base_file, tapt_file, aug_file, tapt_aug_file, max_epochs=10):
    import os
    os.makedirs('images', exist_ok=True)
    
    base_data = load_metrics(base_file)
    tapt_data = load_metrics(tapt_file)
    aug_data = load_metrics(aug_file)
    tapt_aug_data = load_metrics(tapt_aug_file)
    
    epochs = np.arange(1, max_epochs + 1)
    
    colors = {'Base': '#e74c3c', 'TAPT': "#2215da", 'Aug': '#bdc01e', 'TAPT_Aug': '#2ecc71'}
    labels = {
        'Base': 'Baseline RuBERT',
        'TAPT': 'Task-Adapted RuBERT',
        'Aug': 'LLM-Augmented RuBERT',
        'TAPT_Aug': 'Task-Adapted LLM-Augmented RuBERT'
    }
    
    metrics_to_plot = [
        ('train_loss', 'Training Loss', 'Loss', 'images/train_loss.pdf'),
        ('val_loss', 'Validation Loss', 'Loss', 'images/val_loss.pdf'),
        ('val_macro_f1', 'Validation Macro F1', 'Macro F1', 'images/val_f1.pdf')
    ]
    
    for metric_name, title, ylabel, filename in metrics_to_plot:
        plt.figure(figsize=(6, 4.5))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        base_matrix = get_padded_matrix(base_data, metric_name, max_epochs)
        tapt_matrix = get_padded_matrix(tapt_data, metric_name, max_epochs)
        aug_matrix = get_padded_matrix(aug_data, metric_name, max_epochs)
        tapt_aug_matrix = get_padded_matrix(tapt_aug_data, metric_name, max_epochs)
        
        base_mean, base_std = base_matrix.mean(axis=0), base_matrix.std(axis=0)
        tapt_mean, tapt_std = tapt_matrix.mean(axis=0), tapt_matrix.std(axis=0)
        aug_mean, aug_std = aug_matrix.mean(axis=0), aug_matrix.std(axis=0) 
        tapt_aug_mean, tapt_aug_std = tapt_aug_matrix.mean(axis=0), tapt_aug_matrix.std(axis=0)
        
        plt.plot(epochs, base_mean, label=labels['Base'], color=colors['Base'], linewidth=2)
        plt.fill_between(epochs, base_mean - base_std, base_mean + base_std, color=colors['Base'], alpha=0.15)
        
        plt.plot(epochs, tapt_mean, label=labels['TAPT'], color=colors['TAPT'], linewidth=2)
        plt.fill_between(epochs, tapt_mean - tapt_std, tapt_mean + tapt_std, color=colors['TAPT'], alpha=0.15)

        plt.plot(epochs, aug_mean, label=labels['Aug'], color=colors['Aug'], linewidth=2)
        plt.fill_between(epochs, aug_mean - aug_std, aug_mean + aug_std, color=colors['Aug'], alpha=0.15)

        plt.plot(epochs, tapt_aug_mean, label=labels['TAPT_Aug'], color=colors['TAPT_Aug'], linewidth=2)
        plt.fill_between(epochs, tapt_aug_mean - tapt_aug_std, tapt_aug_mean + tapt_aug_std, color=colors['TAPT_Aug'], alpha=0.15)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(epochs)
        
        if metric_name == 'val_macro_f1':
            plt.legend(fontsize=9, frameon=True, loc='best')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

# Запускаем отрисовку
plot_learning_curves(
    'experiment_results/experiment_metrics_tapt_False_aug_False.json',
    'experiment_results/experiment_metrics_tapt_True_aug_False.json',
    'experiment_results/experiment_metrics_tapt_False_aug_True.json',
    'experiment_results/experiment_metrics_tapt_True_aug_True.json',
    max_epochs=10
)

target_metrics = ['precision', 'recall', 'f1-score']
metric_labels = ['Macro Precision', 'Macro Recall', 'Macro F1']

def calc_stats(data_array):
    """Считает среднее и 95% CI для массива"""
    mean = np.mean(data_array)
    sem = stats.sem(data_array)
    ci_margin = stats.t.ppf(0.975, len(data_array)-1) * sem if sem > 0 else 0
    return mean, ci_margin

colors = {'Base': '#e74c3c', 'TAPT': "#2215da", 'Aug': '#bdc01e', 'TAPT_Aug': '#2ecc71'}
plt.style.use('seaborn-v0_8-whitegrid')

for ds_name, report_key in datasets:
    safe_name = ds_name.lower().replace(' ', '_')
    
    base_means, base_errs = [], []
    tapt_means, tapt_errs = [], []
    aug_means, aug_errs = [], []
    tapt_aug_means, tapt_aug_errs = [], []
    
    diff_tapt_means, diff_tapt_errs = [], []
    diff_aug_means, diff_aug_errs = [], []
    diff_tapt_aug_means, diff_tapt_aug_errs = [], []
    
    for metric in target_metrics:
        b_val = np.array([run[report_key]['macro avg'][metric] for run in base_data['runs']])
        t_val = np.array([run[report_key]['macro avg'][metric] for run in tapt_data['runs']])
        
        try:
            a_val = np.array([run[report_key]['macro avg'][metric] for run in aug_data['runs']])
        except KeyError:
            a_val = np.full_like(b_val, np.nan)

        try:
            a_t_val = np.array([run[report_key]['macro avg'][metric] for run in tapt_aug_data['runs']])
        except KeyError:
            a_t_val = np.full_like(b_val, np.nan)
            
        b_mean, b_err = calc_stats(b_val); base_means.append(b_mean); base_errs.append(b_err)
        t_mean, t_err = calc_stats(t_val); tapt_means.append(t_mean); tapt_errs.append(t_err)
        a_mean, a_err = calc_stats(a_val); aug_means.append(a_mean); aug_errs.append(a_err)
        a_t_mean, a_t_err = calc_stats(a_t_val); tapt_aug_means.append(a_t_mean); tapt_aug_errs.append(a_t_err)
        
        d_t_mean, d_t_err = calc_stats(t_val - b_val); diff_tapt_means.append(d_t_mean); diff_tapt_errs.append(d_t_err)
        d_a_mean, d_a_err = calc_stats(a_val - b_val); diff_aug_means.append(d_a_mean); diff_aug_errs.append(d_a_err)
        d_t_a_mean, d_t_a_err = calc_stats(a_t_val - b_val); diff_tapt_aug_means.append(d_t_a_mean); diff_tapt_aug_errs.append(d_t_a_err)

    x = np.arange(len(target_metrics))
    width = 0.18

    fig_bar, ax_bar = plt.subplots(figsize=(9, 5.5))
    
    ax_bar.bar(x - 1.5*width, base_means, width, yerr=base_errs, label='Baseline RuBERT', color=colors['Base'], capsize=4, edgecolor='black', alpha=0.85)
    ax_bar.bar(x - 0.5*width, tapt_means, width, yerr=tapt_errs, label='Task Adapted RuBERT', color=colors['TAPT'], capsize=4, edgecolor='black', alpha=0.85)
    ax_bar.bar(x + 0.5*width, aug_means, width, yerr=aug_errs, label='LLM-Augmented RuBERT', color=colors['Aug'], capsize=4, edgecolor='black', alpha=0.85)
    ax_bar.bar(x + 1.5*width, tapt_aug_means, width, yerr=tapt_aug_errs, label='Task Adapted LLM-Augmented RuBERT', color=colors['TAPT_Aug'], capsize=4, edgecolor='black', alpha=0.85)

    ax_bar.set_title(f'Absolute Macro Metrics: {ds_name}', fontsize=14, fontweight='bold', pad=15)
    ax_bar.set_ylabel('Score Value', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=11)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(loc='lower left', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'performance_absolute_{safe_name}.pdf', bbox_inches='tight')
    plt.close()

    fig_diff, ax_diff = plt.subplots(figsize=(9, 5.5))
    ax_diff.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    offset = 0.2 
    for i in range(len(target_metrics)):
        is_sig_tapt = (diff_tapt_means[i] - diff_tapt_errs[i]) > 0
        color_t = colors['TAPT'] if is_sig_tapt else '#95a5a6'
        ax_diff.errorbar(x[i] - offset, diff_tapt_means[i], yerr=diff_tapt_errs[i], fmt='o', color=color_t, 
                         markersize=9, capsize=5, capthick=2, elinewidth=2, label='Task Adapted' if i==0 else "")
        ax_diff.annotate(f"{diff_tapt_means[i]:+.3f}", (x[i] - offset, diff_tapt_means[i]), 
                         textcoords="offset points", xytext=(0, 10), ha='center', va='bottom', fontsize=9, fontweight='bold' if is_sig_tapt else 'normal')

        if not np.isnan(diff_aug_means[i]): 
            is_sig_aug = (diff_aug_means[i] - diff_aug_errs[i]) > 0
            color_a = colors['Aug'] if is_sig_aug else '#95a5a6'
            ax_diff.errorbar(x[i], diff_aug_means[i], yerr=diff_aug_errs[i], fmt='s', color=color_a, 
                             markersize=9, capsize=5, capthick=2, elinewidth=2, label='LLM-Augmented' if i==0 else "")
            ax_diff.annotate(f"{diff_aug_means[i]:+.3f}", (x[i], diff_aug_means[i]), 
                             textcoords="offset points", xytext=(0, -15), ha='center', va='top', fontsize=9, fontweight='bold' if is_sig_aug else 'normal')
            
        if not np.isnan(diff_tapt_aug_means[i]): 
            is_sig_ta = (diff_tapt_aug_means[i] - diff_tapt_aug_errs[i]) > 0
            color_ta = colors['TAPT_Aug'] if is_sig_ta else '#95a5a6'
            ax_diff.errorbar(x[i] + offset, diff_tapt_aug_means[i], yerr=diff_tapt_aug_errs[i], fmt='^', color=color_ta, 
                             markersize=10, capsize=5, capthick=2, elinewidth=2, label='Task Adapted + LLM-Augmented' if i==0 else "")
            ax_diff.annotate(f"{diff_tapt_aug_means[i]:+.3f}", (x[i] + offset, diff_tapt_aug_means[i]), 
                             textcoords="offset points", xytext=(0, 10), ha='center', va='bottom', fontsize=9, fontweight='bold' if is_sig_ta else 'normal')

    ax_diff.set_title(f'Improvement over Baseline: {ds_name}', fontsize=14, fontweight='bold', pad=15)
    ax_diff.set_ylabel('Difference ($\Delta$) Value', fontsize=12)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(metric_labels, fontsize=11)
    
    all_diffs = np.concatenate([diff_tapt_means, diff_aug_means, diff_tapt_aug_means])
    all_errs = np.concatenate([diff_tapt_errs, diff_aug_errs, diff_tapt_aug_errs])
    valid_mask = ~np.isnan(all_diffs)
    
    if np.any(valid_mask):
        max_abs_diff = max(np.abs(all_diffs[valid_mask] + all_errs[valid_mask]).max(), 
                           np.abs(all_diffs[valid_mask] - all_errs[valid_mask]).max())
        ax_diff.set_ylim(-max_abs_diff * 1.4, max_abs_diff * 1.4) 

    plt.tight_layout()
    plt.savefig(f'performance_delta_{safe_name}.pdf', bbox_inches='tight')
    plt.close()