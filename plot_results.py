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

print("=== СТРОГИЙ АНАЛИЗ: ПАРНЫЙ T-ТЕСТ С ПОПРАВКОЙ ХОЛМА (HOLM-BONFERRONI) ===\n")

for ds_name, report_key in datasets:
    print(f"[{ds_name.upper()}]")
    print("-" * 80)
    
    for cls in classes:
        print(f"КЛАСС / МЕТРИКА: {cls.upper()}")
        print("-" * 40)
        
        # 1. Извлекаем массивы F1-score для каждого условия
        base_f1 = np.array([run[report_key][cls]['f1-score'] for run in base_data['runs']])
        tapt_f1 = np.array([run[report_key][cls]['f1-score'] for run in tapt_data['runs']])
        aug_f1 = np.array([run[report_key][cls]['f1-score'] for run in aug_data['runs']])
        tapt_aug_f1 = np.array([run[report_key][cls]['f1-score'] for run in tapt_aug_data['runs']])
        
        # 2. Определяем 6 пар для сравнения
        comparisons = [
            ("TAPT vs Base", tapt_f1, base_f1),
            ("AUG vs Base", aug_f1, base_f1),
            ("TAPT+AUG vs Base", tapt_aug_f1, base_f1),
            ("TAPT vs AUG", tapt_f1, aug_f1),
            ("TAPT+AUG vs TAPT", tapt_aug_f1, tapt_f1),
            ("TAPT+AUG vs AUG", tapt_aug_f1, aug_f1)
        ]
        
        raw_p_values = []
        comp_stats = []
        
        # 3. Собираем статистику и сырые p-values для всех 6 сравнений
        for comp_name, arr1, arr2 in comparisons:
            mean_diff = np.mean(arr1 - arr2)
            t_stat, p_value = stats.ttest_rel(arr1, arr2)
            
            raw_p_values.append(p_value)
            comp_stats.append({
                'name': comp_name,
                'arr1_mean': arr1.mean(), 'arr1_std': arr1.std(),
                'arr2_mean': arr2.mean(), 'arr2_std': arr2.std(),
                'diff': mean_diff
            })
        
        # 4. Применяем поправку Холма на множественные сравнения (m=6)
        reject_null, corrected_p_values, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
        
        # 5. Красивый вывод с учетом корректировки
        for i, stat in enumerate(comp_stats):
            is_sig = "✅ ЗНАЧИМО" if reject_null[i] else "❌ НЕ ЗНАЧИМО"
            
            print(f"Сравнение: {stat['name']}")
            print(f"  Средние: {stat['arr1_mean']:.4f} vs {stat['arr2_mean']:.4f}")
            print(f"  Дельта : {stat['diff']:+.4f}")
            print(f"  Сырое p-value        : {raw_p_values[i]:.5f}")
            print(f"  Скорр. p-value (Holm): {corrected_p_values[i]:.5f} -> {is_sig}\n")
            
    print("=" * 80 + "\n")


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
    base_data = load_metrics(base_file)
    tapt_data = load_metrics(tapt_file)
    aug_data = load_metrics(aug_file)
    tapt_aug_data = load_metrics(tapt_aug_file)
    
    epochs = np.arange(1, max_epochs + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Learning Curves', fontsize=16, fontweight='bold', y=1.05)
    
    metrics_to_plot = [
        ('train_loss', ax1, 'Training Loss', 'Loss'),
        ('val_loss', ax2, 'Validation Loss', 'Loss'),
        ('val_macro_f1', ax3, 'Validation Macro F1', 'Macro F1')
    ]
    
    colors = {'Base': '#e74c3c', 'TAPT': "#2215da", 'Aug': '#bdc01e', 'TAPT_Aug': '#2ecc71'}
    # Академические названия для легенды
    labels = {
        'Base': 'Baseline RuBERT',
        'TAPT': 'Task-Adapted RuBERT',
        'Aug': 'LLM-Augmented RuBERT',
        'TAPT_Aug': 'Task-Adapted LLM-Augmented RuBERT'
    }
    
    for metric_name, ax, title, ylabel in metrics_to_plot:
        base_matrix = get_padded_matrix(base_data, metric_name, max_epochs)
        tapt_matrix = get_padded_matrix(tapt_data, metric_name, max_epochs)
        aug_matrix = get_padded_matrix(aug_data, metric_name, max_epochs)
        tapt_aug_matrix = get_padded_matrix(tapt_aug_data, metric_name, max_epochs)
        
        base_mean, base_std = base_matrix.mean(axis=0), base_matrix.std(axis=0)
        tapt_mean, tapt_std = tapt_matrix.mean(axis=0), tapt_matrix.std(axis=0)
        aug_mean, aug_std = aug_matrix.mean(axis=0), aug_matrix.std(axis=0) 
        tapt_aug_mean, tapt_aug_std = tapt_aug_matrix.mean(axis=0), tapt_aug_matrix.std(axis=0)
        
        ax.plot(epochs, base_mean, label=labels['Base'], color=colors['Base'], linewidth=2)
        ax.fill_between(epochs, base_mean - base_std, base_mean + base_std, color=colors['Base'], alpha=0.15)
        
        ax.plot(epochs, tapt_mean, label=labels['TAPT'], color=colors['TAPT'], linewidth=2)
        ax.fill_between(epochs, tapt_mean - tapt_std, tapt_mean + tapt_std, color=colors['TAPT'], alpha=0.15)

        ax.plot(epochs, aug_mean, label=labels['Aug'], color=colors['Aug'], linewidth=2)
        ax.fill_between(epochs, aug_mean - aug_std, aug_mean + aug_std, color=colors['Aug'], alpha=0.15)

        ax.plot(epochs, tapt_aug_mean, label=labels['TAPT_Aug'], color=colors['TAPT_Aug'], linewidth=2)
        ax.fill_between(epochs, tapt_aug_mean - tapt_aug_std, tapt_aug_mean + tapt_aug_std, color=colors['TAPT_Aug'], alpha=0.15)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(epochs)
        
    # Выносим единую легенду в самый низ по центру, чтобы не перекрывать линии
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=4, fontsize=12, frameon=True, title="Model", title_fontsize=13)
        
    plt.tight_layout()
    plt.savefig('learning_curves_3_panels.png', dpi=300, bbox_inches='tight')
    plt.show()

# Запускаем отрисовку
# plot_learning_curves(
#     'experiment_results/experiment_metrics_tapt_False_aug_False.json',
#     'experiment_results/experiment_metrics_tapt_True_aug_False.json',
#     'experiment_results/experiment_metrics_tapt_False_aug_True.json',
#     'experiment_results/experiment_metrics_tapt_True_aug_True.json',
#     max_epochs=10
# )

target_metrics = ['precision', 'recall', 'f1-score']
metric_labels = ['Macro Precision', 'Macro Recall', 'Macro F1']

def calc_stats(data_array):
    """Считает среднее и 95% CI для массива"""
    mean = np.mean(data_array)
    sem = stats.sem(data_array)
    ci_margin = stats.t.ppf(0.975, len(data_array)-1) * sem if sem > 0 else 0
    return mean, ci_margin

# Считаем, что base_data, tapt_data, aug_data, tapt_aug_data уже загружены
# и datasets выглядит примерно так: [('Golden Set', 'gold_report'), ('Synthetic Set', 'synth_report')]

colors = {'Base': '#e74c3c', 'TAPT': "#2215da", 'Aug': '#bdc01e', 'TAPT_Aug': '#2ecc71'}

# Проходим по каждому датасету и создаем отдельную фигуру
for ds_name, report_key in datasets:
    
    # Создаем холст 1x2 (Абсолютные метрики + Дельты) только для текущего датасета
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_bar, ax_diff) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Model Performance: {ds_name}', fontsize=16, fontweight='bold', y=1.02)
    
    # Списки для абсолютных значений
    base_means, base_errs = [], []
    tapt_means, tapt_errs = [], []
    aug_means, aug_errs = [], []
    tapt_aug_means, tapt_aug_errs = [], []
    
    # Списки для дельт (каждая относительно Base)
    diff_tapt_means, diff_tapt_errs = [], []
    diff_aug_means, diff_aug_errs = [], []
    diff_tapt_aug_means, diff_tapt_aug_errs = [], []
    
    for metric in target_metrics:
        # Извлекаем метрики
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
            
        # 1. Абсолютные значения
        b_mean, b_err = calc_stats(b_val); base_means.append(b_mean); base_errs.append(b_err)
        t_mean, t_err = calc_stats(t_val); tapt_means.append(t_mean); tapt_errs.append(t_err)
        a_mean, a_err = calc_stats(a_val); aug_means.append(a_mean); aug_errs.append(a_err)
        a_t_mean, a_t_err = calc_stats(a_t_val); tapt_aug_means.append(a_t_mean); tapt_aug_errs.append(a_t_err)
        
        # 2. ИСПРАВЛЕННЫЕ Дельты (TAPT-Base, Aug-Base, TAPT+Aug-Base)
        d_t_mean, d_t_err = calc_stats(t_val - b_val)
        diff_tapt_means.append(d_t_mean); diff_tapt_errs.append(d_t_err)
        
        d_a_mean, d_a_err = calc_stats(a_val - b_val)
        diff_aug_means.append(d_a_mean); diff_aug_errs.append(d_a_err)

        d_t_a_mean, d_t_a_err = calc_stats(a_t_val - b_val) # ИСПРАВЛЕН БАГ ТУТ
        diff_tapt_aug_means.append(d_t_a_mean); diff_tapt_aug_errs.append(d_t_a_err)

    # --- ЛЕВЫЙ ГРАФИК: Столбчатая диаграмма ---
    x = np.arange(len(target_metrics))
    width = 0.2  # Немного сузили столбцы для 4 штук
    
    # Правильное симметричное центрирование 4 столбцов
    ax_bar.bar(x - 1.5*width, base_means, width, yerr=base_errs, label='Baseline RuBERT', color=colors['Base'], capsize=4, edgecolor='black', alpha=0.8)
    ax_bar.bar(x - 0.5*width, tapt_means, width, yerr=tapt_errs, label='Task Adapted RuBERT', color=colors['TAPT'], capsize=4, edgecolor='black', alpha=0.8)
    ax_bar.bar(x + 0.5*width, aug_means, width, yerr=aug_errs, label='LLM-Augmented RuBERT', color=colors['Aug'], capsize=4, edgecolor='black', alpha=0.8)
    ax_bar.bar(x + 1.5*width, tapt_aug_means, width, yerr=tapt_aug_errs, label='Task Adapted LLM-Augmented RuBERT', color=colors['TAPT_Aug'], capsize=4, edgecolor='black', alpha=0.8)

    ax_bar.set_title('Absolute Macro Metrics', fontsize=13)
    ax_bar.set_ylabel('Score', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=11)
    ax_bar.set_ylim(0, 1.0)

    # --- ПРАВЫЙ ГРАФИК: Дельты ---
    ax_diff.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    offset = 0.18 # Разнос точек по X
    
    for i in range(len(target_metrics)):
        # 1. TAPT Delta (Левее центра)
        is_sig_tapt = (diff_tapt_means[i] - diff_tapt_errs[i]) > 0
        color_t = colors['TAPT'] if is_sig_tapt else '#95a5a6'
        ax_diff.errorbar(x[i] - offset, diff_tapt_means[i], yerr=diff_tapt_errs[i], fmt='o', color=color_t, 
                         markersize=9, capsize=5, capthick=2, elinewidth=2)
        ax_diff.annotate(f"{diff_tapt_means[i]:+.3f}", (x[i] - offset, diff_tapt_means[i]), 
                         textcoords="offset points", xytext=(0, 12), ha='center', va='bottom', fontsize=9)

        # 2. Aug Delta (По центру)
        if not np.isnan(diff_aug_means[i]): 
            is_sig_aug = (diff_aug_means[i] - diff_aug_errs[i]) > 0
            color_a = colors['Aug'] if is_sig_aug else '#95a5a6'
            ax_diff.errorbar(x[i], diff_aug_means[i], yerr=diff_aug_errs[i], fmt='s', color=color_a, 
                             markersize=9, capsize=5, capthick=2, elinewidth=2)
            ax_diff.annotate(f"{diff_aug_means[i]:+.3f}", (x[i], diff_aug_means[i]), 
                             textcoords="offset points", xytext=(0, -15), ha='center', va='top', fontsize=9)
            
        # 3. TAPT+Aug Delta (Правее центра)
        if not np.isnan(diff_tapt_aug_means[i]): 
            is_sig_ta = (diff_tapt_aug_means[i] - diff_tapt_aug_errs[i]) > 0
            color_ta = colors['TAPT_Aug'] if is_sig_ta else '#95a5a6' # ИСПРАВЛЕН ЦВЕТ
            # Используем треугольник (fmt='^'), чтобы визуально отличать от просто Aug
            ax_diff.errorbar(x[i] + offset, diff_tapt_aug_means[i], yerr=diff_tapt_aug_errs[i], fmt='^', color=color_ta, 
                             markersize=10, capsize=5, capthick=2, elinewidth=2)
            ax_diff.annotate(f"{diff_tapt_aug_means[i]:+.3f}", (x[i] + offset, diff_tapt_aug_means[i]), 
                             textcoords="offset points", xytext=(0, 12), ha='center', va='bottom', fontsize=9)

    ax_diff.set_title('Improvement over Baseline', fontsize=13)
    ax_diff.set_ylabel('Difference vs Baseline', fontsize=12)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(metric_labels, fontsize=11)
    
    # Динамический расчет лимитов Y (теперь учитываем все 3 массива дельт)
    all_diffs = np.concatenate([diff_tapt_means, diff_aug_means, diff_tapt_aug_means])
    all_errs = np.concatenate([diff_tapt_errs, diff_aug_errs, diff_tapt_aug_errs])
    valid_mask = ~np.isnan(all_diffs)
    
    if np.any(valid_mask):
        max_abs_diff = max(np.abs(all_diffs[valid_mask] + all_errs[valid_mask]).max(), 
                           np.abs(all_diffs[valid_mask] - all_errs[valid_mask]).max())
        ax_diff.set_ylim(-max_abs_diff * 1.3, max_abs_diff * 1.3) 

    plt.tight_layout()

    handles, legend_labels = ax_bar.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), 
               title="Model", fontsize=11, title_fontsize=12, frameon=True, borderpad=1.2)
    
    # Генерируем безопасное имя файла на основе названия датасета
    safe_name = ds_name.lower().replace(' ', '_')
    plt.savefig(f'performance_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.show()