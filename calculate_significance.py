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
            print(f"  Средние: {stat['arr1_mean']:.4f}+-{stat['arr1_std']:.4f} vs {stat['arr2_mean']:.4f}+-{stat['arr2_std']:.4f}")
            print(f"  Дельта : {stat['diff']:+.4f}")
            print(f"  Сырое p-value        : {raw_p_values[i]:.5f}")
            print(f"  Скорр. p-value (Holm): {corrected_p_values[i]:.5f} -> {is_sig}\n")
            
    print("=" * 80 + "\n")