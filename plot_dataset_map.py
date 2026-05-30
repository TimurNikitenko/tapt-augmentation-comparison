import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open('experiment_results/dataset_cartography.json', 'r', encoding='utf-8') as f:
    cartography_data = json.load(f)

MAX_EPOCHS = 5  

records = []
for t_id, metrics in cartography_data.items():
    if str(t_id).startswith("golden_"):
        origin = "Organic"
    elif str(t_id).startswith("aug_"):
        origin = "Synthetic"

    raw_history = metrics["history"]
    truncated_history = raw_history[:MAX_EPOCHS]
    
    new_confidence = np.mean(truncated_history)
    new_variability = np.std(truncated_history)
        
    records.append({
        "ID": t_id,
        "Confidence": new_confidence,
        "Variability": new_variability,
        "Origin": origin
    })

df = pd.DataFrame(records)

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

scatter = sns.scatterplot(
    data=df,
    x="Variability",
    y="Confidence",
    hue="Origin",
    palette={"Organic": "#1f77b4", "Synthetic": "#ff7f0e"},
    alpha=0.6,
    s=45,
    edgecolor="w",
    linewidth=0.5
)

plt.text(df["Variability"].min(), 0.85, ' easy-to-learn', 
         fontsize=10, color='black', alpha=0.7, weight='bold')
plt.text(df["Variability"].max() * 0.75, 0.5, 'ambiguous', 
         fontsize=10, color='black', alpha=0.7, weight='bold')
plt.text(df["Variability"].min(), 0.15, ' hard-to-learn', 
         fontsize=10, color='black', alpha=0.7, weight='bold')

plt.title("Task-Adapted LLM-Augmented RuBERT Data Map", fontsize=16, weight='bold', pad=15)
plt.xlabel("Variability", fontsize=13, weight='bold')
plt.ylabel("Confidence", fontsize=13, weight='bold')
plt.legend(title="Origin", title_fontsize='12', fontsize='11', loc='lower right')

plt.tight_layout()
plt.savefig(
    "dataset_cartography_map.pdf", 
    bbox_inches='tight',
    transparent=False
)

print("Map successfully saved as dataset_cartography_map.pdf!")