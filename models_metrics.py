import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from metrics import (
    calculate_self_bleu, 
    calculate_internal_cosine_similarity, 
    calculate_transrate, 
    get_mean_pooled_embeddings, 
    )
from tqdm import tqdm

rubert_tapt_path = 'rubert-tapt/'

model_pricing = {
    "cerebras/qwen-3-235b-a22b-instruct-2507": {"in": 0.6, "out": 1.2},
    "deepseek/deepseek-v3.2": {"in": 0.252, "out": 0.378},  
    "nvidia/nemotron-3-super-120b-a12b:free": {"in": 0.0, "out": 0.0},
    "openai/gpt-oss-120b": {"in": 0.039, "out": 0.18}, 
    "meta-llama/llama-3.3-70b-instruct": {"in": 0.10, "out": 0.32},
    "google/gemma-4-31b-it": {"in": 0.13, "out": 0.38},
    "minimax/minimax-m2.7": {"in": 0.299, "out": 1.20},
    "google/gemini-3-flash-preview": {"in": 0.5, "out": 3},
}

def main():
    jsonl_path = 'augmented_dataset.jsonl' 
    csv_path = 'models_analysis_raw.csv'

    all_gold_texts = []
    all_gold_labels = []
    
    with open('training_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            all_gold_texts.append(item['text'])
            all_gold_labels.append(item['label'])
    
    with open('golden_dataset_classifier.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            all_gold_texts.append(item['text'])
            all_gold_labels.append(item['label'])
    
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            if 'tokens' in data:
                data['prompt_tokens'] = data['tokens'].get('prompt_tokens', 0)
                data['completion_tokens'] = data['tokens'].get('completion_tokens', 0)
                data['total_tokens'] = data['tokens'].get('total_tokens', 0)
                del data['tokens']
                
            records.append(data)
            
    df = pd.DataFrame(records)
    
    df['tps'] = df['completion_tokens'] / df.apply(lambda x: max(x['latency_sec'], 0.001), axis=1)
    
    def calculate_cost(row):
        model_name = row['model']
        prices = model_pricing.get(model_name, {"in": 0.0, "out": 0.0})
        cost_in = (row['prompt_tokens'] / 1_000_000) * prices["in"]
        cost_out = (row['completion_tokens'] / 1_000_000) * prices["out"]
        return cost_in + cost_out

    df['cost_usd'] = df.apply(calculate_cost, axis=1)
    
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Сырые данные с метриками сохранены в {csv_path}")
    

    summary_df = df.groupby('model').agg(
        records_count=('label', 'count'),
        avg_temperature=('temperature', 'mean'),
        avg_latency_sec=('latency_sec', 'mean'),
        avg_text_length=('text_length_chars', 'mean'),
        avg_tps=('tps', 'mean'),
        total_cost_usd=('cost_usd', 'sum')
    ).reset_index()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer('cointegrated/rubert-tiny2', device=device)
    tapt_tokenizer = AutoTokenizer.from_pretrained(rubert_tapt_path)
    tapt_model = AutoModel.from_pretrained(rubert_tapt_path).to(device)
    
    print("Предвычисление золотых центроидов...")
    gold_embeddings = embedder.encode(all_gold_texts, show_progress_bar=True)
    all_gold_labels = np.array(all_gold_labels)
    gold_centroids = {}
    
    for cls in np.unique(all_gold_labels):
        cls_mask = (all_gold_labels == cls)
        cls_embeds = gold_embeddings[cls_mask]
        gold_centroids[cls] = np.mean(cls_embeds, axis=0).reshape(1, -1)
    
    # Списки для метрик
    self_bleu_scores = []
    internal_cosine_scores = []  
    relative_cosine_scores = []  
    transrates = []
    
    for model_name in tqdm(summary_df['model'].tolist(), desc="Оценка моделей"):
        model_texts = df[df['model'] == model_name]['text'].tolist()
        model_labels = df[df['model'] == model_name]['label'].tolist()
        
        if len(model_texts) < 2:
            self_bleu_scores.append(None)
            internal_cosine_scores.append(None)
            relative_cosine_scores.append(None)
            transrates.append(None)
            continue
            
        sb_score = calculate_self_bleu(model_texts)
        self_bleu_scores.append(sb_score)
        
        embeddings_tapt = get_mean_pooled_embeddings(tapt_model, tapt_tokenizer, model_texts, device)
        transrate = calculate_transrate(embeddings_tapt, model_labels)
        transrates.append(transrate)
        
        int_cos = calculate_internal_cosine_similarity(embedder, model_texts)
        internal_cosine_scores.append(int_cos)
        
        synth_embeds = embedder.encode(model_texts, show_progress_bar=False)
        synth_labels = np.array(model_labels)
        rel_cos_scores_cls = []
        
        for cls in np.unique(synth_labels):
            if cls in gold_centroids:
                cls_synth_embeds = synth_embeds[synth_labels == cls]
                if len(cls_synth_embeds) > 0:
                    sims = cosine_similarity(cls_synth_embeds, gold_centroids[cls])
                    rel_cos_scores_cls.append(np.mean(sims))
                    
        rel_cos = np.mean(rel_cos_scores_cls) if rel_cos_scores_cls else 0.0
        relative_cosine_scores.append(rel_cos)

    # Записываем всё в DataFrame
    summary_df['macro_self_bleu'] = self_bleu_scores
    summary_df['diversity_score'] = 1.0 - summary_df['macro_self_bleu'] 
    summary_df['macro_internal_cosine_sim'] = internal_cosine_scores
    summary_df['macro_relative_cosine_sim'] = relative_cosine_scores
    summary_df['macro_transrate'] = transrates
    
    print("\n" + "="*50)
    print("ИТОГОВЫЙ БЕНЧМАРК МОДЕЛЕЙ (Pareto Frontier Data)")
    print("="*50)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    summary_df = summary_df.sort_values(by='diversity_score', ascending=False)
    
    display_cols = ['model', 'records_count', 'avg_tps', 'total_cost_usd', 'diversity_score', 'macro_internal_cosine_sim']
    print(summary_df[display_cols].to_string(index=False))
    
    summary_df.to_csv('models_benchmark_summary.csv', index=False, encoding='utf-8')
    print("\nАгрегированный отчет сохранен в 'models_benchmark_summary.csv'")

if __name__ == "__main__":
    main()