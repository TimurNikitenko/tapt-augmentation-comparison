import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import random
from metrics import (
    get_mean_pooled_embeddings, 
    calculate_internal_cosine_similarity, 
    calculate_self_bleu, 
    compute_and_plot_tsne,
    calculate_pseudo_perplexity
)

fine_tune_ds_path = 'training_dataset.json'
augmented_ds_path = 'augmented_dataset.jsonl'
tapt_ds_path = 'tapt_corpus.jsonl'
rubert_tapt_path = 'rubert-tapt/'

def load_json_corpus(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Не удалось загрузить {filepath}: {e}")
        return []

def load_classification_data(filepath, is_jsonl=False):
    """Группирует данные по классам."""
    data = {'single': [], 'listing': [], 'junk': []}
    try:
        if is_jsonl:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if item['label'] in data:
                            data[item['label']].append(item['text'])
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for item in items:
                    if item['label'] in data:
                        data[item['label']].append(item['text'])
    except Exception as e:
        print(f"Ошибка загрузки классификационных данных из {filepath}: {e}")
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    base_model_id = "DeepPavlov/rubert-base-cased-conversational"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModel.from_pretrained(base_model_id).to(device)

    # 1. Загружаем базовые корпуса
    base_texts = []
    base_texts.extend(load_json_corpus("ru_bert_training_corpus/open_subtitles_corpus.json") or [])
    base_texts.extend(load_json_corpus("ru_bert_training_corpus/pikabu_corpus.json") or [])
    base_texts.extend(load_json_corpus("ru_bert_training_corpus/taiga_corpus.json") or [])
    
    try:
        with open(tapt_ds_path, "r", encoding="utf-8") as f:
            it_texts_full = [line.strip() for line in f.readlines() if line.strip()]
    except:
        it_texts_full = []

    if base_texts and it_texts_full:
        SAMPLE_SIZE_DRIFT = 3000
        np.random.seed(42)
        
        base_sample = np.random.choice(base_texts, size=min(SAMPLE_SIZE_DRIFT, len(base_texts)), replace=False).tolist()
        it_sample = np.random.choice(it_texts_full, size=min(SAMPLE_SIZE_DRIFT, len(it_texts_full)), replace=False).tolist()

        base_embeddings = get_mean_pooled_embeddings(base_model, base_tokenizer, base_sample, device)
        it_embeddings = get_mean_pooled_embeddings(base_model, base_tokenizer, it_sample, device)

        compute_and_plot_tsne(
            base_embeddings, it_embeddings, 
            label1='Original RuBERT pretrain corpus', label2='Goal task corpus', 
            save_path='semantic_drift_base_vs_tapt.png',
            title='Semantic Drift: Original pretrain corpus vs Goal task corpus'
        )

    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    gold_data = load_classification_data(fine_tune_ds_path, is_jsonl=False)
    synth_data = load_classification_data(augmented_ds_path, is_jsonl=True)

    tapt_tokenizer = AutoTokenizer.from_pretrained(rubert_tapt_path)
    tapt_model = AutoModel.from_pretrained(rubert_tapt_path).to(device)
    
    tapt_mlm = AutoModelForMaskedLM.from_pretrained(rubert_tapt_path).to(device)
    
    embedder_tiny = SentenceTransformer('cointegrated/rubert-tiny2', device=device)

    classes = {'single': 'Class 1', 'listing': 'Class 2', 'junk': 'Class 3'}
    SAMPLE_SIZE_SYNTH = 3000  

    results = {}

    for cls in classes:
        print(f"\nАнализ класса: [{cls.upper()}] ---")
        
        gold_texts = gold_data.get(cls, [])
        synth_texts = synth_data.get(cls, [])
        
        if not gold_texts or not synth_texts:
            print(f"Пропуск {cls}: недостаточно данных (Gold: {len(gold_texts)}, Synth: {len(synth_texts)})")
            continue

        gold_sample = random.sample(gold_texts, min(SAMPLE_SIZE_SYNTH, len(gold_texts)))
        synth_sample = random.sample(synth_texts, min(SAMPLE_SIZE_SYNTH, len(synth_texts)))

        gold_self_bleu = calculate_self_bleu(gold_sample, sample_size=500)
        synth_self_bleu = calculate_self_bleu(synth_sample, sample_size=500)

        gold_cos_sim = calculate_internal_cosine_similarity(embedder_tiny, gold_sample, sample_size=2000)
        synth_cos_sim = calculate_internal_cosine_similarity(embedder_tiny, synth_sample, sample_size=2000)

        gold_pppl = calculate_pseudo_perplexity(tapt_mlm, tapt_tokenizer, gold_sample, device)
        synth_pppl = calculate_pseudo_perplexity(tapt_mlm, tapt_tokenizer, synth_sample, device)

        print(f"Self-BLEU   | Gold: {gold_self_bleu:.4f} | Synth: {synth_self_bleu:.4f}")
        print(f"Cosine Sim  | Gold: {gold_cos_sim:.4f} | Synth: {synth_cos_sim:.4f}")
        print(f"Pseudo-PPPL | Gold: {gold_pppl:.2f} | Synth: {synth_pppl:.2f}")

        print(f"Построение t-SNE для {cls}")
        gold_emb = get_mean_pooled_embeddings(tapt_model, tapt_tokenizer, gold_sample, device)
        synth_emb = get_mean_pooled_embeddings(tapt_model, tapt_tokenizer, synth_sample, device)

        compute_and_plot_tsne(
            gold_emb, synth_emb, 
            label1=f'Organic', label2=f'Synthetic', 
            save_path=f'tsne_class_{cls}.png',
            title=f'Task Pretrained Model Embedding Space: Organic vs LLM-Generated ({classes[cls]})'
        )

        results[cls] = {
            'gold_self_bleu': float(gold_self_bleu),
            'synth_self_bleu': float(synth_self_bleu),
            'gold_cos_sim': float(gold_cos_sim),
            'synth_cos_sim': float(synth_cos_sim),
            # Раскомментируй PPPL, когда будешь делать финальный, долгий прогон
            # 'gold_pppl': float(gold_pppl),
            # 'synth_pppl': float(synth_pppl)
        }

    output_filename = 'text_metrics.json'
    print(f"\nСохранение всех вычисленных метрик в {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    del tapt_model
    del tapt_tokenizer
    del tapt_mlm
    del embedder_tiny

if __name__ == "__main__":
    main()