import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def get_mean_pooled_embeddings(model, tokenizer, texts, device, batch_size=16):
    """
    Прогоняет тексты через модель и возвращает усредненные (Mean Pooling) эмбеддинги предложений.
    """
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Извлечение эмбеддингов"):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs[0] 
            
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

def calculate_internal_cosine_similarity(embedder_model, texts, sample_size=1000):

    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
            
    embeddings = embedder_model.encode(texts, show_progress_bar=True)
    
    sim_matrix = cosine_similarity(embeddings)
    
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    unique_similarities = sim_matrix[upper_triangle_indices]
    
    mean_sim = np.mean(unique_similarities)
    return float(mean_sim)


def compute_and_plot_tsne(emb1, emb2, label1='Dataset 1', label2='Dataset 2', save_path='tsne.png', title='t-SNE'):
    SAMPLE_SIZE_1 = emb1.shape[0]
    SAMPLE_SIZE_2 = emb2.shape[0]
    X = np.vstack([emb1, emb2])
    y = [label1] * SAMPLE_SIZE_1 + [label2] * SAMPLE_SIZE_2

    n_comp = min(50, X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X_pca)

    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'Dataset': y
    })

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df, x='x', y='y', hue='Dataset',
        palette=['#bdc3c7', '#e74c3c'],
        alpha=0.7, s=50, edgecolor=None
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_self_bleu(texts, sample_size=300):
    
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
        
    tokenized_texts = [text.lower().split() for text in texts]
    
    bleu_scores = []
    smoothie = SmoothingFunction().method1 

    for i, hypothesis in enumerate(tokenized_texts):
        references = tokenized_texts[:i] + tokenized_texts[i+1:]
        
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(score)
        
    mean_self_bleu = np.mean(bleu_scores)
    return float(mean_self_bleu)


def calculate_pseudo_perplexity(model, tokenizer, texts, device):
    """
    Pseudo-Perplexity computation algorith from Salazar et al. (2020).
    Маскирует строго один токен за раз, сохраняя полный контекст остальных слов.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in tqdm(texts, desc="Exact PPPL (Salazar et al.)"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"][0] # Берем одномерный тензор
        
        special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        valid_indices = [i for i, token_id in enumerate(input_ids.tolist()) if token_id not in special_tokens]
        
        if not valid_indices:
            continue
            
        K = len(valid_indices) 
        
        batch_input_ids = input_ids.unsqueeze(0).repeat(K, 1)
        batch_labels = torch.full_like(batch_input_ids, -100)
        
        for row_idx, token_idx in enumerate(valid_indices):
            batch_input_ids[row_idx, token_idx] = tokenizer.mask_token_id
            batch_labels[row_idx, token_idx] = input_ids[token_idx]
            
        batch_inputs = {
            "input_ids": batch_input_ids.to(device),
            "attention_mask": inputs["attention_mask"].repeat(K, 1).to(device),
            "labels": batch_labels.to(device)
        }
        
        with torch.no_grad():
            outputs = model(**batch_inputs)
            loss = outputs.loss
            
            total_loss += loss.item() * K
            total_tokens += K

    exact_pppl = math.exp(total_loss / total_tokens)
    return exact_pppl


def calculate_transrate(embeddings: np.ndarray, labels: list, epsilon: float = 1e-4) -> float:
    """
    Вычисляет метрику TransRate (Transferability Rate) на основе взаимной информации.
    
    :param embeddings: numpy array размерности (N_samples, Embedding_dim)
    :param labels: список или numpy array меток классов (N_samples,)
    :param epsilon: коэффициент регуляризации (для избежания вырожденных матриц)
    :return: значение TransRate (чем выше, тем лучше разделимость классов)
    """
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    n_samples, embed_dim = embeddings.shape
    
    embeddings_centered = embeddings - np.mean(embeddings, axis=0)
    
    I = np.eye(embed_dim) * epsilon
    
    sigma_all = np.cov(embeddings_centered, rowvar=False) 
    
    _, log_det_all = np.linalg.slogdet(sigma_all + I)
    H_Z = 0.5 * log_det_all
    
    H_Z_given_Y = 0.0
    
    for cls in unique_classes:
        cls_mask = (labels == cls)
        cls_embeddings = embeddings_centered[cls_mask]
        n_cls_samples = len(cls_embeddings)
        
        if n_cls_samples <= 1:
            continue
            
        sigma_cls = np.cov(cls_embeddings, rowvar=False)
        
        _, log_det_cls = np.linalg.slogdet(sigma_cls + I)
        H_cls = 0.5 * log_det_cls
        
        weight = n_cls_samples / n_samples
        H_Z_given_Y += weight * H_cls
        
    transrate_value = H_Z - H_Z_given_Y
    
    return float(transrate_value)

def calculate_relative_cosine_similarity(embedder, synth_texts, synth_labels, gold_texts, gold_labels):
    """
    Вычисляет макро-усредненное относительное косинусное сходство 
    между синтетическими текстами и центроидами золотого датасета.
    
    :param embedder: Модель для извлечения эмбеддингов (например, SentenceTransformer)
    :param synth_texts: Список сгенерированных текстов модели
    :param synth_labels: Список лейблов для сгенерированных текстов
    :param gold_texts: Список всех текстов из объединенного золотого датасета (Train + Val)
    :param gold_labels: Список лейблов для золотого датасета
    :return: Float значение (Macro Relative Cosine Similarity)
    """
    synth_labels = np.array(synth_labels)
    gold_labels = np.array(gold_labels)
    
    unique_classes = np.unique(synth_labels)
    class_scores = []
    
    for cls in unique_classes:
        gold_cls_texts = [text for text, label in zip(gold_texts, gold_labels) if label == cls]
        synth_cls_texts = [text for text, label in zip(synth_texts, synth_labels) if label == cls]
        
        if not gold_cls_texts or not synth_cls_texts:
            continue
            
        gold_embeddings = embedder.encode(gold_cls_texts, show_progress_bar=False)
        synth_embeddings = embedder.encode(synth_cls_texts, show_progress_bar=False)
        
        gold_centroid = np.mean(gold_embeddings, axis=0).reshape(1, -1)
        
        similarities = cosine_similarity(synth_embeddings, gold_centroid)
        
        class_scores.append(np.mean(similarities))
        
    if not class_scores:
        return 0.0
        
    return float(np.mean(class_scores))
