import json
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open('training_dataset.json', 'r', encoding='utf-8') as f:
    training_dataset = json.load(f)
print(f"Загружено: {len(training_dataset)} примеров обучающей выборки.")

with open('golden_dataset_classifier.json', 'r', encoding='utf-8') as f:
    golden_dataset = json.load(f)
print(f"Загружено: {len(golden_dataset)} примеров отложенной тестовой выборки.")

with open('tapt_corpus.jsonl', 'r', encoding='utf-8') as json_file:
    tapt_corpus = [json.loads(line) for line in json_file]
print(f"Загружено: {len(tapt_corpus)} примеров TAPT корпуса.")


def get_text_hash(text: str) -> str:
    """Нормализованный MD5-хэш текста."""
    normalized_text = " ".join(str(text).lower().split())
    return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

def remove_leaks_from_dicts(target_data: list[dict], golden_data: list[dict], dataset_name: str, fuzzy_threshold: float = 0.9):
    
    golden_urls = {item['url'] for item in golden_data if 'url' in item}
    golden_hashes = {get_text_hash(item.get('text', '')) for item in golden_data}
    golden_texts = [item.get('text', '') for item in golden_data]

    temp_data = []
    exact_leaks = 0

    for item in target_data:
        text = item.get('text', '')
        url = item.get('url', '')
        
        if url in golden_urls or get_text_hash(text) in golden_hashes:
            exact_leaks += 1
            continue 
            
        temp_data.append(item)

    if not temp_data:
        print(f"Удалено {exact_leaks} точных дубликатов. Датасет пуст!")
        return []

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
    
    target_texts = [item.get('text', '') for item in temp_data]
    all_texts = golden_texts + target_texts
    
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    golden_matrix = tfidf_matrix[:len(golden_texts)]
    target_matrix = tfidf_matrix[len(golden_texts):]

    sim_matrix = cosine_similarity(target_matrix, golden_matrix)
    max_similarities = sim_matrix.max(axis=1)

    clean_data = []
    fuzzy_leaks = 0

    for i, item in enumerate(temp_data):
        if max_similarities[i] >= fuzzy_threshold:
            fuzzy_leaks += 1
        else:
            clean_data.append(item)

    total_leaks = exact_leaks + fuzzy_leaks
    if total_leaks > 0:
        print(f"Найдено точных совпадений: {exact_leaks}")
        print(f"Найдено нечетких совпадений (>{fuzzy_threshold}): {fuzzy_leaks}")
    else:
        print("Утечек не обнаружено.")
        
    print(f"Итоговый размер после очистки: {len(clean_data)}")
    
    return clean_data


clean_training_dataset = remove_leaks_from_dicts(
    target_data=training_dataset, 
    golden_data=golden_dataset, 
    dataset_name="Обучающая выборка (Fine-tune)"
)

clean_tapt_corpus = remove_leaks_from_dicts(
    target_data=tapt_corpus, 
    golden_data=golden_dataset, 
    dataset_name="Претрейн корпус (TAPT)"
)


with open('clean_training_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(clean_training_dataset, f, ensure_ascii=False, indent=4)

with open('clean_tapt_corpus.jsonl', 'w', encoding='utf-8') as f:
    for item in clean_tapt_corpus:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

