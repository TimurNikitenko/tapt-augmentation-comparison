import pandas as pd
import pymorphy3
import nltk
from nltk.tokenize import RegexpTokenizer
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from text_preprocessor import TextProcessor 
from keywords import EVENT_KEYWORDS 

morph = pymorphy3.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
processor = TextProcessor()

LEMMA_KEYWORDS = set()
for kw in EVENT_KEYWORDS:
    clean_kw = kw.lower().strip()
    lemma = morph.parse(clean_kw)[0].normal_form
    LEMMA_KEYWORDS.add(lemma)

def get_lemmas(text):
    """Токенизация и лемматизация текста"""
    tokens = tokenizer.tokenize(text.lower())
    return [morph.parse(token)[0].normal_form for token in tokens]

def calculate_relevance(lemmas):
    """
    Строгий скоринг: считаем количество уникальных совпадений 
    лемм текста с нашим словарем.
    """
    lemmas_set = set(lemmas)
    matches = lemmas_set.intersection(LEMMA_KEYWORDS)
    return len(matches)

def prepare_tapt_corpus(input_csv, output_txt):
    print("Чтение сырых данных")
    df = pd.read_csv(input_csv)
    
    tqdm.pandas(desc="Очистка текста")
    df['clean_text'] = df['text'].astype(str).progress_apply(processor.preprocess)
    
    df = df[df['clean_text'] != 'empty']
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df = df[(df['word_count'] >= 20)] 

    tqdm.pandas(desc="Лемматизация и скоринг")
    df['lemmas'] = df['clean_text'].progress_apply(get_lemmas)
    df['score'] = df['lemmas'].apply(calculate_relevance)
    
    df_filtered = df[df['score'] >= 2].copy()
    print(f"\nНайдено релевантных текстов: {len(df_filtered)}")

    print("Запуск нечеткой дедупликации (MinHash LSH)...")
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    
    unique_records = [] 
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        lemmas = row['lemmas']
        if not lemmas:
            continue
            
        m = MinHash(num_perm=128)
        for d in lemmas:
            m.update(d.encode('utf8'))
            
        result = lsh.query(m)
        
        if not result:
            lsh.insert(str(idx), m)
            unique_records.append({
                'channel': row['channel'],
                'url': row['url'],
                'text': row['clean_text']
            })

    print(f"После удаления дубликатов осталось: {len(unique_records)} уникальных событий")

    output_jsonl = output_txt.replace('.txt', '.jsonl')
    df_final = pd.DataFrame(unique_records)
    
    df_final.to_json(output_jsonl, orient='records', lines=True, force_ascii=False)
            
    print(f"\nTAPT корпус с метаданными сохранен в {output_jsonl}")

if __name__ == "__main__":
    prepare_tapt_corpus('extended_tg_dataset.csv', 'tapt_corpus_clean.jsonl')



