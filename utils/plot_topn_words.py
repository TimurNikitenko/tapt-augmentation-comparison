from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json

with open('training_dataset.json', 'r', encoding='utf-8') as f:
    golden = json.load(f)
    golden_texts = [item['text'] for item in golden]

with open('augmented_dataset.jsonl', 'r', encoding='utf-8') as f:
    generated_records = [json.loads(line) for line in f]
    synthetic_texts = [record['text'] for record in generated_records]


def get_top_bigrams(corpus, n=20):
    vec = CountVectorizer(ngram_range=(2, 2), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

top_organic = get_top_bigrams(golden_texts, 50)
top_synthetic = get_top_bigrams(synthetic_texts, 50)

print(f"{'Organic (Human) Bigrams':<35} | {'Synthetic (LLM) Bigrams'}")
print("-" * 75)
for (org_word, org_count), (syn_word, syn_count) in zip(top_organic, top_synthetic):
    print(f"{org_word:<25} ({org_count:<4}) | {syn_word:<25} ({syn_count:<4})")