import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import emoji

class TextProcessor:
    """Предобработка текста для RuBERT"""

    def __init__(self):
        self.html_pattern = re.compile(r"<[^>]+>")

        self.video_url_pattern = re.compile(
            r"https?://(?:www\.)?(?:youtube\.com|youtu\.be|vk\.com/video|rutube\.ru|vimeo\.com|music\.yandex\.ru|podcasts\.apple\.com)[^\s]+"
        )
        self.url_pattern = re.compile(r"https?://[^\s]+")

        self.mail_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.phone_pattern = re.compile(r"[\+]?[0-9\s\-\(\)]{10,}")
        self.multiple_spaces = re.compile(r"\s+")

    def preprocess(self, text: str) -> str:
        """Предобработка текста для классификации (Transformer-friendly)"""
        if not isinstance(text, str) or len(text) == 0:
            return "empty"

        

        text = text.lower()
        text = emoji.demojize(text, language="ru")
        
        text = self.html_pattern.sub(" ", text)
        text = self.video_url_pattern.sub(" [MEDIA_URL] ", text)
        text = self.url_pattern.sub(" [URL] ", text)
        text = self.mail_pattern.sub(" [EMAIL] ", text)
        text = self.phone_pattern.sub(" [PHONE] ", text)

        text = self.multiple_spaces.sub(" ", text)

        text = text.strip()

        if len(text) <= 10:
            return "empty"

        words = text.split()

        if len(words) <= 400:
            return text

        return " ".join(words[:400])
    
processor = TextProcessor()

# 1. Prepare your Golden Data
# Assuming 'golden_texts' is a simple list of your 3000 original organic texts
with open('training_dataset.json', 'r', encoding='utf-8') as f:
    golden = json.load(f)
    golden_texts = [item['text'] for item in golden]

with open('augmented_dataset.jsonl', 'r', encoding='utf-8') as f:
    generated_records = [json.loads(line) for line in f]

model_texts = {}
for record in generated_records:
    model_name = record['model']
    if model_name not in model_texts:
        model_texts[model_name] = []
    model_texts[model_name].append(record['text'])

print(f"Found {len(model_texts)} unique LLMs to evaluate.\n")

pad_results = []

for model_name, synthetic_texts in model_texts.items():
    num_synthetic = len(synthetic_texts)
    
    np.random.seed(42)
    sampled_golden = np.random.choice(golden_texts, size=num_synthetic, replace=False).tolist()
    
    processed_golden = [processor.preprocess(text) for text in sampled_golden]
    processed_synthetic = [processor.preprocess(text) for text in synthetic_texts]

    # Now build X using the processed texts
    X = processed_golden + processed_synthetic
    y = [0] * len(processed_golden) + [1] * len(processed_synthetic)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train a fast, robust linear classifier (TF-IDF + Support Vector Machine)
    # We use LinearSVC because it is the gold standard for text boundary separation in PAD literature
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svm', LinearSVC(random_state=42, dual=False))
    ])
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1.0 - accuracy
    
    # If error is somehow > 0.5, the classifier is doing worse than random guessing. 
    # Mathematically, PAD bounds this to 0.
    error_rate = min(error_rate, 0.5)
    
    # The Official PAD Formula
    pad_score = 2 * (1 - 2 * error_rate)
    
    pad_results.append({
        "LLM": model_name,
        "Texts Evaluated": num_synthetic,
        "Classifier Error": round(error_rate, 4),
        "Proxy A-Distance (PAD)": round(pad_score, 4)
    })

    # Assuming 'clf' is your trained Pipeline from the PAD script

# 4. Display the Leaderboard
leaderboard_df = pd.DataFrame(pad_results)
# Sort from lowest PAD (Most Authentic) to highest PAD (Most Detectable)
leaderboard_df = leaderboard_df.sort_values(by="Proxy A-Distance (PAD)", ascending=True).reset_index(drop=True)

print("🏆 LLM Generator Authenticity Leaderboard (Lower PAD = Better Mimicry) 🏆")
print("-" * 80)
print(leaderboard_df.to_string(index=False))


svm_model = clf.named_steps['svm']
vectorizer = clf.named_steps['tfidf']

# Get the feature names and their corresponding SVM weights
feature_names = vectorizer.get_feature_names_out()
weights = svm_model.coef_[0]

# Zip them together, sort, and print the top 15 LLM "Tells"
top_synthetic_tells = sorted(zip(weights, feature_names), reverse=True)[:15]
print("Top 15 phrases that gave away the LLM:")
for weight, word in top_synthetic_tells:
    print(f"{word}: {weight:.4f}")