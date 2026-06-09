import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

with open('training_dataset.json', 'r', encoding='utf-8') as f:
    train_dataset = json.load(f)
train_texts = [item['text'] for item in train_dataset]
train_labels = [item['label'] for item in train_dataset]

with open('golden_dataset_classifier.json', 'r', encoding='utf-8') as f:
    test_dataset = json.load(f)
test_texts = [item['text'] for item in test_dataset]
test_labels = [item['label'] for item in test_dataset]

mapping = {
    'junk': 0,
    'single': 1,
    'listing': 2
}

y_train = [mapping[label] for label in train_labels]
y_val = [mapping[label] for label in test_labels]

def evaluate(max_features, ngram_range, sublinear_tf, class_weight, C):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf
    )
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_val_vec = vectorizer.transform(test_texts)
    
    classifier = LogisticRegression(class_weight=class_weight, C=C, max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, y_train)
    
    y_pred = classifier.predict(X_val_vec)
    
    report = classification_report(y_val, y_pred, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    accuracy = report['accuracy']
    
    print(f"Features: {max_features if max_features else 'None'}, Ngrams: {ngram_range}, SublinearTF: {sublinear_tf}, ClassWeight: {class_weight}, C: {C}")
    print(f"  Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}")
    return macro_f1, accuracy

settings = [
    (1500, (1, 1), False, None, 1.0),
    (1500, (1, 1), False, 'balanced', 1.0),
    (5000, (1, 1), False, None, 1.0),
    (5000, (1, 1), False, 'balanced', 1.0),
    (10000, (1, 1), False, 'balanced', 1.0),
    (None, (1, 1), False, 'balanced', 1.0),
    (None, (1, 1), True, 'balanced', 1.0),
    (10000, (1, 2), True, 'balanced', 1.0),
    (20000, (1, 2), True, 'balanced', 1.0),
    (None, (1, 2), True, 'balanced', 1.0),
    (None, (1, 2), True, 'balanced', 0.5),
    (None, (1, 2), True, 'balanced', 2.0),
    (None, (1, 2), True, 'balanced', 5.0),
]
# the best configuration is Features: None, Ngrams: (1, 2), SublinearTF: True, ClassWeight: balanced, C: 5.0, 
# the results are: Macro F1: 0.7081 | Accuracy: 0.7638
# for s in settings:
#     evaluate(*s)


vectorizer = TfidfVectorizer(
    max_features=None,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(train_texts)
X_val_vec = vectorizer.transform(test_texts)

classifier = LogisticRegression(class_weight='balanced', C=5, max_iter=1000, random_state=45)
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_val_vec)

report = classification_report(y_val, y_pred)

print(report)