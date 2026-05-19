import logging
import numpy as np
import scipy.sparse as sp
import torch
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class RuBertTfidfEnsemble:
    """
    Ensemble classifier combining TF-IDF lexical features with 
    dense semantic embeddings from a pre-trained RuBERT model.
    """
    def __init__(self, rubert_model_instance):
        # We store the pre-trained RuBERT instance to act as our feature extractor
        self.rubert = rubert_model_instance
        self.device = self.rubert.device
        
        # Initialize the classical ML components
        self.tfidf = TfidfVectorizer(
            max_features=15000, 
            ngram_range=(1, 2), # Capture single words and short phrases (e.g., "it meetup")
            min_df=3,
            max_df=0.8
        )
        # Using saga solver as it handles sparse data efficiently
        self.logreg = LogisticRegression(
            solver='saga', 
            max_iter=1000, 
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

    def _extract_rubert_embeddings(self, texts, batch_size=16):
        """Extracts the [CLS] token embeddings from RuBERT in batches."""
        self.rubert.model.eval()
        all_embeddings = []
        
        # Process in batches to prevent GPU OOM
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting RuBERT Features"):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [self.rubert.processor.preprocess(t) for t in batch_texts]
            
            inputs = self.rubert.tokenizer(
                batch_texts, 
                truncation=True, 
                max_length=512, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # We request hidden states to get the raw layer outputs
                outputs = self.rubert.model(**inputs, output_hidden_states=True)
                # hidden_states[-1] is the last layer. [:, 0, :] grabs the [CLS] token for the batch.
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def train(self, train_texts, train_labels, batch_size=16, ensemble=True):
        """Trains the TF-IDF vectorizer and the final Logistic Regression model."""
        logger.info("Fitting TF-IDF Vectorizer...")
        X_tfidf = self.tfidf.fit_transform(train_texts)
        
        if ensemble:
            logger.info("Extracting RuBERT Embeddings...")
            X_rubert = self._extract_rubert_embeddings(train_texts, batch_size)
        
        logger.info("Fusing Feature Spaces...")
        # Convert dense RuBERT numpy array to sparse matrix, then horizontally stack
        X_combined = X_tfidf

        if ensemble:
            X_rubert_sparse = sp.csr_matrix(X_rubert)
            X_combined = sp.hstack([X_tfidf, X_rubert_sparse])
                    
        
        logger.info("Training Final Logistic Regression...")
        self.logreg.fit(X_combined, train_labels)
        
        # Quick evaluation on train set to verify learning
        preds = self.logreg.predict(X_combined)
        logger.info("\n" + classification_report(train_labels, preds))

    def predict(self, text, ensemble=True):
        """Inference pipeline for a single text."""
        if not text or len(text.strip()) == 0:
            return "junk"
            
        # 1. Get Lexical Features
        X_tfidf = self.tfidf.transform([text])
        
        if ensemble:
            # 2. Get Semantic Features
            X_rubert = self._extract_rubert_embeddings([text], batch_size=1)
            X_rubert_sparse = sp.csr_matrix(X_rubert)

        X_combined = X_tfidf
        if ensemble:
            # 3. Combine
             X_combined = sp.hstack([X_tfidf, X_rubert_sparse])
        
        # 4. Predict
        pred_class = self.logreg.predict(X_combined)[0]
        return pred_class

    def save(self, path):
        """Saves the classical ML components (RuBERT should be saved separately)."""
        import os
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tfidf, f"{path}/tfidf.joblib")
        joblib.dump(self.logreg, f"{path}/logreg.joblib")
        logger.info(f"Ensemble components saved to {path}")