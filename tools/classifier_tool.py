"""
Tool 2: Classifier Tool
Local ML classifier on embeddings + handcrafted features.
No external API calls.
"""
import numpy as np
import time
import re
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Try to import XGBoost; fall back to LogisticRegression if unavailable
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, ValueError):
    HAS_XGBOOST = False

class ClassifierTool:
    """
    Trains a classifier on embeddings + handcrafted features.
    Uses XGBoost if available, falls back to Logistic Regression.
    """

    def __init__(self, model_type='auto', random_state=42):
        if model_type == 'auto':
            self.model_type = 'xgboost' if HAS_XGBOOST else 'logistic_regression'
        else:
            self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def extract_features(self, review_texts, embeddings):
        """
        Extract handcrafted features from reviews.

        Features:
            - Embedding vector (TF-IDF)
            - Review length (normalized)
            - Punctuation density (exclamation, question marks)
            - Word count
            - Unique word ratio
            - Uppercase ratio
            - Number density (has numbers?)
            - URL presence
            - Rating sentiment consistency (if available)
        """
        features = []
        self.feature_names = []

        for i, text in enumerate(review_texts):
            feature_vec = []

            # 1. Review length (normalized, log scale)
            length = len(text)
            feature_vec.append(np.log1p(length))

            # 2. Punctuation density
            exclamation_count = text.count('!')
            question_count = text.count('?')
            period_count = text.count('.')
            total_punct = exclamation_count + question_count + period_count
            punct_density = total_punct / max(1, length)
            feature_vec.append(punct_density)

            # 3. Word count
            words = text.split()
            word_count = len(words)
            feature_vec.append(np.log1p(word_count))

            # 4. Unique word ratio
            unique_ratio = len(set(words)) / max(1, word_count)
            feature_vec.append(unique_ratio)

            # 5. Uppercase ratio
            uppercase_chars = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_chars / max(1, length)
            feature_vec.append(uppercase_ratio)

            # 6. Number density
            numbers = re.findall(r'\d+', text)
            number_density = len(numbers) / max(1, word_count)
            feature_vec.append(number_density)

            # 7. URL presence
            has_url = 1.0 if re.search(r'http[s]?://', text) else 0.0
            feature_vec.append(has_url)

            # 8. Avg word length
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            feature_vec.append(avg_word_length)

            # 9. Repeated words (sign of templating)
            word_freq = {}
            for w in words:
                word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1
            max_word_freq = max(word_freq.values()) if word_freq else 0
            repetition_ratio = max_word_freq / max(1, word_count)
            feature_vec.append(repetition_ratio)

            features.append(feature_vec)

        if not self.feature_names:
            self.feature_names = [
                'log_length', 'punct_density', 'log_word_count', 'unique_word_ratio',
                'uppercase_ratio', 'number_density', 'has_url', 'avg_word_length',
                'repetition_ratio'
            ]

        features = np.array(features)
        return features

    def combine_features(self, embeddings, handcrafted_features):
        """Concatenate embeddings with handcrafted features"""
        combined = np.hstack([embeddings, handcrafted_features])
        return combined

    def train(self, review_texts, embeddings, labels):
        """
        Train classifier on embeddings + features.

        Args:
            review_texts: list of review strings
            embeddings: np.ndarray of embeddings
            labels: np.ndarray of binary labels (0=genuine, 1=fake)
        """
        print("Extracting handcrafted features...")
        handcrafted = self.extract_features(review_texts, embeddings)

        print("Combining features...")
        X = self.combine_features(embeddings, handcrafted)

        print(f"Feature matrix shape: {X.shape}")
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'xgboost' and HAS_XGBOOST:
            print("Training XGBoost classifier...")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            print("Training Logistic Regression classifier...")
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )

        self.model.fit(X_scaled, labels)
        self.is_trained = True
        print("Training complete.")

    def predict(self, review_texts, embeddings):
        """
        Predict on new reviews.

        Returns:
            dict with:
                - 'predictions': binary predictions (0/1)
                - 'probabilities': probability of fake (0-1)
                - 'confidence': max(P(fake), P(genuine))
                - 'latency_seconds': float
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained first")

        start_time = time.time()

        handcrafted = self.extract_features(review_texts, embeddings)
        X = self.combine_features(embeddings, handcrafted)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Confidence = max probability
        confidence = np.max(self.model.predict_proba(X_scaled), axis=1)

        latency = time.time() - start_time

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'latency_seconds': latency,
            'tool_name': 'classifier',
            'feature_matrix': X_scaled
        }

    def save(self, path):
        """Save trained classifier"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, f)
        print(f"Classifier saved to {path}")

    def load(self, path):
        """Load trained classifier"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.is_trained = True
        print(f"Classifier loaded from {path}")
