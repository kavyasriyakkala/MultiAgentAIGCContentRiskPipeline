"""
Tool 1: Similarity Clustering Tool
Detects near-duplicate/templated reviews using embeddings and clustering.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import time

class SimilarityTool:
    """
    Embeds reviews using TF-IDF and flags near-duplicates.
    Signature of low-quality/templated review clusters (common in AI-generated reviews).
    """

    def __init__(self, model_name="tfidf"):
        """Initialize embedding model (sklearn TF-IDF, local, no API)"""
        print(f"Initializing embedding model: {model_name}")
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.embeddings_cache = {}
        self.clusters = None
        self.fitted_vectorizer = None

    def embed_reviews(self, review_texts):
        """
        Embed a list of review texts using TF-IDF.

        Args:
            review_texts: list of strings

        Returns:
            embeddings: np.ndarray of shape (n, embedding_dim)
        """
        start_time = time.time()

        # Fit vectorizer if not already done
        if self.fitted_vectorizer is None:
            self.fitted_vectorizer = self.vectorizer.fit(review_texts)

        # Transform all texts
        embeddings = self.fitted_vectorizer.transform(review_texts).toarray()
        latency = time.time() - start_time

        return embeddings, latency

    def fit_training_embeddings(self, train_embeddings: np.ndarray, train_labels: np.ndarray):
        """
        Store training embeddings split by label for KNN similarity scoring at inference.
        Called once after training data is embedded.
        """
        self.train_fake_embeddings = train_embeddings[train_labels == 1]
        self.train_genuine_embeddings = train_embeddings[train_labels == 0]

    def compute_knn_risk_score(self, embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """
        For each review, compute a risk score based on how much closer it is
        to known-fake training examples than known-genuine ones (cosine distance).

        Returns values in [0, 1]: >0.5 means closer to fakes.
        Falls back to DBSCAN-based scores if training embeddings are not available.
        """
        if not hasattr(self, 'train_fake_embeddings') or len(self.train_fake_embeddings) == 0:
            return self._dbscan_risk_scores(embeddings)

        from sklearn.metrics.pairwise import cosine_distances as _cd
        risk_scores = np.zeros(len(embeddings))

        dist_to_fake = _cd(embeddings, self.train_fake_embeddings)
        dist_to_genuine = _cd(embeddings, self.train_genuine_embeddings)

        for i in range(len(embeddings)):
            k_fake = min(k, dist_to_fake.shape[1])
            k_genuine = min(k, dist_to_genuine.shape[1])
            avg_fake = np.mean(np.partition(dist_to_fake[i], k_fake - 1)[:k_fake])
            avg_genuine = np.mean(np.partition(dist_to_genuine[i], k_genuine - 1)[:k_genuine])
            # score = P(closer to fake): low dist_to_fake → high risk
            total = avg_fake + avg_genuine
            risk_scores[i] = avg_genuine / total if total > 0 else 0.5

        return risk_scores

    def _dbscan_risk_scores(self, embeddings: np.ndarray, eps: float = 0.7, min_samples: int = 2) -> np.ndarray:
        """Fallback: DBSCAN with a permissive eps so clusters actually form."""
        distances = cosine_distances(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distances)

        non_noise = cluster_labels[cluster_labels >= 0]
        if len(non_noise) == 0:
            return np.full(len(embeddings), 0.5)

        cluster_sizes = np.bincount(non_noise)
        risk = np.zeros(len(embeddings))
        max_size = np.max(cluster_sizes)
        for i, label in enumerate(cluster_labels):
            if label >= 0:
                risk[i] = min(cluster_sizes[label] / max(2, max_size), 1.0)
            else:
                risk[i] = 0.0
        return risk

    def __call__(self, review_texts, review_ids=None):
        """
        Main entry point: embed reviews and compute KNN-based similarity risk scores.

        Returns:
            dict with:
                - 'cluster_labels': placeholder array (kept for interface compatibility)
                - 'cluster_risk_scores': np.ndarray of risk scores (0-1), >0.5 = closer to fakes
                - 'latency_seconds': float
                - 'embeddings': np.ndarray
        """
        start_time = time.time()

        embeddings, _ = self.embed_reviews(review_texts)
        risk_scores = self.compute_knn_risk_score(embeddings)

        # Cluster labels kept as placeholder for downstream compatibility
        cluster_labels = np.where(risk_scores > 0.5, 1, 0)

        total_latency = time.time() - start_time

        return {
            'cluster_labels': cluster_labels,
            'cluster_risk_scores': risk_scores,
            'latency_seconds': total_latency,
            'tool_name': 'similarity_clustering',
            'embeddings': embeddings
        }
