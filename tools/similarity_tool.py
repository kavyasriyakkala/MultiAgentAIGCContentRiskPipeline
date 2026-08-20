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

    def detect_clusters(self, embeddings, eps=0.3, min_samples=2):
        """
        Run DBSCAN clustering on embeddings.
        Near-duplicates will have high cosine similarity (low distance).

        Args:
            embeddings: np.ndarray of shape (n, d)
            eps: max distance for DBSCAN
            min_samples: min points to form a cluster

        Returns:
            cluster_labels: np.ndarray of cluster IDs (-1 = noise)
        """
        distances = cosine_distances(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distances)

        return cluster_labels

    def __call__(self, review_texts, review_ids=None, eps=0.3, min_samples=2):
        """
        Main entry point: embed and cluster a batch of reviews.

        Args:
            review_texts: list of review strings
            review_ids: optional list of IDs to match with results
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            results: dict with keys:
                - 'cluster_labels': np.ndarray of cluster IDs per review
                - 'cluster_risk_scores': np.ndarray of risk scores (0-1)
                  Higher = more likely to be in a templated cluster
                - 'latency_seconds': float
        """
        start_time = time.time()

        # Embed
        embeddings, embed_latency = self.embed_reviews(review_texts)

        # Cluster
        cluster_labels = self.detect_clusters(embeddings, eps=eps, min_samples=min_samples)

        # Compute risk scores based on cluster size
        # Rationale: very large near-duplicate clusters suggest template/bot activity
        cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0], minlength=len(np.unique(cluster_labels)))
        cluster_density = np.zeros(len(review_texts))

        for i, label in enumerate(cluster_labels):
            if label >= 0:
                cluster_size = cluster_sizes[label]
                # Normalize: cluster_size / max_cluster_size, capped at 1.0
                cluster_density[i] = min(cluster_size / max(2, np.max(cluster_sizes)), 1.0)
            else:
                cluster_density[i] = 0.0  # Noise point

        total_latency = time.time() - start_time

        return {
            'cluster_labels': cluster_labels,
            'cluster_risk_scores': cluster_density,
            'latency_seconds': total_latency,
            'tool_name': 'similarity_clustering',
            'embeddings': embeddings
        }
