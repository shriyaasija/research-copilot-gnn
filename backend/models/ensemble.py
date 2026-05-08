"""
Ensemble scoring model for paper retrieval.

Combines text similarity (from sentence embeddings), GNN activation scores
(from R-GCN embeddings), and graph-structural features (degree centrality)
into a single weighted ranking score.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Weights for the ensemble scoring formula."""
    text_similarity: float = 0.60
    gnn_activation: float = 0.25
    degree_centrality: float = 0.15
    
    def __post_init__(self):
        total = self.text_similarity + self.gnn_activation + self.degree_centrality
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}"
            )


class EnsembleScorer:
    """
    Ensemble scorer that combines multiple signals for paper ranking.
    
    Score = w_text × text_sim + w_gnn × gnn_score + w_cent × centrality
    
    All component scores are normalized to [0, 1] before combination.
    
    Args:
        weights: ScoringWeights instance (default: 0.6/0.25/0.15)
    """
    
    def __init__(self, weights: ScoringWeights = None):
        self.weights = weights or ScoringWeights()
    
    def compute_text_similarity(
        self, query_embedding: np.ndarray, paper_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all papers.
        
        Args:
            query_embedding: Query vector [dim]
            paper_embeddings: Paper embedding matrix [num_papers, dim]
        
        Returns:
            Similarity scores [num_papers], range [-1, 1]
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        paper_norms = paper_embeddings / (
            np.linalg.norm(paper_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = paper_norms @ query_norm
        return similarities
    
    def compute_gnn_scores(
        self, query_embedding: np.ndarray, gnn_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute GNN-based activation scores.
        
        Uses cosine similarity in the GNN embedding space as the activation
        score. If GNN embeddings are not available, returns zeros.
        
        Args:
            query_embedding: Query vector projected to GNN space [gnn_dim]
            gnn_embeddings: GNN embedding matrix [num_papers, gnn_dim]
        
        Returns:
            GNN activation scores [num_papers], range [0, 1]
        """
        if gnn_embeddings is None:
            return np.zeros(len(query_embedding) if isinstance(query_embedding, np.ndarray) else 0)
        
        # Cosine similarity in GNN space
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        gnn_norms = gnn_embeddings / (
            np.linalg.norm(gnn_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        scores = gnn_norms @ query_norm
        # Shift from [-1, 1] to [0, 1]
        scores = (scores + 1.0) / 2.0
        return scores
    
    def normalize_centrality(self, centrality_scores: np.ndarray) -> np.ndarray:
        """Normalize degree centrality to [0, 1]."""
        if centrality_scores is None or len(centrality_scores) == 0:
            return np.zeros(0)
        
        max_val = centrality_scores.max()
        min_val = centrality_scores.min()
        
        if max_val == min_val:
            return np.ones_like(centrality_scores) * 0.5
        
        return (centrality_scores - min_val) / (max_val - min_val)
    
    def score(
        self,
        text_similarities: np.ndarray,
        gnn_scores: np.ndarray,
        centrality_scores: np.ndarray,
        use_gnn: bool = True,
    ) -> np.ndarray:
        """
        Compute ensemble scores for all papers.
        
        Args:
            text_similarities: Text cosine similarities [num_papers]
            gnn_scores: GNN activation scores [num_papers]  
            centrality_scores: Degree centrality scores [num_papers]
            use_gnn: Whether to include GNN scores
        
        Returns:
            Combined scores [num_papers], range [0, 1]
        """
        # Normalize text similarities to [0, 1]
        text_norm = (text_similarities + 1.0) / 2.0
        
        # Normalize centrality
        cent_norm = self.normalize_centrality(centrality_scores)
        
        if use_gnn and gnn_scores is not None and len(gnn_scores) > 0:
            scores = (
                self.weights.text_similarity * text_norm
                + self.weights.gnn_activation * gnn_scores
                + self.weights.degree_centrality * cent_norm
            )
        else:
            # Text-only mode: redistribute GNN weight to text
            adjusted_text_weight = (
                self.weights.text_similarity + self.weights.gnn_activation
            )
            scores = (
                adjusted_text_weight * text_norm
                + self.weights.degree_centrality * cent_norm
            )
        
        return scores
    
    def rank(
        self,
        text_similarities: np.ndarray,
        gnn_scores: np.ndarray,
        centrality_scores: np.ndarray,
        k: int = 10,
        use_gnn: bool = True,
    ) -> list[dict]:
        """
        Rank papers and return top-K with score breakdown.
        
        Returns:
            List of dicts with keys: index, score, text_similarity,
            gnn_activation, degree_centrality
        """
        scores = self.score(text_similarities, gnn_scores, centrality_scores, use_gnn)
        
        # Get top-K indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "text_similarity": float(text_similarities[idx]),
                "gnn_activation": float(gnn_scores[idx]) if gnn_scores is not None and len(gnn_scores) > idx else 0.0,
                "degree_centrality": float(centrality_scores[idx]) if centrality_scores is not None and len(centrality_scores) > idx else 0.0,
            })
        
        return results
