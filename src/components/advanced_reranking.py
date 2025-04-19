"""
Advanced Multi-Stage Reranking

This module implements an advanced multi-stage reranking pipeline that combines
multiple reranking signals for improved precision in RAG systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import time

class AdvancedReranker:
    """Advanced multi-stage reranking implementation"""
    
    def __init__(self, 
                cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                use_gpu: bool = torch.cuda.is_available()):
        """
        Initialize reranker with desired models
        
        Args:
            cross_encoder_model: Name of cross-encoder model to use
            use_gpu: Whether to use GPU acceleration
        """
        self.cross_encoder_model = cross_encoder_model
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    def score_with_cross_encoder(self,
                               query: str,
                               documents: List[Dict[str, str]],
                               batch_size: int = 16) -> List[Tuple[Dict[str, str], float]]:
        """
        Score query-document pairs using a cross-encoder model
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' field
            batch_size: Batch size for processing
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        # Prepare input pairs: (query, doc_text)
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Score with the cross-encoder
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        # Pair documents with scores
        doc_score_pairs = [(doc, float(score)) for doc, score in zip(documents, scores)]
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
        
        return doc_score_pairs
    
    def score_with_keyword_matching(self,
                                  query: str,
                                  documents: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], float]]:
        """
        Score documents based on keyword matching with the query
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' field
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        import re
        from collections import Counter
        
        # Tokenize query into keywords (remove common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'about', 'is', 'are'}
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        query_keywords = [token for token in query_tokens if token not in stop_words]
        
        # If we have a multi-word query, also look for exact phrases
        query_phrases = []
        if len(query_tokens) > 1:
            query_phrases = [' '.join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)]
            query_phrases += [' '.join(query_tokens[i:i+3]) for i in range(len(query_tokens)-2)]
        
        # Calculate scores for each document
        doc_scores = []
        
        for doc in documents:
            text = doc["text"].lower()
            
            # Count keyword occurrences
            keyword_count = sum(text.count(keyword) for keyword in query_keywords)
            
            # Count phrase occurrences (with higher weight)
            phrase_count = sum(text.count(phrase) * 2 for phrase in query_phrases) 
            
            # Calculate keyword density (normalize by document length)
            doc_length = len(text.split())
            keyword_density = (keyword_count + phrase_count) / max(1, doc_length)
            
            # Exact query match bonus
            exact_match_bonus = 5 if query.lower() in text else 0
            
            # Calculate final score
            score = keyword_density * 10 + exact_match_bonus
            
            doc_scores.append((doc, score))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda pair: pair[1], reverse=True)
        
        return doc_scores
    
    def score_with_semantic_similarity(self,
                                     query_embedding: np.ndarray,
                                     documents: List[Dict[str, str]],
                                     document_embeddings: np.ndarray) -> List[Tuple[Dict[str, str], float]]:
        """
        Score documents based on semantic similarity with query embedding
        
        Args:
            query_embedding: Query embedding vector
            documents: List of document dictionaries
            document_embeddings: Matrix of document embeddings
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        # Calculate cosine similarity
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Pair documents with similarity scores
        doc_score_pairs = [(doc, float(score)) for doc, score in zip(documents, similarities)]
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
        
        return doc_score_pairs
    
    def reciprocal_rank_fusion(self,
                              ranked_lists: List[List[Tuple[Dict[str, str], float]]],
                              k: int = 60) -> List[Tuple[Dict[str, str], float]]:
        """
        Apply Reciprocal Rank Fusion to multiple ranked lists
        
        Args:
            ranked_lists: List of ranked lists, each with (doc, score) tuples
            k: Constant to stabilize rankings
            
        Returns:
            Fused list of (doc, score) tuples sorted by fused score
        """
        # Calculate RRF scores
        fused_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, (doc, _) in enumerate(ranked_list, start=1):
                doc_id = doc.get("chunk_id", str(id(doc)))  # Use chunk_id or object ID as identifier
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                
                # Add to fused scores
                if doc_id in fused_scores:
                    fused_scores[doc_id]["score"] += rrf_score
                else:
                    fused_scores[doc_id] = {"doc": doc, "score": rrf_score}
        
        # Sort by fused score
        fused_docs = [(item["doc"], item["score"]) for item in fused_scores.values()]
        fused_docs.sort(key=lambda pair: pair[1], reverse=True)
        
        return fused_docs
    
    def diversify_results(self,
                         ranked_docs: List[Tuple[Dict[str, str], float]],
                         document_embeddings: Optional[np.ndarray] = None,
                         alpha: float = 0.5) -> List[Tuple[Dict[str, str], float]]:
        """
        Diversify results using Maximal Marginal Relevance (MMR)
        
        Args:
            ranked_docs: Initial ranked list of (doc, score) tuples
            document_embeddings: Optional matrix of document embeddings
            alpha: Trade-off between relevance and diversity
            
        Returns:
            Diversified list of (doc, score) tuples
        """
        if len(ranked_docs) <= 1:
            return ranked_docs
        
        # Extract documents and scores
        docs = [doc for doc, _ in ranked_docs]
        scores = np.array([score for _, score in ranked_docs])
        
        # If no embeddings provided, use document text for similarity
        if document_embeddings is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            try:
                doc_texts = [doc.get("text", "") for doc in docs]
                document_embeddings = vectorizer.fit_transform(doc_texts).toarray()
            except:
                # If vectorization fails, return original ranking
                return ranked_docs
        
        # Normalize scores to [0, 1]
        if scores.size > 0:
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score
        
        # MMR implementation
        selected_indices = []
        unselected_indices = list(range(len(docs)))
        
        while unselected_indices and len(selected_indices) < len(docs):
            mmr_scores = []
            
            for i in unselected_indices:
                # Relevance component is the original score
                relevance = scores[i]
                
                # Diversity component is the maximum similarity to any selected document
                if not selected_indices:
                    diversity = 0
                else:
                    # Calculate cosine similarity between document i and all selected documents
                    similarities = []
                    for j in selected_indices:
                        emb_i = document_embeddings[i]
                        emb_j = document_embeddings[j]
                        
                        # Calculate cosine similarity
                        similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                        similarities.append(similarity)
                    
                    diversity = max(similarities) if similarities else 0
                
                # MMR score = alpha * relevance - (1 - alpha) * diversity
                mmr_score = alpha * relevance - (1 - alpha) * diversity
                mmr_scores.append((i, mmr_score))
            
            # Select document with highest MMR score
            next_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(next_idx)
            unselected_indices.remove(next_idx)
        
        # Reorder documents according to MMR ranking
        diversified_docs = [(docs[i], float(scores[i])) for i in selected_indices]
        
        return diversified_docs
    
    def multi_stage_reranking(self,
                            query: str,
                            query_embedding: np.ndarray,
                            documents: List[Dict[str, str]],
                            document_embeddings: np.ndarray,
                            top_k: int = 10,
                            stages: List[str] = ["semantic", "cross_encoder", "diversity"],
                            alpha: float = 0.7) -> List[Tuple[Dict[str, str], float]]:
        """
        Perform a comprehensive multi-stage reranking process
        
        Args:
            query: Query string
            query_embedding: Query embedding vector
            documents: List of document dictionaries
            document_embeddings: Matrix of document embeddings
            top_k: Number of final results to return
            stages: Reranking stages to apply
            alpha: Diversity parameter
            
        Returns:
            List of (document, score) tuples after multi-stage reranking
        """
        # Limit to 100 documents for initial reranking to improve efficiency
        max_initial_docs = 100
        if len(documents) > max_initial_docs:
            # First filter by vector similarity
            semantic_results = self.score_with_semantic_similarity(
                query_embedding, documents, document_embeddings
            )
            documents = [doc for doc, _ in semantic_results[:max_initial_docs]]
            document_embeddings = document_embeddings[:max_initial_docs]
        
        stage_results = []
        
        # Apply requested stages
        if "semantic" in stages:
            # Score with semantic similarity
            semantic_results = self.score_with_semantic_similarity(
                query_embedding, documents, document_embeddings
            )
            stage_results.append(semantic_results)
        
        if "keyword" in stages:
            # Score with keyword matching
            keyword_results = self.score_with_keyword_matching(query, documents)
            stage_results.append(keyword_results)
        
        if "cross_encoder" in stages:
            # Score with cross-encoder
            cross_encoder_results = self.score_with_cross_encoder(query, documents)
            stage_results.append(cross_encoder_results)
        
        # Fuse results from all scoring methods
        if len(stage_results) > 1:
            fused_results = self.reciprocal_rank_fusion(stage_results)
        else:
            fused_results = stage_results[0] if stage_results else []
        
        # Apply diversity reranking if requested
        if "diversity" in stages and fused_results:
            # Get document indices for the fused results
            fused_docs = [doc for doc, _ in fused_results]
            doc_indices = [documents.index(doc) for doc in fused_docs if doc in documents]
            
            # Get corresponding embeddings
            if len(doc_indices) > 0:
                fused_embeddings = document_embeddings[doc_indices]
                diversified_results = self.diversify_results(fused_results, fused_embeddings, alpha)
            else:
                diversified_results = fused_results
            
            final_results = diversified_results
        else:
            final_results = fused_results
        
        # Return top_k results
        return final_results[:top_k]

# Example usage:
# reranker = AdvancedReranker()
# reranked_docs = reranker.multi_stage_reranking(
#     query, query_embedding, documents, document_embeddings,
#     stages=["semantic", "cross_encoder", "diversity"]
# )