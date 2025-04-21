"""
Reranking Module

This module implements different reranking approaches for RAG systems:
- Cross-encoder reranking
- LLM-based reranking
- Reciprocal Rank Fusion
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np

class RerankerModule:
    """Implementation of different reranking approaches"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = self.config.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Add logic here to load the actual reranking model based on self.model_name
        # self.reranker_model = CrossEncoder(self.model_name) # Example using sentence-transformers CrossEncoder
        print(f"Reranker initialized with model: {self.model_name}")

    
    @staticmethod
    def score_with_cross_encoder(
        query: str,
        documents: List[Dict[str, str]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Score query-document pairs using a cross-encoder model
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' field
            model_name: Name of the cross-encoder model to use
            
        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Prepare input pairs: (query, doc_text)
            pairs = [(query, doc["text"]) for doc in documents]
            
            # Tokenize
            features = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            # Score
            with torch.no_grad():
                scores = model(**features).logits.flatten().tolist()
                
            # Pair documents with scores
            doc_score_pairs = [(doc, score) for doc, score in zip(documents, scores)]
            
            # Sort by score (descending)
            doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
            
            return doc_score_pairs
        except ImportError:
            print("Warning: transformers package not installed. Install with 'pip install transformers'")
            # Fallback to simple word overlap scoring
            return RerankerModule.score_with_word_overlap(query, documents)
    
    def rerank_documents(self, query: str, documents: List[Any]) -> List[Any]:
        """Reranks documents based on query using the initialized model."""
        if not documents:
            return []
        if not hasattr(self, 'reranker_model'):
            print("ERROR: Reranker model not loaded in __init__. Cannot rerank.")
            return documents # Return original list

        print(f"Reranking {len(documents)} docs with model {self.model_name}...")
        # --- Implementation depends on your chosen reranker library ---
        # Example using sentence-transformers CrossEncoder (if self.reranker_model is loaded)
        try:
            # Assuming 'documents' are Langchain Docs or dicts with 'text'
            texts_to_rank = [doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '') for doc in documents]
            sentence_pairs = [[query, text] for text in texts_to_rank]
            scores = self.reranker_model.predict(sentence_pairs)

            # Combine documents with scores and sort
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Return only the sorted documents
            reranked_docs = [doc for doc, score in scored_docs]
            return reranked_docs
        except Exception as e:
            print(f"ERROR during reranking prediction: {e}")
            return documents # Return original list on error

    @staticmethod
    def score_with_word_overlap(
        query: str,
        documents: List[Dict[str, str]]
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Simple fallback scoring based on word overlap
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' field
            
        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        query_words = set(query.lower().split())
        doc_score_pairs = []
        
        for doc in documents:
            doc_words = set(doc["text"].lower().split())
            if not doc_words:
                score = 0.0
            else:
                overlap = len(query_words.intersection(doc_words))
                score = overlap / (len(query_words) + len(doc_words) - overlap)  # Jaccard similarity
            
            doc_score_pairs.append((doc, score))
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
        
        return doc_score_pairs
    
    @staticmethod
    def score_with_llm_api(
        query: str,
        documents: List[Dict[str, str]],
        llm_function: Callable
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Score query-document pairs using an LLM
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' field
            llm_function: Function to call an LLM
            
        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        # For each document, ask LLM to score its relevance to the query
        doc_score_pairs = []
        
        for doc in documents:
            # Create prompt
            prompt = f"""
            On a scale from 1 to 10, rate how relevant the following document is to the query.
            Only respond with a single number from 1-10, nothing else.
            
            Query: {query}
            Document: {doc["text"]}
            
            Relevance score (1-10):
            """
            
            # Get response from LLM
            response = llm_function(prompt)
            
            try:
                # Extract numeric score (will look for first digit in the response)
                import re
                match = re.search(r'(\d+)', response)
                score = float(match.group(1)) if match else 5.0  # Default to 5 if parsing fails
                
                # Normalize to 0-1 range
                score = min(10, max(1, score)) / 10.0
            except:
                # Default score if parsing fails
                score = 0.5
                
            doc_score_pairs.append((doc, score))
            
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
        
        return doc_score_pairs
    
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[Dict[str, str], float]]],
        k: int = 60
    ) -> List[Tuple[Dict[str, str], float]]:
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
    
    @staticmethod
    def contextual_reranking(
        query: str,
        documents: List[Dict[str, str]],
        conversation_history: List[str] = None
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Rerank documents considering conversation context
        
        Args:
            query: Current query
            documents: List of document dictionaries
            conversation_history: Optional list of previous queries
            
        Returns:
            Reranked list of (document, score) tuples
        """
        # If no conversation history, fall back to word overlap
        if not conversation_history:
            return RerankerModule.score_with_word_overlap(query, documents)
            
        # Combine current query with conversation history
        context = " ".join(conversation_history[-3:])  # Use last 3 turns
        contextual_query = f"{context} {query}"
        
        # Score with the contextual query
        return RerankerModule.score_with_word_overlap(contextual_query, documents)
    
    @staticmethod
    def diversity_reranking(
        query: str,
        documents: List[Dict[str, str]],
        alpha: float = 0.5,
        initial_ranker: Callable = None
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Rerank for diversity using Maximal Marginal Relevance (MMR)
        
        Args:
            query: Query string
            documents: List of document dictionaries
            alpha: Trade-off between relevance and diversity (1 = only relevance)
            initial_ranker: Initial ranking function (defaults to word overlap)
            
        Returns:
            Reranked list of (document, score) tuples
        """
        if len(documents) <= 1:
            return [(doc, 1.0) for doc in documents]
            
        # Initial ranking
        if initial_ranker:
            initial_ranking = initial_ranker(query, documents)
        else:
            initial_ranking = RerankerModule.score_with_word_overlap(query, documents)
            
        # Extract document texts and compute token overlap matrix for diversity
        doc_texts = [doc["text"] for doc, _ in initial_ranking]
        
        # Function to compute similarity between documents (simple token overlap)
        def doc_similarity(doc1, doc2):
            tokens1 = set(doc1.lower().split())
            tokens2 = set(doc2.lower().split())
            if not tokens1 or not tokens2:
                return 0.0
            overlap = len(tokens1.intersection(tokens2))
            return overlap / (len(tokens1) + len(tokens2) - overlap)  # Jaccard
            
        # MMR implementation
        selected = []
        remaining = list(initial_ranking)
        
        while remaining and len(selected) < len(documents):
            # MMR score for each remaining document
            max_score = -1
            max_idx = -1
            
            for i, (doc, rel_score) in enumerate(remaining):
                # Relevance component
                relevance = rel_score
                
                # Diversity component (max similarity to already selected docs)
                if not selected:
                    diversity = 0
                else:
                    # Find maximum similarity to any selected document
                    diversity = max([doc_similarity(doc["text"], sel_doc["text"]) 
                                    for sel_doc, _ in selected])
                
                # MMR score = alpha * relevance - (1 - alpha) * diversity
                mmr_score = alpha * relevance - (1 - alpha) * diversity
                
                if mmr_score > max_score:
                    max_score = mmr_score
                    max_idx = i
                    
            # Add the document with max MMR score to selected
            if max_idx >= 0:
                selected.append(remaining[max_idx])
                remaining.pop(max_idx)
            else:
                break
                
        return selected