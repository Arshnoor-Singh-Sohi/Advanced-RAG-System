"""
Evaluation Module

This module provides metrics and evaluation functions for RAG systems:
- Precision and recall for retrieval
- Mean Reciprocal Rank (MRR)
- ROUGE scores for text generation
- Faithfulness scoring
"""

import numpy as np
from typing import List, Dict, Any

class RAGEvaluator:
    """Evaluator for RAG system components"""
    
    @staticmethod
    def precision_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
        """Calculate precision@k"""
        if not retrieved_docs or k <= 0:
            return 0.0
            
        # Consider only top-k retrieved documents
        retrieved_k = retrieved_docs[:k]
        
        # Count relevant documents among retrieved
        relevant_count = len(set(relevant_docs) & set(retrieved_k))
        
        return relevant_count / min(k, len(retrieved_k))
    
    @staticmethod
    def recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int = 5) -> float:
        """Calculate recall@k"""
        if not relevant_docs or not retrieved_docs or k <= 0:
            return 0.0
            
        # Consider only top-k retrieved documents
        retrieved_k = retrieved_docs[:k]
        
        # Count relevant documents among retrieved
        relevant_count = len(set(relevant_docs) & set(retrieved_k))
        
        return relevant_count / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        if not relevant_docs or not retrieved_docs:
            return 0.0
            
        # Find the rank of the first relevant document
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
                
        return 0.0
    
    @staticmethod
    def rouge_scores(generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for generated text against reference
        
        Args:
            generated_text: The generated text to evaluate
            reference_text: The reference text to compare against
            
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            import evaluate
            
            rouge = evaluate.load('rouge')
            scores = rouge.compute(predictions=[generated_text], references=[reference_text])
            
            return scores
        except ImportError:
            # Fallback to a simple word overlap metric if evaluate is not available
            print("Warning: 'evaluate' package not found. Using simple overlap metric instead.")
            
            generated_words = set(generated_text.lower().split())
            reference_words = set(reference_text.lower().split())
            
            if not reference_words:
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                
            overlap = len(generated_words.intersection(reference_words))
            precision = overlap / len(generated_words) if generated_words else 0
            recall = overlap / len(reference_words) if reference_words else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {"rouge1": f1, "rouge2": f1 * 0.8, "rougeL": f1 * 0.9}  # Approximate
    
    @staticmethod
    def faithfulness_score(answer: str, contexts: List[str]) -> float:
        """
        Basic implementation of faithfulness scoring
        Checks how many n-grams from the answer are present in the context
        
        Args:
            answer: The generated answer
            contexts: List of context passages used for generation
            
        Returns:
            Faithfulness score between 0 and 1
        """
        import nltk
        from nltk.util import ngrams
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Tokenize answer and contexts
        answer_tokens = nltk.word_tokenize(answer.lower())
        
        # Generate n-grams from answer (using bigrams and trigrams)
        answer_bigrams = list(ngrams(answer_tokens, 2))
        answer_trigrams = list(ngrams(answer_tokens, 3))
        
        # Calculate how many answer n-grams are in contexts
        combined_context = " ".join(contexts).lower()
        context_tokens = nltk.word_tokenize(combined_context)
        context_bigrams = list(ngrams(context_tokens, 2))
        context_trigrams = list(ngrams(context_tokens, 3))
        
        # Calculate coverage
        bigram_matches = sum(1 for bg in answer_bigrams if bg in context_bigrams)
        trigram_matches = sum(1 for tg in answer_trigrams if tg in context_trigrams)
        
        # Calculate faithfulness as percentage of covered n-grams
        if len(answer_bigrams) == 0 and len(answer_trigrams) == 0:
            return 1.0  # Empty answer case
            
        bigram_coverage = bigram_matches / len(answer_bigrams) if len(answer_bigrams) > 0 else 0
        trigram_coverage = trigram_matches / len(answer_trigrams) if len(answer_trigrams) > 0 else 0
        
        # Return average coverage
        return (bigram_coverage + trigram_coverage) / 2 if answer_trigrams else bigram_coverage
    
    @staticmethod
    def relevance_precision(query: str, retrieved_texts: List[str], model_name: str = None) -> float:
        """
        Calculate relevance precision using cross-encoder models if available
        
        Args:
            query: The query text
            retrieved_texts: List of retrieved text passages
            model_name: Name of cross-encoder model to use
            
        Returns:
            Relevance precision score between 0 and 1
        """
        try:
            from sentence_transformers import CrossEncoder
            
            if not model_name:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                
            model = CrossEncoder(model_name)
            
            # Score each query-document pair
            pairs = [[query, doc] for doc in retrieved_texts]
            scores = model.predict(pairs)
            
            # Calculate average score
            return float(np.mean(scores))
        except ImportError:
            # Fallback to word overlap if cross-encoder is not available
            query_words = set(query.lower().split())
            scores = []
            
            for doc in retrieved_texts:
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = overlap / len(query_words) if query_words else 0
                scores.append(score)
                
            return float(np.mean(scores)) if scores else 0.0
    
    @staticmethod
    def answer_correctness(generated_answer: str, expected_answer: str, contexts: List[str] = None) -> float:
        """
        Estimate answer correctness by comparing to expected answer
        
        Args:
            generated_answer: The generated answer
            expected_answer: The expected/reference answer
            contexts: Optional list of context passages
            
        Returns:
            Correctness score between 0 and 1
        """
        # Use ROUGE scores as a proxy for correctness
        rouge_scores = RAGEvaluator.rouge_scores(generated_answer, expected_answer)
        rouge1 = rouge_scores.get("rouge1", 0)
        
        # If contexts are provided, also check faithfulness
        if contexts:
            faithfulness = RAGEvaluator.faithfulness_score(generated_answer, contexts)
            # Weight rouge1 and faithfulness equally
            return (rouge1 + faithfulness) / 2
        
        return rouge1