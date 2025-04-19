"""
Enhanced HyDE (Hypothetical Document Embeddings) Implementation

This module implements an advanced version of HyDE for improved RAG retrieval quality.
HyDE works by generating a hypothetical document that would answer the query,
then using the embedding of this document for retrieval instead of the query embedding.

Reference: Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval 
without Relevance Labels. arXiv preprint arXiv:2212.10496.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from sentence_transformers import SentenceTransformer
import re

class EnhancedHyDE:
    """
    Enhanced implementation of Hypothetical Document Embeddings (HyDE)
    with query classification and multi-document generation
    """
    
    def __init__(self, embedding_model: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2"):
        """
        Initialize HyDE processor
        
        Args:
            embedding_model: Sentence transformer model or name
        """
        # Initialize embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model
            
        # Query type detection patterns
        self.query_patterns = {
            "definition": r"what is|define|meaning of|definition of",
            "factual": r"who|when|where|how many|which|what country|what year",
            "procedural": r"how to|how do|steps|process|procedure",
            "comparative": r"difference between|compare|versus|vs|similarities|better",
            "causal": r"why|cause|effect|impact|result in",
            "exploratory": r"tell me about|explain|describe"
        }

    def classify_query(self, query: str) -> str:
        """
        Classify the query type to guide document generation
        
        Args:
            query: User query
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        # Check each pattern
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query_lower):
                return query_type
        
        # Default to exploratory if no pattern matches
        return "exploratory"
    
    def generate_hypothetical_document(self, query: str, llm_function: Callable) -> str:
        """
        Generate a hypothetical document that would answer the query
        
        Args:
            query: User query
            llm_function: Function to call an LLM
            
        Returns:
            Generated hypothetical document
        """
        # Classify the query
        query_type = self.classify_query(query)
        
        # Create a type-specific prompt
        if query_type == "definition":
            hyde_prompt = f"""
            Write a clear, concise definition that would be a perfect answer to the question: '{query}'
            The definition should include key characteristics and examples where appropriate.
            Keep your answer focused and authoritative, like an encyclopedia entry.
            """
        elif query_type == "factual":
            hyde_prompt = f"""
            Write a factual, precise response to the question: '{query}'
            Include specific details, dates, numbers, or names that would be found in an authoritative reference.
            The response should be objective and well-supported.
            """
        elif query_type == "procedural":
            hyde_prompt = f"""
            Write a step-by-step guide that addresses: '{query}'
            Include clear instructions, important considerations, and potential challenges.
            Structure the content as an expert how-to guide.
            """
        elif query_type == "comparative":
            hyde_prompt = f"""
            Write a balanced comparison addressing: '{query}'
            Include key similarities and differences, with specific criteria for comparison.
            Present the information in a structured, objective manner.
            """
        elif query_type == "causal":
            hyde_prompt = f"""
            Write an explanatory response that addresses the causes and effects related to: '{query}'
            Include key factors, mechanisms, and supporting evidence.
            The explanation should be logical and connect causes to their effects.
            """
        else:  # exploratory or default
            hyde_prompt = f"""
            Write a comprehensive yet concise overview that addresses: '{query}'
            Include key facts, concepts, and relevant context.
            The content should be informative and well-structured, like an educational resource.
            """
        
        # Generate the hypothetical document
        return llm_function(hyde_prompt)
    
    def process_query(self, 
                     query: str, 
                     llm_function: Callable,
                     multi_document: bool = False,
                     num_variants: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Process a query using HyDE
        
        Args:
            query: User query
            llm_function: Function to call an LLM
            multi_document: Whether to generate multiple hypothetical documents
            num_variants: Number of document variants to generate
            
        Returns:
            Tuple of (final_embedding, hypothetical_documents)
        """
        if not multi_document:
            # Single document approach
            hypothetical_doc = self.generate_hypothetical_document(query, llm_function)
            hypothetical_docs = [hypothetical_doc]
            
            # Generate embedding
            doc_embedding = self.embedding_model.encode(hypothetical_doc)
            return doc_embedding, hypothetical_docs
        else:
            # Multi-document approach for improved robustness
            hypothetical_docs = []
            
            # Generate variant prompts
            variant_suffixes = [
                "Focus on the most common interpretation of this question.",
                "Consider alternative or specialized perspectives on this topic.",
                "Include technical or domain-specific details in your answer."
            ]
            
            # Generate multiple hypothetical documents
            for i in range(min(num_variants, len(variant_suffixes) + 1)):
                if i == 0:
                    # First document uses the standard prompt
                    doc = self.generate_hypothetical_document(query, llm_function)
                else:
                    # Subsequent documents use variant prompts
                    modified_query = f"{query} {variant_suffixes[i-1]}"
                    doc = self.generate_hypothetical_document(modified_query, llm_function)
                
                hypothetical_docs.append(doc)
            
            # Generate embeddings for all documents
            doc_embeddings = self.embedding_model.encode(hypothetical_docs)
            
            # Calculate average embedding
            average_embedding = np.mean(doc_embeddings, axis=0)
            
            # Normalize to unit length
            final_embedding = average_embedding / np.linalg.norm(average_embedding)
            
            return final_embedding, hypothetical_docs
    
    def hyde_retrieval(self,
                       query: str,
                       chunk_embeddings: np.ndarray,
                       doc_ids: List[str],
                       llm_function: Callable,
                       top_k: int = 5,
                       multi_document: bool = False) -> List[Tuple[str, float]]:
        """
        Perform retrieval using HyDE
        
        Args:
            query: User query
            chunk_embeddings: Matrix of document chunk embeddings
            doc_ids: Document IDs corresponding to embeddings
            llm_function: Function to call an LLM
            top_k: Number of results to retrieve
            multi_document: Whether to use multiple hypothetical documents
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Process query using HyDE
        doc_embedding, hypothetical_docs = self.process_query(
            query, llm_function, multi_document=multi_document
        )
        
        # Calculate cosine similarity
        similarities = np.dot(chunk_embeddings, doc_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(doc_embedding)
        )
        
        # Get indices of top_k highest similarities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return doc_ids and scores
        results = [(doc_ids[idx], similarities[idx]) for idx in top_indices]
        
        return results, hypothetical_docs
    
    def hybrid_hyde_retrieval(self,
                             query: str,
                             chunk_embeddings: np.ndarray,
                             doc_ids: List[str],
                             llm_function: Callable,
                             query_weight: float = 0.3,
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform hybrid retrieval combining original query and HyDE
        
        Args:
            query: User query
            chunk_embeddings: Matrix of document chunk embeddings
            doc_ids: Document IDs corresponding to embeddings
            llm_function: Function to call an LLM
            query_weight: Weight for original query (1-query_weight for HyDE)
            top_k: Number of results to retrieve
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Get HyDE embedding
        hyde_embedding, hypothetical_docs = self.process_query(query, llm_function)
        
        # Calculate weighted embedding
        combined_embedding = (query_weight * query_embedding + 
                             (1 - query_weight) * hyde_embedding)
        
        # Normalize to unit length
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        # Calculate cosine similarity
        similarities = np.dot(chunk_embeddings, combined_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(combined_embedding)
        )
        
        # Get indices of top_k highest similarities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return doc_ids and scores
        results = [(doc_ids[idx], similarities[idx]) for idx in top_indices]
        
        return results, hypothetical_docs

# Example usage:
# hyde = EnhancedHyDE("all-mpnet-base-v2")
# results, hypothetical_docs = hyde.hyde_retrieval(query, chunk_embeddings, doc_ids, llm_function)
# hybrid_results, hybrid_docs = hyde.hybrid_hyde_retrieval(query, chunk_embeddings, doc_ids, llm_function)