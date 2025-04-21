"""
Query Transformation Module

This module implements various query transformation techniques:
- MultiQuery: Generate multiple query variations using LLMs
- HyDE: Hypothetical Document Embeddings (integration with enhanced_hyde.py)
- Query Expansion: Simple query expansion techniques

These transformations help improve retrieval quality by addressing vocabulary
mismatch and other challenges in RAG systems.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import numpy as np
import os
import re
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

class MultiQueryGenerator:
    """Generate multiple query variations using LLM to improve retrieval recall"""
    
    def __init__(self, llm_function: Optional[Callable] = None):
        """
        Initialize MultiQuery generator
        
        Args:
            llm_function: Optional callable that takes a prompt and returns text
        """
        self.llm_function = llm_function
        # Default prompt template for generating query variations
        self.prompt_template = """
        Generate {num_variations} different versions of the following question.
        Each version should ask for the same information but with different wording, style, or perspective.
        Keep each version clear, natural, and concise.
        Return ONLY the variations, one per line, with no numbers or prefixes.

        Original question: {query}
        """
    
    def set_llm_function(self, llm_function: Callable):
        """
        Set the LLM function to use for query generation
        
        Args:
            llm_function: Function that takes a prompt and returns generated text
        """
        self.llm_function = llm_function
    
    def generate_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of the input query
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations including the original query
        """
        # Start with the original query
        variations = [query]
        
        # If no LLM function available, use rule-based variations
        if self.llm_function is None:
            # Simple rule-based variations (fallback if no LLM available)
            simple_variations = self._generate_simple_variations(query)
            variations.extend(simple_variations[:num_variations-1])
            return variations
        
        # Use LLM to generate variations
        prompt = self.prompt_template.format(
            query=query, 
            num_variations=num_variations
        )
        
        try:
            # Generate variations using the LLM
            response = self.llm_function(prompt)
            
            # Process the response to extract variations
            # Split by newlines and clean up
            llm_variations = [
                line.strip() for line in response.split('\n')
                if line.strip() and line.strip() != query
            ]
            
            # Add unique variations (avoid duplicates of original query)
            for var in llm_variations:
                if var not in variations:
                    variations.append(var)
                    
                # Stop once we have enough variations
                if len(variations) >= num_variations:
                    break
                    
        except Exception as e:
            print(f"Error generating query variations: {e}")
            # Fallback to simple variations
            simple_variations = self._generate_simple_variations(query)
            variations.extend(simple_variations[:num_variations-1])
        
        # Ensure we return at most num_variations
        return variations[:num_variations]
    
    def _generate_simple_variations(self, query: str) -> List[str]:
        """
        Generate simple rule-based variations of the query
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = []
        
        # Variation 1: Add "please" prefix
        if not query.lower().startswith("please"):
            variations.append(f"Please {query}")
            
        # Variation 2: Convert to "What" question if not already one
        if not query.lower().startswith("what"):
            variations.append(f"What about {query}")
            
        # Variation 3: Add "I want to know" prefix
        if not query.lower().startswith("i want to know"):
            variations.append(f"I want to know {query}")
            
        # Variation 4: Convert to "How" question if not already one
        if not query.lower().startswith("how"):
            variations.append(f"How does {query} work")
            
        # Variation 5: Ask for explanation
        if not "explain" in query.lower():
            variations.append(f"Explain {query}")
            
        # Return unique variations
        return [v for v in variations if v != query]

class QueryTransformationRouter:
    """Router to different query transformation techniques"""
    
    def __init__(self):
        """Initialize the query transformation router"""
        # We'll keep references to various transformation strategies
        self.multi_query_generator = MultiQueryGenerator()
        self.hyde_initialized = False
        self.hyde_generator = None
        
    def initialize_hyde(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the HyDE generator (lazy initialization to avoid loading models unnecessarily)
        
        Args:
            model_name: Name of the embedding model to use
        """
        try:
            from src.components.enhanced_hyde import EnhancedHyDE
            self.hyde_generator = EnhancedHyDE(embedding_model=model_name)
            self.hyde_initialized = True
            print(f"HyDE initialized with model: {model_name}")
        except Exception as e:
            print(f"Failed to initialize HyDE: {e}")
            self.hyde_initialized = False
    
    def transform_query(self, 
                       query: str, 
                       method: str = "none",
                       llm_function: Optional[Callable] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Transform a query using the specified method
        
        Args:
            query: Original query
            method: Transformation method ("none", "multi_query", "hyde", "expansion")
            llm_function: Optional function for LLM-based transformations
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dict with transformation results (varies by method)
        """
        if method == "none":
            # No transformation, return original query
            return {"transformed_queries": [query], "original_query": query}
        
        elif method == "multi_query":
            # Use MultiQuery technique
            num_variations = kwargs.get("num_variations", 3)
            
            # Set the LLM function if provided
            if llm_function:
                self.multi_query_generator.set_llm_function(llm_function)
                
            # Generate variations
            variations = self.multi_query_generator.generate_variations(
                query=query,
                num_variations=num_variations
            )
            
            return {
                "transformed_queries": variations,
                "original_query": query,
                "method": "multi_query"
            }
            
        elif method == "hyde":
            # Use HyDE technique
            if not self.hyde_initialized:
                model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
                self.initialize_hyde(model_name)
            
            if not self.hyde_initialized or not self.hyde_generator:
                # Fallback if HyDE initialization failed
                return {"transformed_queries": [query], "original_query": query}
            
            # Generate hypothetical document
            multi_document = kwargs.get("multi_document", False)
            num_variants = kwargs.get("num_variants", 3)
            
            try:
                # Process query using HyDE
                doc_embedding, hypothetical_docs = self.hyde_generator.process_query(
                    query=query,
                    llm_function=llm_function,
                    multi_document=multi_document,
                    num_variants=num_variants
                )
                
                return {
                    "transformed_queries": [query],  # Original query still used for logging
                    "original_query": query,
                    "doc_embedding": doc_embedding,
                    "hypothetical_docs": hypothetical_docs,
                    "method": "hyde"
                }
            except Exception as e:
                print(f"Error in HyDE processing: {e}")
                return {"transformed_queries": [query], "original_query": query}
            
        elif method == "expansion":
            # Simple query expansion technique
            try:
                from src.components.rag_components import QueryProcessor
                expanded_queries = QueryProcessor.expand_query(
                    query=query,
                    method=kwargs.get("expansion_method", "simple")
                )
                
                return {
                    "transformed_queries": expanded_queries,
                    "original_query": query,
                    "method": "expansion"
                }
            except Exception as e:
                print(f"Error in query expansion: {e}")
                return {"transformed_queries": [query], "original_query": query}
        
        else:
            # Unknown method, return original query
            print(f"Unknown query transformation method: {method}")
            return {"transformed_queries": [query], "original_query": query}

# Example usage:
# router = QueryTransformationRouter()
# result = router.transform_query(
#     query="How does vector search work?",
#     method="multi_query",
#     llm_function=some_llm_function,
#     num_variations=3
# )
# transformed_queries = result["transformed_queries"]