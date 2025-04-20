"""
RAG Components Module

This module contains the core components for a Retrieval-Augmented Generation (RAG) system:
- DocumentChunker: For splitting documents into chunks
- EmbeddingProvider: For generating embeddings using different models
- RetrievalMethods: For different retrieval approaches
- QueryProcessor: For query processing techniques
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
# Import heavy libraries only when needed or within methods/init
# from sentence_transformers import SentenceTransformer # Moved to __init__
# from langchain_openai import OpenAIEmbeddings # Moved to respective method
# from transformers import AutoTokenizer, AutoModel # Moved to respective method
# import torch # Moved to respective method
from dotenv import load_dotenv

# Import from refactored modules for backward compatibility
from src.components.data_processing import DocumentChunker
from src.components.vectorstore_handler import EmbeddingProvider, RetrievalMethods, save_vectorstore, load_vectorstore, get_vectorstore
from src.components.llm_integrations import LLMProvider, get_conversation_chain, generate_response

# Keep Query Processor in this file until it's refactored separately
from typing import List, Dict, Any, Tuple, Optional, Callable
import nltk


class DocumentChunker:
    """Class for chunking documents with different strategies"""
    
    @staticmethod
    def chunk_by_fixed_size(documents: List[Dict[str, str]], 
                           chunk_size: int,
                           overlap: int = 0) -> List[Dict[str, str]]:
        """
        Chunk documents by fixed token size
        
        Args:
            documents: List of documents with 'text' and other metadata
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            # Simple tokenization by words (in production you'd use a better tokenizer)
            tokens = nltk.word_tokenize(text)
            
            # Create chunks with the specified size and overlap
            for i in range(0, len(tokens), chunk_size - overlap):
                # Get chunk tokens
                chunk_tokens = tokens[i:i + chunk_size]
                
                # Skip chunks that are too small (last partial chunks)
                if len(chunk_tokens) < chunk_size // 2:
                    continue
                    
                # Combine tokens back into text
                chunk_text = " ".join(chunk_tokens)
                
                # Create a new document with the chunk and original metadata
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_{i//(chunk_size - overlap)}"
                
                chunked_docs.append(chunk_doc)
                
        return chunked_docs
    
    @staticmethod
    def chunk_by_paragraph(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk documents by paragraph breaks
        
        Args:
            documents: List of documents with 'text' and other metadata
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            
            # Split by paragraph breaks (double newline)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # If no clear paragraphs found, try single newlines
            if len(paragraphs) <= 1:
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
                
            # If still no clear paragraphs, keep as single chunk
            if len(paragraphs) <= 1:
                chunked_docs.append(doc)
                continue
                
            # Create new documents for each paragraph
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.split()) < 10:  # Skip very short paragraphs
                    continue
                    
                chunk_doc = doc.copy()
                chunk_doc['text'] = paragraph
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_p{i}"
                
                chunked_docs.append(chunk_doc)
                
        return chunked_docs
    
    @staticmethod
    def chunk_by_semantic_units(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk documents by trying to preserve semantic units
        This is a simplified version that uses sentence boundaries and length limits
        
        Args:
            documents: List of documents with 'text' and other metadata
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        chunked_docs = []
        target_chunk_size = 300  # Target word count per chunk
        
        for doc in documents:
            text = doc['text']
            
            # Split by sentences
            sentences = nltk.sent_tokenize(text)
            
            current_chunk = []
            current_size = 0
            chunk_num = 0
            
            for sentence in sentences:
                sentence_len = len(sentence.split())
                
                # If adding this sentence exceeds target size and we already have content,
                # finish the current chunk
                if current_size > 0 and current_size + sentence_len > target_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunk_doc = doc.copy()
                    chunk_doc['text'] = chunk_text
                    chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_s{chunk_num}"
                    chunked_docs.append(chunk_doc)
                    
                    # Start a new chunk
                    current_chunk = [sentence]
                    current_size = sentence_len
                    chunk_num += 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_size += sentence_len
                    
            # Don't forget to add the last chunk if it has content
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_s{chunk_num}"
                chunked_docs.append(chunk_doc)
                
        return chunked_docs


class EmbeddingProvider:
    """Interface for different embedding models (ORIGINAL STATIC VERSION)"""

    @staticmethod
    def get_sentence_transformer_embeddings(texts: List[str],
                                           model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """
        Get embeddings using SentenceTransformers models

        Args:
            texts: List of text strings to embed
            model_name: Name of the SentenceTransformers model to use

        Returns:
            Numpy array of embeddings
        """
        # Import here to avoid loading unless method is called
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
             raise ImportError("sentence-transformers package not installed. Run 'pip install sentence-transformers'")

        print(f"Loading SentenceTransformer model: {model_name} (Static Method Call)") # Added print for clarity
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"Finished encoding {len(texts)} texts.") # Added print
        return embeddings

    @staticmethod
    def get_openai_embeddings(texts: List[str],
                             model_name: str = "text-embedding-ada-002") -> np.ndarray:
        """
        Get embeddings using OpenAI's embedding API

        Args:
            texts: List of text strings to embed
            model_name: Name of the OpenAI embedding model

        Returns:
            Numpy array of embeddings
        """
        try:
            # Imports specific to this method
            from langchain_openai import OpenAIEmbeddings
            from dotenv import load_dotenv
            load_dotenv() # Ensure environment variables are loaded

            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")

            embeddings_model = OpenAIEmbeddings(model=model_name)

            # Process in batches
            batch_size = 100 # Or another suitable size
            all_embeddings = []
            print(f"Getting OpenAI embeddings for {len(texts)} texts...")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            print("Finished getting OpenAI embeddings.")
            return np.array(all_embeddings)
        except ImportError:
            raise ImportError("langchain-openai or python-dotenv package not installed. Install with 'pip install langchain-openai python-dotenv'")
        except Exception as e:
            # Add more specific error handling if needed
            raise RuntimeError(f"Error getting OpenAI embeddings: {str(e)}")

    @staticmethod
    def get_huggingface_embeddings(texts: List[str],
                                  model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
        """
        Get embeddings using any HuggingFace model via transformers (ORIGINAL STATIC VERSION)

        Args:
            texts: List of text strings to embed
            model_name: HuggingFace model identifier

        Returns:
            Numpy array of embeddings
        """
        try:
            # Imports specific to this method
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
             raise ImportError("transformers or torch package not installed. Run 'pip install transformers torch'")

        try:
            print(f"Loading HuggingFace tokenizer/model: {model_name} (Static Method Call)")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print("Finished loading HuggingFace model.")

            # Process in batches
            batch_size = 32 # Adjust as needed
            all_embeddings = []
            print(f"Getting HuggingFace embeddings for {len(texts)} texts...")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded_input = tokenizer(batch_texts, padding=True, truncation=True,
                                         return_tensors='pt', max_length=512) # Ensure max_length is suitable

                with torch.no_grad():
                    model_output = model(**encoded_input)

                # Example: CLS token embedding. Adjust if mean pooling etc. was intended.
                embeddings = model_output.last_hidden_state[:, 0].cpu().numpy() # Use CLS token, move to CPU
                all_embeddings.extend(embeddings)
            print("Finished getting HuggingFace embeddings.")
            return np.array(all_embeddings)
        except Exception as e:
            # Add more specific error handling if needed
            raise RuntimeError(f"Error getting HuggingFace embeddings: {str(e)}")

class RetrievalMethods:
    """Implementation of different retrieval approaches"""
    
    @staticmethod
    def vector_search(query_embedding: np.ndarray, 
                     document_embeddings: np.ndarray,
                     doc_ids: List[str], 
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search
        
        Args:
            query_embedding: Embedding vector of the query
            document_embeddings: Matrix of document embeddings
            doc_ids: Document identifiers corresponding to the embeddings
            top_k: Number of top results to return
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # Calculate cosine similarity
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get indices of top_k highest similarities
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return doc_ids and scores
        results = [(doc_ids[idx], similarities[idx]) for idx in top_indices]
        
        return results
    
    @staticmethod
    def bm25_search(query: str, 
                   documents: List[Dict[str, str]],
                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            documents: List of documents with 'text' field
            top_k: Number of top results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        try:
            import nltk
            from rank_bm25 import BM25Okapi
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            # Extract document texts and IDs
            doc_texts = [doc['text'] for doc in documents]
            doc_ids = [doc.get('chunk_id', f"doc_{i}") for i, doc in enumerate(documents)]
            
            # Tokenize documents and query
            tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in doc_texts]
            tokenized_query = nltk.word_tokenize(query.lower())
            
            # Create BM25 object
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores
            scores = bm25.get_scores(tokenized_query)
            
            # Get indices of top_k highest scores
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Return doc_ids and scores
            results = [(doc_ids[idx], scores[idx]) for idx in top_indices]
            
            return results
        except ImportError:
            raise ImportError("rank_bm25 package not installed. Install with 'pip install rank-bm25'")
        except Exception as e:
            raise RuntimeError(f"Error in BM25 search: {str(e)}")
    
    @staticmethod
    def hybrid_search(query: str, 
                     query_embedding: np.ndarray,
                     documents: List[Dict[str, str]], 
                     document_embeddings: np.ndarray,
                     alpha: float = 0.5, 
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining vector and keyword approaches
        
        Args:
            query: Search query
            query_embedding: Embedding of the query
            documents: List of documents with 'text' field
            document_embeddings: Matrix of document embeddings
            alpha: Weight for vector search (1-alpha for keyword search)
            top_k: Number of top results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Extract document IDs
        doc_ids = [doc.get('chunk_id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Get results from both methods
        vector_results = RetrievalMethods.vector_search(
            query_embedding, document_embeddings, doc_ids, top_k=top_k*2
        )
        
        bm25_results = RetrievalMethods.bm25_search(
            query, documents, top_k=top_k*2
        )
        
        # Combine results
        score_map = {}
        
        # Add vector scores
        for doc_id, score in vector_results:
            score_map[doc_id] = alpha * score
            
        # Add BM25 scores (normalize first)
        bm25_scores = [score for _, score in bm25_results]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        
        for doc_id, score in bm25_results:
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0
            score_map[doc_id] = score_map.get(doc_id, 0) + (1 - alpha) * normalized_score
            
        # Sort by combined score
        combined_results = [(doc_id, score) for doc_id, score in score_map.items()]
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:top_k]



    """Implements query processing techniques like expansion, reformulation, etc."""
    
    @staticmethod
    def expand_query(query: str, method: str = "simple") -> List[str]:
        """
        Expand a query to improve retrieval recall
        
        Args:
            query: Original query
            method: Method for expansion ('simple', 'wordnet', etc.)
        
        Returns:
            List of expanded queries
        """
        if method == "simple":
            # Simple query expansion with related forms
            import nltk
            from nltk.corpus import wordnet
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                
            # Tokenize query
            tokens = nltk.word_tokenize(query.lower())
            
            expanded_queries = [query]  # Start with original query
            
            # Generate query variations with synonyms
            for i, token in enumerate(tokens):
                # Skip short tokens and stopwords
                if len(token) <= 3:
                    continue
                    
                # Get WordNet synsets
                synsets = wordnet.synsets(token)
                if not synsets:
                    continue
                    
                # Get synonyms from first synset
                synonyms = [lemma.name() for lemma in synsets[0].lemmas()]
                
                # Create new queries with synonyms
                for synonym in synonyms[:2]:  # Limit to 2 synonyms
                    if synonym != token and "_" not in synonym:
                        new_tokens = tokens.copy()
                        new_tokens[i] = synonym
                        expanded_queries.append(" ".join(new_tokens))
                        
            return expanded_queries
        
        elif method == "llm":
            # This would use an LLM to generate query variations
            # For demo purposes, just return a few manually created variations
            variations = [
                query,
                f"information about {query}",
                f"explain {query}"
            ]
            return variations
        
        else:
            # Default to original query
            return [query]
    
    @staticmethod
    def hyde_expansion(query: str, llm_function) -> Tuple[str, str]:
        """
        Implement HyDE (Hypothetical Document Embeddings) approach
        
        Args:
            query: Original query
            llm_function: Function to call an LLM
        
        Returns:
            Tuple of (original_query, hypothetical_document)
        """
        # Create a prompt for generating a hypothetical document
        hyde_prompt = f"""
        Write a short but detailed passage that would contain the answer to the question: '{query}'
        Make the passage factual, objective, and comprehensive.
        The passage should be 3-5 sentences long.
        """
        
        # Generate hypothetical document using provided LLM function
        hypothetical_doc = llm_function(hyde_prompt)
        
        return query, hypothetical_doc
    

class QueryProcessor:
    """Implements query processing techniques like expansion, reformulation, etc."""
    
    @staticmethod
    def expand_query(query: str, method: str = "simple") -> List[str]:
        """
        Expand a query to improve retrieval recall
        
        Args:
            query: Original query
            method: Method for expansion ('simple', 'wordnet', etc.)
        
        Returns:
            List of expanded queries
        """
        if method == "simple":
            # Simple query expansion with related forms
            import nltk
            from nltk.corpus import wordnet
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                
            # Tokenize query
            tokens = nltk.word_tokenize(query.lower())
            
            expanded_queries = [query]  # Start with original query
            
            # Generate query variations with synonyms
            for i, token in enumerate(tokens):
                # Skip short tokens and stopwords
                if len(token) <= 3:
                    continue
                    
                # Get WordNet synsets
                synsets = wordnet.synsets(token)
                if not synsets:
                    continue
                    
                # Get synonyms from first synset
                synonyms = [lemma.name() for lemma in synsets[0].lemmas()]
                
                # Create new queries with synonyms
                for synonym in synonyms[:2]:  # Limit to 2 synonyms
                    if synonym != token and "_" not in synonym:
                        new_tokens = tokens.copy()
                        new_tokens[i] = synonym
                        expanded_queries.append(" ".join(new_tokens))
                        
            return expanded_queries
        
        elif method == "llm":
            # This would use an LLM to generate query variations
            # For demo purposes, just return a few manually created variations
            variations = [
                query,
                f"information about {query}",
                f"explain {query}"
            ]
            return variations
        
        else:
            # Default to original query
            return [query]
    
    @staticmethod
    def hyde_expansion(query: str, llm_function) -> Tuple[str, str]:
        """
        Implement HyDE (Hypothetical Document Embeddings) approach
        
        Args:
            query: Original query
            llm_function: Function to call an LLM
        
        Returns:
            Tuple of (original_query, hypothetical_document)
        """
        # Create a prompt for generating a hypothetical document
        hyde_prompt = f"""
        Write a short but detailed passage that would contain the answer to the question: '{query}'
        Make the passage factual, objective, and comprehensive.
        The passage should be 3-5 sentences long.
        """
        
        # Generate hypothetical document using provided LLM function
        hypothetical_doc = llm_function(hyde_prompt)
        
        return query, hypothetical_doc