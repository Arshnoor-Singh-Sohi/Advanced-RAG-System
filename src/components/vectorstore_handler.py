"""
Vector store handling module for the RAG system.

This module provides functions and classes for:
- Generating embeddings
- Vector search
- Vector store management
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv



def save_vectorstore(vectorstore, path: str) -> None:
    """
    Save a vector store to disk
    
    Args:
        vectorstore: The vector store to save
        path: Path to save the vector store
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Different vector stores have different save methods
        if hasattr(vectorstore, "save_local"):
            # For FAISS vector stores
            vectorstore.save_local(path)
        else:
        # Generic pickling as fallback
            with open(path, 'wb') as f:
                pickle.dump(vectorstore, f)
            
        print(f"Vector store saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Error saving vector store: {str(e)}")
    
def load_vectorstore(path: str, embeddings=None):
    """
    Load a vector store from disk
    
    Args:
        path: Path to the vector store
        embeddings: Optional embeddings for some vector store types
        
    Returns:
        Loaded vector store
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}")
        
    try:
        # Try to load as FAISS first
        try:
            import faiss
            from langchain.vectorstores import FAISS
            
            if embeddings:
                return FAISS.load_local(path, embeddings)
        except (ImportError, Exception):
            pass
            
        # Fall back to generic pickle loading
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading vector store: {str(e)}")

def get_vectorstore(texts: List[str], embeddings, ids: List[str] = None):
    """
    Create a vector store from texts and embeddings
    
    Args:
        texts: List of text strings
        embeddings: Embeddings instance or embedded vectors
        ids: Optional list of document IDs
        
    Returns:
        Vector store
    """
    try:
        from langchain.vectorstores import FAISS
        
        # Check if embeddings is already a numpy array of vectors
        if isinstance(embeddings, np.ndarray):
            # We need to create a FAISS index manually
            import faiss
            
            # Normalize vectors for cosine similarity
            normalized_vectors = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # IP = Inner Product for cosine similarity
            index.add(normalized_vectors)
            
            # Create a custom FAISS vectorstore
            vectorstore = FAISS(embeddings=embeddings, index=index, docstore=texts, ids=ids or [str(i) for i in range(len(texts))])
            return vectorstore
        else:
            # Assume embeddings is a LangChain embeddings object
            return FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
    except ImportError:
        raise ImportError("langchain and/or faiss-cpu packages not installed. Install with 'pip install langchain faiss-cpu'")
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {str(e)}")

@staticmethod
def get_retriever(vectorstore, use_mmr=False, k=4, fetch_k=20, lambda_mult=0.5):
    """
    Create a retriever from a vector store with the configured settings.
    
    Args:
        vectorstore: FAISS vector store
        use_mmr: Whether to use Maximal Marginal Relevance
        k: Number of documents to retrieve
        fetch_k: Number of documents to initially fetch for MMR (only used if use_mmr=True)
        lambda_mult: Balance between relevance and diversity for MMR (only used if use_mmr=True)
        
    Returns:
        A configured retriever
    """
    if use_mmr:
        # Create retriever with MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
    else:
        # Create standard similarity retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
    
    return retriever

class EmbeddingProvider:
    """Interface for different embedding models"""

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

        print(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        print(f"Finished encoding {len(texts)} texts.")
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
        Get embeddings using any HuggingFace model via transformers

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
            print(f"Loading HuggingFace tokenizer/model: {model_name}")
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

    