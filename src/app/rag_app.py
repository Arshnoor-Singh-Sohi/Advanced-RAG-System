"""
RAG Application

This module provides a command-line RAG application that uses the components
from the RAG system to answer questions using a knowledge base.
"""

import os
import sys
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import RAG components
from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods, QueryProcessor
from src.components.reranking import RerankerModule
from src.components.evaluation import RAGEvaluator


class RAGApplication:
    """
    RAG Application for question answering using a knowledge base
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the RAG application
        
        Args:
            config_path: Path to configuration file
        """
        # Set default configuration
        self.config = {
            # Data and knowledge base configuration
            "corpus_path": "data/corpus.pkl",
            "index_path": "data/index.pkl",
            
            # Chunking configuration
            "chunking_strategy": "fixed",
            "chunk_size": 128,
            "chunk_overlap": 32,
            
            # Embedding configuration
            "embedding_model": "all-MiniLM-L6-v2",
            
            # Retrieval configuration
            "retrieval_method": "hybrid",
            "retrieval_alpha": 0.7,  # Weight for vector search in hybrid retrieval
            "top_k": 5,
            
            # Query processing configuration
            "query_expansion": False,
            "expansion_method": "simple",
            
            # Reranking configuration
            "use_reranking": True,
            "reranking_method": "cross_encoder",
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            
            # Generation configuration
            "prompt_template": "Answer the question based ONLY on the following context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        }
        
        # Override with provided configuration if available
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            print(f"Loaded configuration from {config_path}")
        
        # State variables
        self.corpus = None
        self.chunked_docs = None
        self.doc_embeddings = None
        self.doc_ids = None
        self.conversation_history = []
        
        print("RAG application initialized")
        
    def load_corpus(self, corpus_path: str = None):
        """
        Load the document corpus
        
        Args:
            corpus_path: Path to corpus file (optional, uses config if not provided)
        """
        path = corpus_path or self.config["corpus_path"]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")
            
        with open(path, 'rb') as f:
            self.corpus = pickle.load(f)
            
        print(f"Loaded corpus with {len(self.corpus)} documents")
        
    def prepare_knowledge_base(self, force_rebuild: bool = False):
        """
        Prepare the knowledge base by chunking documents and generating embeddings
        
        Args:
            force_rebuild: Whether to force rebuilding the index
        """
        index_path = self.config["index_path"]
        
        # Try to load pre-built index if it exists and not forcing rebuild
        if os.path.exists(index_path) and not force_rebuild:
            try:
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
                    
                self.chunked_docs = index_data.get("chunked_docs")
                self.doc_embeddings = index_data.get("doc_embeddings")
                self.doc_ids = index_data.get("doc_ids")
                
                if (self.chunked_docs is not None and 
                    self.doc_embeddings is not None and 
                    self.doc_ids is not None):
                    print(f"Loaded pre-built index from {index_path}")
                    print(f"Index contains {len(self.chunked_docs)} chunks")
                    return
            except Exception as e:
                print(f"Error loading index: {e}")
                print("Rebuilding index...")
        
        # Load corpus if not already loaded
        if self.corpus is None:
            self.load_corpus()
            
        # Apply chunking strategy
        chunking_strategy = self.config["chunking_strategy"]
        
        print(f"Chunking documents using '{chunking_strategy}' strategy...")
        
        if chunking_strategy == "fixed":
            chunk_size = self.config["chunk_size"]
            chunk_overlap = self.config["chunk_overlap"]
            self.chunked_docs = DocumentChunker.chunk_by_fixed_size(
                self.corpus, chunk_size=chunk_size, overlap=chunk_overlap
            )
        elif chunking_strategy == "paragraph":
            self.chunked_docs = DocumentChunker.chunk_by_paragraph(self.corpus)
        elif chunking_strategy == "semantic":
            self.chunked_docs = DocumentChunker.chunk_by_semantic_units(self.corpus)
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
            
        print(f"Created {len(self.chunked_docs)} chunks")
        
        # Generate embeddings
        embedding_model = self.config["embedding_model"]
        
        print(f"Generating embeddings using {embedding_model}...")
        
        chunk_texts = [doc["text"] for doc in self.chunked_docs]
        self.doc_ids = [doc["chunk_id"] for doc in self.chunked_docs]
        
        self.doc_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
            chunk_texts, model_name=embedding_model
        )
        
        print(f"Generated embeddings of shape {self.doc_embeddings.shape}")
        
        # Save index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        with open(index_path, 'wb') as f:
            index_data = {
                "chunked_docs": self.chunked_docs,
                "doc_embeddings": self.doc_embeddings,
                "doc_ids": self.doc_ids,
                "config": {
                    "chunking_strategy": chunking_strategy,
                    "embedding_model": embedding_model,
                    "chunk_size": self.config.get("chunk_size"),
                    "chunk_overlap": self.config.get("chunk_overlap")
                }
            }
            pickle.dump(index_data, f)
            
        print(f"Saved index to {index_path}")
    
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process a query and return an answer with supporting contexts
        
        Args:
            query: The query string
            
        Returns:
            Tuple of (answer, context_documents)
        """
        # Ensure knowledge base is prepared
        if self.chunked_docs is None or self.doc_embeddings is None:
            self.prepare_knowledge_base()
            
        # Add query to conversation history
        self.conversation_history.append(query)
        
        # Apply query expansion if enabled
        if self.config["query_expansion"]:
            expansion_method = self.config["expansion_method"]
            expanded_queries = QueryProcessor.expand_query(query, method=expansion_method)
            print(f"Expanded query to {len(expanded_queries)} variations")
        else:
            expanded_queries = [query]
            
        # Retrieve relevant documents
        retrieval_method = self.config["retrieval_method"]
        top_k = self.config["top_k"]
        
        print(f"Retrieving documents using '{retrieval_method}' method...")
        
        # Get embeddings for the query
        query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
            [query], model_name=self.config["embedding_model"]
        )[0]
        
        # Retrieve documents based on method
        retrieved_docs = []
        
        if retrieval_method == "vector":
            # For each expanded query, get results and combine
            all_doc_ids = []
            
            for exp_query in expanded_queries:
                exp_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                    [exp_query], model_name=self.config["embedding_model"]
                )[0]
                
                results = RetrievalMethods.vector_search(
                    exp_embedding, self.doc_embeddings, self.doc_ids, top_k=top_k
                )
                
                all_doc_ids.extend([doc_id for doc_id, _ in results])
                
            # Remove duplicates while preserving order
            unique_doc_ids = []
            for doc_id in all_doc_ids:
                if doc_id not in unique_doc_ids:
                    unique_doc_ids.append(doc_id)
                    
            # Get the actual documents
            retrieved_docs = []
            for doc_id in unique_doc_ids[:top_k]:
                idx = self.doc_ids.index(doc_id)
                retrieved_docs.append(self.chunked_docs[idx])
                
        elif retrieval_method == "bm25":
            # For each expanded query, get results and combine
            all_doc_ids = []
            
            for exp_query in expanded_queries:
                results = RetrievalMethods.bm25_search(
                    exp_query, self.chunked_docs, top_k=top_k
                )
                
                all_doc_ids.extend([doc_id for doc_id, _ in results])
                
            # Remove duplicates while preserving order
            unique_doc_ids = []
            for doc_id in all_doc_ids:
                if doc_id not in unique_doc_ids:
                    unique_doc_ids.append(doc_id)
                    
            # Get the actual documents
            retrieved_docs = []
            for doc_id in unique_doc_ids[:top_k]:
                idx = self.doc_ids.index(doc_id)
                retrieved_docs.append(self.chunked_docs[idx])
                
        elif retrieval_method == "hybrid":
            # For each expanded query, get results and combine
            all_doc_ids = []
            
            for exp_query in expanded_queries:
                exp_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                    [exp_query], model_name=self.config["embedding_model"]
                )[0]
                
                results = RetrievalMethods.hybrid_search(
                    exp_query, exp_embedding, self.chunked_docs, self.doc_embeddings,
                    alpha=self.config["retrieval_alpha"], top_k=top_k
                )
                
                all_doc_ids.extend([doc_id for doc_id, _ in results])
                
            # Remove duplicates while preserving order
            unique_doc_ids = []
            for doc_id in all_doc_ids:
                if doc_id not in unique_doc_ids:
                    unique_doc_ids.append(doc_id)
                    
            # Get the actual documents
            retrieved_docs = []
            for doc_id in unique_doc_ids[:top_k]:
                idx = self.doc_ids.index(doc_id)
                retrieved_docs.append(self.chunked_docs[idx])
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
            
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Apply reranking if enabled
        if self.config["use_reranking"]:
            reranking_method = self.config["reranking_method"]
            
            print(f"Reranking documents using '{reranking_method}' method...")
            
            if reranking_method == "cross_encoder":
                reranking_model = self.config["reranking_model"]
                
                try:
                    reranked_pairs = RerankerModule.score_with_cross_encoder(
                        query, retrieved_docs, model_name=reranking_model
                    )
                    
                    # Extract reranked documents
                    retrieved_docs = [doc for doc, _ in reranked_pairs]
                except Exception as e:
                    print(f"Error in reranking: {e}")
                    print("Using original retrieval order")
            
            elif reranking_method == "contextual":
                # Use conversation history for reranking
                reranked_pairs = RerankerModule.contextual_reranking(
                    query, retrieved_docs, 
                    conversation_history=self.conversation_history[:-1]  # Exclude current query
                )
                
                # Extract reranked documents
                retrieved_docs = [doc for doc, _ in reranked_pairs]
                
            elif reranking_method == "diversity":
                # Rerank for diversity
                reranked_pairs = RerankerModule.diversity_reranking(
                    query, retrieved_docs, alpha=0.7
                )
                
                # Extract reranked documents
                retrieved_docs = [doc for doc, _ in reranked_pairs]
                
        # Generate answer
        context_texts = [doc["text"] for doc in retrieved_docs[:top_k]]
        context_str = "\n\n".join(context_texts)
        
        # Use the specified prompt template for generation
        prompt_template = self.config["prompt_template"]
        prompt = prompt_template.format(context=context_str, query=query)
        
        # Generate answer based on selected mode
        response_mode = getattr(self, "response_mode", "extractive")
        
        if response_mode == "llm":
            # Use LLM for answer generation
            answer = self._llm_answer(query, context_texts)
        else:
            # Use extractive approach
            answer = self._extractive_answer(query, context_texts)
        
        return answer, retrieved_docs
    

    def _llm_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using OpenAI's language model"""
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Prepare the prompt with context and query
            context_str = "\n\n".join(contexts)
            prompt = f"""
            Answer the question based ONLY on the following context:
            
            Context:
            {context_str}
            
            Question: {query}
            
            Provide a comprehensive, well-structured answer. If the information is not in the context, say you don't have enough information.
            """
            temperature = getattr(self, "temperature", 0.3)

            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use "gpt-4" for better quality if you have access
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based only on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,  # Lower temperature for more consistent responses
                max_tokens=500  # Adjust based on how detailed you want responses to be
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Log the error and fall back to extractive method
            print(f"Error using OpenAI API: {e}")
            return self._extractive_answer(query, contexts)
        
    def _extractive_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate a simple extractive answer from contexts
        
        Args:
            query: The query string
            contexts: List of context texts
            
        Returns:
            Extractive answer
        """
        import re
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Extract sentences from contexts
        all_sentences = []
        for context in contexts:
            sentences = nltk.sent_tokenize(context)
            all_sentences.extend(sentences)
            
        # Score sentences by similarity to query
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in all_sentences:
            # Simple word overlap scoring
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            if overlap > 0:
                # Score by overlap and sentence length
                score = overlap / (1 + abs(len(sentence_words) - len(query_words))*0.1)
                scored_sentences.append((sentence, score))
                
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # If no matching sentences, return a generic response
        if not scored_sentences:
            return "I couldn't find a specific answer to your question in the available knowledge base."
            
        # Take top sentences and combine into an answer
        top_sentences = [s for s, _ in scored_sentences[:3]]
        
        # If all sentences are very short, combine them
        if all(len(s.split()) < 10 for s in top_sentences):
            answer = " ".join(top_sentences)
        else:
            # Otherwise, use the top sentence with some context
            answer = top_sentences[0]
            
            # Add a second sentence for context if available and different enough
            if len(top_sentences) > 1:
                # Check if second sentence is different enough
                words1 = set(top_sentences[0].lower().split())
                words2 = set(top_sentences[1].lower().split())
                
                # If overlap is less than 70%, add it
                if len(words1.intersection(words2)) / len(words1.union(words2)) < 0.7:
                    answer += " " + top_sentences[1]
                    
        return answer
    
    def run_interactive(self):
        """Run an interactive session with the RAG system"""
        # Prepare knowledge base
        self.prepare_knowledge_base()
        
        print("\n===== RAG Interactive Mode =====")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'config' to see the current configuration")
        print("==========================================\n")
        
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting interactive mode")
                break
                
            if query.lower() == 'config':
                print("\nCurrent Configuration:")
                for key, value in self.config.items():
                    print(f"  {key}: {value}")
                continue
                
            try:
                answer, contexts = self.process_query(query)
                
                print("\n--- Answer ---")
                print(answer)
                
                print("\n--- Sources ---")
                for i, doc in enumerate(contexts[:3]):
                    source = doc.get("title", doc.get("chunk_id", f"Document {i+1}"))
                    print(f"{i+1}. {source}")
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                import traceback
                traceback.print_exc()


# Command line interface
def main():
    parser = argparse.ArgumentParser(description="RAG Application")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the index")
    parser.add_argument("--query", type=str, help="Run a single query")
    
    args = parser.parse_args()
    
    # Initialize the RAG application
    app = RAGApplication(config_path=args.config)
    
    # Load corpus if specified
    if args.corpus:
        app.load_corpus(args.corpus)
        
    # Prepare knowledge base
    app.prepare_knowledge_base(force_rebuild=args.rebuild_index)
    
    # Run single query or interactive mode
    if args.query:
        answer, contexts = app.process_query(args.query)
        
        print("\n--- Answer ---")
        print(answer)
        
        print("\n--- Sources ---")
        for i, doc in enumerate(contexts[:3]):
            source = doc.get("title", doc.get("chunk_id", f"Document {i+1}"))
            print(f"{i+1}. {source}")
    else:
        app.run_interactive()


if __name__ == "__main__":
    main()