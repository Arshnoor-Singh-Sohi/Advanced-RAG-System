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
import streamlit as st
import traceback
from langchain_community.vectorstores import FAISS

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import RAG components
from src.components.reranking import RerankerModule
from src.components.evaluation import RAGEvaluator

# In src/app/rag_app.py

# Replace the original imports with these:
from src.components.data_processing import DocumentChunker, get_text_from_file, create_document_from_text
from src.components.vectorstore_handler import EmbeddingProvider, RetrievalMethods, save_vectorstore, load_vectorstore
from src.components.llm_integrations import LLMProvider, get_conversation_chain, generate_response, extract_answer_from_context
from src.components.rag_components import QueryProcessor
from src.utils.utils import ensure_directory, save_json, load_json
from src.components.llm_integrations import get_embedding_model
from src.components.rag_components import EmbeddingProvider, RetrievalMethods, QueryProcessor



class RAGApplication:
    """
    RAG Application for question answering using a knowledge base
    """
    def __init__(self):
        """
        Initialize the RAG application state variables.
        Configuration will be set externally after initialization via the 'config' attribute,
        typically from Streamlit session state.
        """
        # Configuration dictionary - will be populated externally
        self.config = {}

        # State variables for data and components
        self.corpus = None                   # List of document dicts (e.g., {"title":..., "text":...})
        self.chunked_docs = []               # List of chunked text strings or Langchain Document objects
        self.embedding_model_instance = None # Loaded Langchain embedding model instance
        self.vector_store = None             # Loaded/Built Langchain VectorStore instance (e.g., FAISS)
        self.retriever = None                # Configured Langchain retriever instance
        self.bm25_index = None               # Placeholder for BM25 index object if implementing separate BM25
        self.conversation_history = []       # Internal history (optional, if not solely managed by Streamlit)

        # Attributes related to the previous manual index format (likely obsolete with VectorStore)
        # self.doc_embeddings = None         # Replaced by vector_store internal embeddings
        # self.doc_ids = None                # Replaced by vector_store internal IDs/mapping
    
    
    def _load_embedding_model(self):
        """Loads the embedding model based on the current configuration."""
        model_identifier = self.config.get("embedding_model", "all-MiniLM-L6-v2") # Default
        print(f"Attempting to load embedding model: {model_identifier}")
        # Pass the config which might contain the API key
        self.embedding_model_instance = get_embedding_model(model_identifier, self.config)
        if self.embedding_model_instance is None:
            # Handle error: Log, raise exception, or use a default fallback model
            print(f"ERROR: Failed to load embedding model '{model_identifier}'. Check configuration and API keys.")
            # Example fallback (optional):
            # self.embedding_model_instance = get_embedding_model("all-MiniLM-L6-v2", self.config)
            # Or raise an exception:
            raise ValueError(f"Failed to initialize embedding model: {model_identifier}")

    # src/app/rag_app.py (add this method inside RAGApplication class)
    def _load_or_get_embedding_model(self):
        """Loads or retrieves the cached embedding model based on config."""
        # Check if model needs reloading (e.g., identifier changed)
        current_config_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        model_needs_reload = False
        if self.embedding_model_instance is None:
            model_needs_reload = True
        else:
            # Crude check if model name seems to have changed. Langchain model objects
            # don't always have an easy way to get the original identifier back.
            # This check might need refinement based on the objects returned by get_embedding_model.
            # For now, we reload if the config identifier changes significantly.
            # A better approach might involve storing the identifier with the instance.
            # This simple check reloads if the identifier in config changes.
            # We rely on st.cache_resource in get_embedding_model to avoid redundant computation.
            pass # For now, rely on cache and reload mainly if None

        # For simplicity, let's use the cache_resource in get_embedding_model
        # Call the cached function - it will return cached instance or load anew
        try:
            print(f"Requesting embedding model: {current_config_model}")
            # Pass config for potential API key usage
            loaded_model = get_embedding_model(current_config_model, self.config)
            if loaded_model is None:
                 raise ValueError(f"get_embedding_model returned None for {current_config_model}")
            self.embedding_model_instance = loaded_model # Store the instance
            print(f"Using embedding model instance: {type(self.embedding_model_instance)}")
            return True # Indicate success
        except Exception as e:
             st.error(f"Fatal Error: Could not load embedding model '{current_config_model}'. Cannot proceed.")
             traceback.print_exc()
             self.embedding_model_instance = None # Ensure it's None on failure
             return False # Indicate failure

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
        
    # src/app/rag_app.py (add this method inside RAGApplication class)
    def _create_retriever(self):
         """Creates the retriever based on config after vector store is ready."""
         if self.vector_store is None:
              print("Cannot create retriever: Vector store not available.")
              self.retriever = None
              return

         top_k = self.config.get("top_k", 5)
         use_mmr = self.config.get("use_mmr", False)
         retrieval_method = self.config.get("retrieval_method", "hybrid")

         search_type = "similarity"
         search_kwargs = {'k': top_k}

         if use_mmr and retrieval_method in ["vector", "hybrid"]:
              search_type = "mmr"
              fetch_k = self.config.get("mmr_fetch_k", max(20, top_k * 2))
              if fetch_k < top_k: fetch_k = top_k # Ensure fetch_k >= k
              # Use 'mmr_lambda' to match UI key
              lambda_mult = self.config.get("mmr_lambda", 0.5)
              search_kwargs['fetch_k'] = fetch_k
              search_kwargs['lambda_mult'] = lambda_mult # Langchain uses lambda_mult
              print(f"Creating retriever: type='mmr', kwargs={search_kwargs}")
         else:
              if use_mmr: print(f"Warning: MMR ignored for retrieval method '{retrieval_method}'.")
              print(f"Creating retriever: type='similarity', kwargs={search_kwargs}")

         try:
              self.retriever = self.vector_store.as_retriever(
                   search_type=search_type,
                   search_kwargs=search_kwargs
              )
              print("Retriever created successfully.")
         except Exception as e:
              st.error(f"Failed to create retriever: {e}"); traceback.print_exc(); self.retriever = None


    def prepare_knowledge_base(self, force_rebuild: bool = False):
        """
        Prepares the knowledge base: loads/builds vector store and retriever.
        Uses LangChain components based on configuration.

        Args:
            force_rebuild (bool): If True, ignores existing index and rebuilds.
        """
        index_path = self.config.get("index_path", "data/index.pkl") # Get path from config
        rebuild_is_needed = False # Default to not rebuilding

        # 1. Determine if a rebuild is truly necessary
        if force_rebuild:
            st.warning("Forcing knowledge base rebuild...")
            rebuild_is_needed = True
        elif not os.path.exists(index_path):
            st.info(f"No existing index found at {index_path}. Building knowledge base...")
            rebuild_is_needed = True
        else:
            # Optional: Add more sophisticated checks here, e.g.,
            # - Compare hash of corpus file to a stored hash
            # - Compare critical config parameters (embedding model, chunking) stored in the index
            #   with the current config.
            # For now, we only rebuild if forced or index doesn't exist.
             print(f"Existing index found at {index_path}. Loading unless config requires rebuild (check not implemented).")
             # If you implement config checking, set rebuild_is_needed = True if critical config changed


        # --- CORE LOGIC BRANCH ---
        try:
            if rebuild_is_needed:
                # --- Rebuild Path ---
                st.write("Starting knowledge base build process...")

                # a) Check Corpus
                if not self.corpus:
                    # Attempt to load corpus if not already loaded (should be handled by initialize_rag_app ideally)
                    self.load_corpus(self.config.get("corpus_path"))
                    if not self.corpus:
                         st.error("Cannot build knowledge base: Corpus is empty or failed to load.")
                         return # Critical failure

                # b) Load Embedding Model (Crucial First Step)
                st.write("Loading embedding model...")
                if not self._load_or_get_embedding_model(): # Uses cached function via helper
                    # Error message already shown in helper
                    return # Critical failure

                # c) Perform Chunking (Using Refactored DocumentChunker)
                chunking_strategy = self.config.get("chunking_strategy", "recursive")
                st.write(f"Chunking documents using '{chunking_strategy}' strategy...")
                try:
                    # Prepare kwargs for chunker if needed (e.g., model name for token strategy)
                    chunker_kwargs = {}
                    if chunking_strategy == "token":
                         chunker_kwargs["embedding_model_name"] = self.config.get("embedding_model")

                    # Instantiate DocumentChunker - assumes its __init__ now uses Langchain splitters
                    chunker = DocumentChunker(
                        strategy=chunking_strategy,
                        chunk_size=self.config.get("chunk_size", 1000),
                        chunk_overlap=self.config.get("chunk_overlap", 200),
                        **chunker_kwargs
                    )
                    # chunk_corpus should return list of strings or Langchain Documents
                    # Assuming it returns list of strings based on paste-6.txt structure
                    self.chunked_docs = chunker.chunk_corpus(self.corpus) # list[str]
                    
                except Exception as e:
                    st.error(f"Error during chunking process: {e}")
                    traceback.print_exc()
                    return # Critical failure

                if not self.chunked_docs:
                    st.error("Chunking resulted in zero document chunks. Cannot build index.")
                    return # Critical failure
                st.write(f"Created {len(self.chunked_docs)} chunks.")

                # d) Create and Save Vector Store (Using LangChain FAISS)
                st.write(f"Creating vector store with '{self.config.get('embedding_model')}'...")
                try:
                    # Use from_texts if self.chunked_docs is list[str]
                    vectorstore = FAISS.from_texts(
                        texts=self.chunked_docs,
                        embedding=self.embedding_model_instance # Use the loaded embedding object
                    )
                    # If self.chunked_docs was changed to return Langchain Document objects:
                    # vectorstore = FAISS.from_documents(
                    #    documents=self.chunked_docs,
                    #    embedding=self.embedding_model_instance
                    # )

                    # Ensure directory exists before saving
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    vectorstore.save_local(index_path)
                    st.write(f"Vector store saved successfully to {index_path}")
                    self.vector_store = vectorstore # Store the newly created instance

                except Exception as e:
                    st.error(f"Error creating or saving vector store: {e}")
                    traceback.print_exc()
                    return # Critical failure

            else:
                # --- Load Existing Index Path ---
                st.write(f"Loading existing knowledge base from {index_path}...")

                # a) Load Embedding Model (Needed to load FAISS index)
                st.write("Loading embedding model...")
                if not self._load_or_get_embedding_model():
                    return # Critical failure

                # b) Load Vector Store from Disk
                self._load_vector_store() # This helper uses self.embedding_model_instance
                if self.vector_store is None:
                     st.error("Failed to load vector store. Consider rebuilding.")
                     # Optionally trigger a rebuild here? Or just fail.
                     return # Critical failure

            # --- 3. Create Retriever (Always run after vector store is ready) ---
            st.write("Creating retriever...")
            self._create_retriever() # Uses self.vector_store and config (MMR settings)
            if self.retriever is None:
                 st.error("Failed to create retriever.")
                 # Decide if this is critical - depends if queries can run without it
                 # return # Might be critical

            # --- Final Status ---
            st.success("Knowledge base is ready.")
            print("Knowledge base preparation/loading complete.")

        except Exception as e:
             # Catch-all for unexpected errors during the process
             st.error(f"An unexpected error occurred during knowledge base preparation: {e}")
             traceback.print_exc()
             # Reset states maybe?
             self.vector_store = None
             self.retriever = None

    
    def _load_vector_store(self):
        """Loads the vector store from disk."""
        index_path = self.config.get("index_path", "data/index.pkl")
        # --- Ensure embedding model is loaded first ---
        if not self.embedding_model_instance:
            st.warning("Embedding model not loaded before trying to load vector store.")
            # Optionally try loading it here, though prepare_kb should handle it
            if not self._load_or_get_embedding_model():
                st.error("Cannot load vector store: Embedding model failed.")
                self.vector_store = None
                return

        if os.path.exists(index_path) and self.embedding_model_instance:
            try:
                print(f"Loading vector store from {index_path}")
                self.vector_store = FAISS.load_local(
                        index_path,
                        self.embedding_model_instance, # <<< USE LOADED INSTANCE
                        allow_dangerous_deserialization=True
                )
                print("Vector store loaded successfully.")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                st.error(f"Failed to load vector store: {e}")
                self.vector_store = None
        else:
            print(f"Vector store not found at {index_path} or model not ready.")
            self.vector_store = None

    # src/app/rag_app.py (inside RAGApplication class)

    def process_query(self, query: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Processes a query using the configured RAG pipeline, including LangChain components.

        Args:
            query (str): The user's query.

        Returns:
            Tuple[str, List[Dict[str, any]]]: A tuple containing the generated answer
                                               and a list of context document dictionaries
                                               (e.g., {"text": ..., "metadata": ...}).
        """
        print(f"\n--- Processing Query: '{query}' ---")
        final_contexts_dicts = [] # Store results as list of dicts for compatibility
        answer = "Error: Failed to generate answer." # Default error answer

        try:
            # 1. Ensure Retriever and potentially other components are ready
            #    (prepare_knowledge_base should ideally be called *before* process_query)
            if self.retriever is None: # Check if retriever (vector search part) is ready
                 st.warning("Retriever not initialized. Attempting lazy initialization...")
                 print("Retriever not initialized. Attempting lazy initialization...")
                 # This might call prepare_knowledge_base which loads/builds index + creates retriever
                 self.prepare_knowledge_base(force_rebuild=False)
                 if self.retriever is None:
                      st.error("Retriever initialization failed. Cannot process query.")
                      return "Error: Knowledge base retriever is not ready.", []
            # Add similar check for BM25 index if needed for bm25/hybrid modes
            # if self.bm25_index is None and self.config.get("retrieval_method") in ["bm25", "hybrid"]:
            #     st.error("BM25 index not ready...") etc.

            # NOTE: Removed history append - should be handled by caller (streamlit_app.py)
            # self.conversation_history.append(query)

            # 2. Apply query expansion if enabled
            if self.config.get("query_expansion", False):
                expansion_method = self.config.get("expansion_method", "simple")
                try:
                     # Assuming QueryProcessor.expand_query is compatible
                     expanded_queries = QueryProcessor.expand_query(query, method=expansion_method)
                     print(f"Expanded query to {len(expanded_queries)} variations using '{expansion_method}'.")
                except Exception as qe_error:
                     print(f"Warning: Query expansion failed: {qe_error}. Using original query only.")
                     expanded_queries = [query]
            else:
                expanded_queries = [query]

            # 3. Retrieve relevant documents (Major Change: Using self.retriever + BM25)
            retrieval_method = self.config.get("retrieval_method", "hybrid")
            top_k = self.config.get("top_k", 5)
            print(f"Retrieving top-{top_k} documents using '{retrieval_method}' method...")

            # --- Combined results from all expanded queries before deduplication/ranking ---
            combined_retrieved_docs = [] # List to hold Langchain Document objects initially

            for i, exp_query in enumerate(expanded_queries):
                print(f"  Processing expanded query {i+1}/{len(expanded_queries)}: '{exp_query}'")
                vector_docs = []
                bm25_docs_info = [] # Store bm25 results if needed (e.g., list of {"doc": Document, "score": score})

                # a) Vector Store Retrieval (if method is vector or hybrid)
                if retrieval_method in ["vector", "hybrid"]:
                    try:
                        # self.retriever already configured with top_k, MMR settings
                        # It returns Langchain Document objects
                        vector_docs = self.retriever.get_relevant_documents(exp_query)
                        print(f"    Vector search returned {len(vector_docs)} docs.")
                    except Exception as vs_error:
                         print(f"    Error during vector search for '{exp_query}': {vs_error}")
                         st.warning(f"Vector search failed for sub-query.")

                # b) BM25 Retrieval (if method is bm25 or hybrid)
                if retrieval_method in ["bm25", "hybrid"]:
                    try:
                         # Assumes _perform_bm25_search returns list of tuples (Document, score) or similar
                         # You need to implement this method and BM25 indexing
                         bm25_results = self._perform_bm25_search(exp_query, k=top_k * 2) # Fetch more for hybrid?
                         # Store results with score for potential fusion
                         bm25_docs_info = [{"doc": doc, "score": score} for doc, score in bm25_results]
                         print(f"    BM25 search returned {len(bm25_docs_info)} docs.")
                    except Exception as bm25_error:
                         print(f"    Error during BM25 search for '{exp_query}': {bm25_error}")
                         st.warning(f"BM25 search failed for sub-query.")

                # c) Combine/Select results for this expanded query
                if retrieval_method == "vector":
                     combined_retrieved_docs.extend(vector_docs)
                elif retrieval_method == "bm25":
                     # Add only the Document objects from BM25 results
                     combined_retrieved_docs.extend([info["doc"] for info in bm25_docs_info])
                elif retrieval_method == "hybrid":
                     # --- Hybrid Fusion Logic ---
                     # This needs proper implementation (e.g., Reciprocal Rank Fusion)
                     # Simple Placeholder: Combine, deduplicate, take top K based on original scores? Very basic.
                     print("    Applying basic hybrid fusion (combine, deduplicate). Needs refinement.")
                     temp_combined = {} # Use dict for deduplication based on content or ID
                     # Add vector docs (assuming retriever already handled MMR/similarity ranking)
                     for doc in vector_docs: temp_combined[doc.page_content] = doc # Or use a unique ID if available
                     # Add BM25 docs
                     for info in bm25_docs_info: temp_combined[info["doc"].page_content] = info["doc"]
                     # Add the unique documents from this sub-query
                     combined_retrieved_docs.extend(list(temp_combined.values()))
                     # --- End Hybrid Fusion Logic ---

            # 4. Deduplicate results across all expanded queries
            final_unique_docs = {} # Dict for deduplication
            for doc in combined_retrieved_docs:
                 # Use page content as key for simplicity, or a unique ID from metadata if available
                 doc_key = doc.page_content
                 if doc_key not in final_unique_docs:
                      final_unique_docs[doc_key] = doc

            # Now we have unique Langchain Document objects
            documents_to_process = list(final_unique_docs.values())
            print(f"Retrieved {len(documents_to_process)} unique documents total before reranking/final top_k.")

            # Ensure we don't exceed top_k if no reranking is done later
            # Note: Reranking might change the final selection size
            if not self.config.get("use_reranking", False):
                 documents_to_process = documents_to_process[:top_k]

            # 5. Apply Reranking (if enabled)
            if self.config.get("use_reranking", False) and documents_to_process:
                reranking_method = self.config.get("reranking_method", "cross_encoder")
                print(f"Reranking {len(documents_to_process)} documents using '{reranking_method}' method...")
                try:
                    # Assuming RerankerModule needs config and can handle list of Langchain Documents
                    reranker = RerankerModule(self.config)
                    # Assuming reranker.rerank_documents returns list of Langchain Documents sorted
                    documents_to_process = reranker.rerank_documents(query, documents_to_process)
                    print(f"Reranking complete. Selecting top {top_k} docs.")
                    # Select final top_k *after* reranking
                    documents_to_process = documents_to_process[:top_k]
                except Exception as rerank_error:
                    print(f"Error during reranking: {rerank_error}. Using documents before reranking.")
                    st.warning(f"Reranking failed: {rerank_error}")
                    # Take top_k from the pre-reranking list if reranking failed
                    documents_to_process = documents_to_process[:top_k]

            print(f"Final {len(documents_to_process)} documents selected for context.")

            # 6. Prepare Contexts for Generation and Return Value
            # Convert final LangChain Document objects back to dicts for compatibility
            final_contexts_dicts = []
            context_texts = []
            for doc in documents_to_process:
                 context_dict = {
                      "text": doc.page_content,
                      # Reconstruct metadata - adapt based on what metadata your Document objects have
                      "metadata": doc.metadata,
                      # Add title if available in metadata
                      "title": doc.metadata.get("source", "Source Unknown")
                      # Add original chunk ID if needed and available
                      # "chunk_id": doc.metadata.get("chunk_index", -1)
                 }
                 final_contexts_dicts.append(context_dict)
                 context_texts.append(doc.page_content)

            context_str = "\n\n".join(context_texts)

            # 7. Generate Answer
            prompt_template = self.config.get("prompt_template", "Context: {context}\nQuery: {query}\nAnswer:")
            prompt = prompt_template.format(context=context_str, query=query)

            # Get response mode set by Streamlit UI (passed via initialize_rag_app or similar)
            response_mode = getattr(self, "response_mode", "extractive") # Default if not set
            temperature = self.config.get("temperature", 0.5) # Use temp from config

            print(f"Generating answer using mode: '{response_mode}', Temp: {temperature}")
            if response_mode == "llm":
                # Pass relevant config (like temperature) if _llm_answer needs it
                answer = self._llm_answer(query, context_texts, temperature=temperature)
            else: # extractive
                answer = self._extractive_answer(query, context_texts)

            print("Answer generation complete.")
            return answer, final_contexts_dicts # Return dicts

        except Exception as e:
             st.error(f"An unexpected error occurred in process_query: {e}")
             print(f"FATAL ERROR in process_query: {e}")
             traceback.print_exc()
             return answer, final_contexts_dicts # Return default error answer and any contexts gathered

    # --- Placeholder for BM25 Search ---
    def _perform_bm25_search(self, query, k=5):
        # Requires self.bm25_index and a mapping back to original docs/chunks
        print("BM25 search logic needs implementation.")
        # Example structure:
        # if hasattr(self, 'bm25_index') and self.bm25_index:
        #     tokenized_query = query.lower().split(" ")
        #     # Assuming bm25_index.get_top_n returns scores for documents based on corpus used for indexing
        #     scores = self.bm25_index.get_scores(tokenized_query)
        #     # Need to map these scores back to Langchain Document objects
        #     # This requires storing/mapping original docs during BM25 index creation
        #     # top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        #     # results = [(self.get_document_by_index(i), scores[i]) for i in top_n_indices] # Implement get_document_by_index
        #     return [] # Placeholder for actual results
        # else:
        #     print("BM25 index not available.")
        #     return []
        return [] # Return empty list

    # --- Placeholder for Answer Generation Methods ---
    # def _llm_answer(self, query: str, context_texts: List[str], temperature: float = 0.5):
    #     # Implement LLM call using self.config, query, context_texts, temperature
    #     print(f"LLM generation needs implementation. Using placeholder. Temp={temperature}")
    #     context_str = "\n\n".join(context_texts)
    #     return f"Placeholder LLM Answer for '{query}' based on {len(context_texts)} contexts."

    def _extractive_answer(self, query, context_texts):
        # Implement simple extractive logic (e.g., return first context)
        print("Extractive answer needs implementation. Using placeholder.")
        if context_texts:
             return f"Placeholder Extractive Answer: Best match might be '{context_texts[0][:150]}...'"
        else:
             return "Placeholder Extractive Answer: No context found."



    def _llm_answer(self, query: str, contexts: List[str], temperature: float = 0.5) -> str:
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
            # temperature = getattr(self, "temperature", 0.3)

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