"""
Chunking Experiment

This script runs experiments to compare different document chunking strategies:
- Fixed-size chunking with different sizes and overlap
- Paragraph-based chunking
- Semantic unit chunking

It evaluates how these strategies affect retrieval performance using various metrics.
"""

import os
import sys
import pickle
import time
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
from src.components.advanced_chunking import AdvancedSemanticChunker

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker

def run_chunking_experiment(
    corpus_file: str = "data/sample_corpus.pkl",
    queries_file: str = "data/sample_queries.pkl",
    sample_size: int = 50,  # Number of documents to use
    num_queries: int = 5,  # Number of queries to evaluate
    embedding_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "results"
):
    """
    Run experiments to compare different chunking strategies (Based on paste-4.txt + KeyError Fix + COMPLETE Exception Logging)
    """
    # --- Keep os.makedirs, data loading, sampling, tracker init, config logging exactly as in the previous response ---
    os.makedirs(output_dir, exist_ok=True)
    try: # Load corpus
        with open(corpus_file, "rb") as f: corpus_data = pickle.load(f)
        print(f"Loaded {len(corpus_data)} documents from {corpus_file}")
    except FileNotFoundError: print(f"Corpus file not found: {corpus_file}"); return None
    except Exception as e: print(f"Error loading corpus file {corpus_file}: {e}"); return None
    try: # Load queries
        with open(queries_file, "rb") as f: query_data = pickle.load(f)
        print(f"Loaded {len(query_data)} queries from {queries_file}")
    except FileNotFoundError: print(f"Queries file not found: {queries_file}"); return None
    except Exception as e: print(f"Error loading queries file {queries_file}: {e}"); return None

    if sample_size < len(corpus_data): corpus_sample = corpus_data[:sample_size]
    else: corpus_sample = corpus_data
    if num_queries < len(query_data): queries_sample = query_data[:num_queries]
    else: queries_sample = query_data
    print(f"Running chunking experiments with {len(corpus_sample)} documents and {len(queries_sample)} queries")

    try: # Init tracker
        tracker = ExperimentTracker("chunking_experiment")
    except Exception as e: print(f"Error initializing ExperimentTracker: {e}"); return None

    tracker.log_experiment_config({ # Log config
        "corpus_file": os.path.basename(corpus_file),
        "sample_size": len(corpus_sample),
        "num_queries": len(queries_sample),
        "embedding_model": embedding_model,
        "output_dir_arg": output_dir
    })

    # Define chunking configurations (Keep exactly as previous response)
    chunking_configs = [
        {"strategy": "fixed", "size": 64, "overlap": 0}, {"strategy": "fixed", "size": 64, "overlap": 16},
        {"strategy": "fixed", "size": 128, "overlap": 0}, {"strategy": "fixed", "size": 128, "overlap": 32},
        {"strategy": "fixed", "size": 256, "overlap": 0}, {"strategy": "fixed", "size": 256, "overlap": 64},
        {"strategy": "paragraph"}, {"strategy": "semantic"},
        {"strategy": "semantic_segmentation", "chunker": AdvancedSemanticChunker(), "description": "Advanced semantic segmentation"},
        {"strategy": "topic_modeling", "chunker": AdvancedSemanticChunker(), "description": "Topic-based chunking"},
        {"strategy": "sliding_window_coherence", "chunker": AdvancedSemanticChunker(), "description": "Sliding window with coherence"}
    ]
    # --- End Setup ---


    # Process each chunking configuration
    for config in chunking_configs:
        print(f"\n--- Evaluating Chunking Config: {config} ---")
        chunked_docs = [] # Reset
        start_chunk_time = time.time()

        try: # Outer try-except for chunking and validation stages
            # --- Apply chunking strategy (Keep if/elif block exactly as previous response) ---
            if config["strategy"] == "semantic_segmentation":
                chunker = config.get("chunker", AdvancedSemanticChunker())
                chunked_docs = chunker.chunk_by_semantic_segmentation(corpus_sample, target_chunk_size=200, min_chunk_size=100)
            elif config["strategy"] == "topic_modeling":
                chunker = config.get("chunker", AdvancedSemanticChunker())
                chunked_docs = chunker.chunk_by_topic_modeling(corpus_sample)
            elif config["strategy"] == "sliding_window_coherence":
                chunker = config.get("chunker", AdvancedSemanticChunker())
                chunked_docs = chunker.chunk_by_sliding_window_with_coherence(corpus_sample)
            elif config["strategy"] == "fixed":
                chunked_docs = DocumentChunker.chunk_by_fixed_size(corpus_sample, chunk_size=config["size"], overlap=config["overlap"])
            elif config["strategy"] == "paragraph":
                chunked_docs = DocumentChunker.chunk_by_paragraph(corpus_sample)
            elif config["strategy"] == "semantic":
                 chunked_docs = DocumentChunker.chunk_by_semantic_units(corpus_sample)
            else:
                print(f"Warning: Unknown chunking strategy: {config['strategy']}. Skipping.")
                continue
            # --- End Chunking Logic ---

            chunk_time = time.time() - start_chunk_time
            print(f"Chunking completed in {chunk_time:.2f} seconds.")
            print(f"Created {len(chunked_docs)} chunks.")

            if not chunked_docs:
                 print("Warning: Chunking produced no documents. Skipping evaluation.")
                 continue

            # --- Validation Loop for KeyError: 'chunk_id' (Keep exactly as previous response) ---
            validated_chunks = []
            for i, doc in enumerate(chunked_docs):
                if not isinstance(doc, dict):
                    print(f"Warning: Chunk {i} type {type(doc)}, not dict. Wrapping.")
                    doc = {"text": str(doc)}
                if "text" not in doc:
                    print(f"Warning: Chunk {i} missing 'text'. Skipping.")
                    continue
                if "chunk_id" not in doc or not doc["chunk_id"]:
                    size_str = config.get('size', 'na'); overlap_str = config.get('overlap', 'na')
                    strat_name = config.get('description', config['strategy']).replace(" ", "_")
                    generated_id = f"{strat_name}_{size_str}_{overlap_str}_{i}"
                    doc["chunk_id"] = generated_id
                validated_chunks.append(doc)
            chunked_docs = validated_chunks
            if not chunked_docs:
                 print("Warning: No valid chunks after validation. Skipping evaluation.")
                 continue
            # --- End Validation Loop ---

            # --- Access chunk texts and IDs (Keep try/except block exactly as previous response) ---
            try:
                 chunk_texts = [doc["text"] for doc in chunked_docs]
                 doc_ids = [doc["chunk_id"] for doc in chunked_docs]
            except KeyError as ke:
                 print(f"FATAL INTERNAL ERROR: KeyError '{ke}' accessing chunk keys AFTER validation. Config: {config}")
                 traceback.print_exc(); continue
            # --- End Access ---

            # --- Inner try-except for Embeddings, Evaluation, Logging ---
            try:
                # --- Generate embeddings (Keep static call exactly as previous response) ---
                start_time = time.time()
                chunk_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
                    chunk_texts, model_name=embedding_model
                )
                embed_time = time.time() - start_time
                print(f"Generated embeddings in {embed_time:.2f} seconds")

                # --- Evaluate retrieval (Keep calls exactly as previous response) ---
                vector_search_metrics = evaluate_retrieval(queries_sample, chunk_embeddings, chunked_docs, doc_ids, "vector", embedding_model)
                bm25_search_metrics = evaluate_retrieval(queries_sample, chunk_embeddings, chunked_docs, doc_ids, "bm25", embedding_model)
                hybrid_search_metrics = evaluate_retrieval(queries_sample, chunk_embeddings, chunked_docs, doc_ids, "hybrid", embedding_model)

                # --- Log Results (Keep sanitize function and logging blocks exactly as previous response) ---
                def sanitize_dict_for_json(d):
                    if not isinstance(d, dict): return d
                    return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}

                base_log_info = {
                    "chunking_strategy": config.get("description", config["strategy"]),
                    "chunk_size": config.get("size", "auto"),
                    "chunk_overlap": config.get("overlap", "auto"),
                    "num_chunks": len(chunked_docs),
                }

                if vector_search_metrics:
                     log_data = {**base_log_info, "retrieval_method": "vector", "embedding_time_sec": embed_time, **sanitize_dict_for_json({f"metric_{k}": v for k, v in vector_search_metrics.items()})}
                     tracker.log_iteration(log_data)
                if bm25_search_metrics:
                     log_data = {**base_log_info, "retrieval_method": "bm25", "embedding_time_sec": 0, **sanitize_dict_for_json({f"metric_{k}": v for k, v in bm25_search_metrics.items()})}
                     tracker.log_iteration(log_data)
                if hybrid_search_metrics:
                     log_data = {**base_log_info, "retrieval_method": "hybrid", "embedding_time_sec": embed_time, **sanitize_dict_for_json({f"metric_{k}": v for k, v in hybrid_search_metrics.items()})}
                     tracker.log_iteration(log_data)
                # --- End Logging ---

            # --- CATCH AttributeError during inner try block ---
            except AttributeError as ae:
                print(f"\n\nCRITICAL ERROR during embedding/evaluation ({config}): {ae}")
                print("This likely means EmbeddingProvider class definition in rag_components.py is the refactored version.")
                print("You MUST replace it with the ORIGINAL static version provided.\n")
                traceback.print_exc() # Show where the error occurred
                # --- COMPLETE failure log for AttributeError ---
                failure_log_data = {
                    "chunking_strategy": config.get("description", config["strategy"]),
                    "chunk_size": config.get("size", "auto"),
                    "chunk_overlap": config.get("overlap", "auto"),
                    "num_chunks": len(chunked_docs), # Log how many chunks were created before failure
                    "status": "Failed",
                    "error": f"AttributeError: {ae} - Check EmbeddingProvider class definition"
                }
                tracker.log_iteration(failure_log_data)
                # --- End COMPLETE failure log ---
                print("Continuing to next config if possible...")
                continue # Try next config

            # --- CATCH other exceptions during inner try block ---
            except Exception as e:
                print(f"Error during embedding/evaluation/logging for config {config}: {e}")
                traceback.print_exc()
                # --- COMPLETE failure log for other inner errors ---
                failure_log_data = {
                    "chunking_strategy": config.get("description", config["strategy"]),
                    "chunk_size": config.get("size", "auto"),
                    "chunk_overlap": config.get("overlap", "auto"),
                    "num_chunks": len(chunked_docs), # Log num chunks if available
                    "status": "Failed",
                    "error": str(e) # Convert general error to string
                }
                tracker.log_iteration(failure_log_data)
                # --- End COMPLETE failure log ---
                print("Continuing to next config if possible...")
                continue # Try next config
            # --- End Inner Try-Except ---

        # --- CATCH exceptions during outer try block (chunking/validation) ---
        except Exception as e:
            print(f"\nFATAL ERROR processing chunking config {config}: {e}")
            traceback.print_exc()
            # --- COMPLETE failure log for outer errors ---
            failure_log_data = {
                "chunking_strategy": config.get("description", config["strategy"]),
                "chunk_size": config.get("size", "auto"),
                "chunk_overlap": config.get("overlap", "auto"),
                # num_chunks might not be available if error was during chunking
                "num_chunks": len(chunked_docs) if chunked_docs else 0,
                "status": "Failed",
                "error": str(e) # Convert general error to string
            }
            tracker.log_iteration(failure_log_data)
            # --- End COMPLETE failure log ---
            print("Attempting to continue with the next configuration...")
            continue # Try next config
        # --- End Outer Try-Except ---
    # --- End Main Loop ---

    # --- Generate Report (Keep exactly as previous response) ---
    print("\nGenerating experiment report...")
    try:
        report_path = tracker.generate_report()
        print(f"Report generated at {report_path}")
        return report_path
    except Exception as e: print(f"Error generating report: {e}"); return None
    # --- End Report ---


def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    method: str = "vector",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Dict[str, float]:
    """
    Evaluate retrieval performance
    
    Args:
        queries: List of query objects with 'question' and 'answer'
        chunk_embeddings: Embeddings of all chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs corresponding to chunk_embeddings
        method: Retrieval method ('vector', 'bm25', or 'hybrid')
        embedding_model: Name of embedding model
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize metrics
    precision_at_1_sum = 0
    precision_at_3_sum = 0
    precision_at_5_sum = 0
    recall_at_3_sum = 0
    recall_at_5_sum = 0
    mrr_sum = 0
    
    # For each query, calculate retrieval metrics
    for query_obj in tqdm(queries, desc=f"Evaluating {method} retrieval"):
        query = query_obj["question"]
        expected_answer = query_obj.get("answer", "")
        
        # Find relevant chunks (simplified approach for this experiment)
        # In a real-world scenario, you'd have ground truth relevance judgments
        # Here we'll use a simple word overlap heuristic
        relevant_chunks = []
        
        query_words = set(query.lower().split())
        answer_words = set(expected_answer.lower().split())
        combined_words = query_words.union(answer_words)
        
        for i, doc in enumerate(chunked_docs):
            doc_words = set(doc["text"].lower().split())
            # Calculate word overlap
            overlap = len(combined_words.intersection(doc_words))
            if overlap >= min(3, len(combined_words)):
                relevant_chunks.append(doc_ids[i])
        
        # If no relevant chunks found, use chunks with highest word overlap
        if not relevant_chunks:
            overlaps = []
            for i, doc in enumerate(chunked_docs):
                doc_words = set(doc["text"].lower().split())
                overlap = len(combined_words.intersection(doc_words))
                overlaps.append((doc_ids[i], overlap))
            
            # Sort by overlap and take top 2
            overlaps.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = [doc_id for doc_id, _ in overlaps[:2]]
        
        # Retrieve documents using specified method
        if method == "vector":
            # Embed query
            query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                [query], model_name=embedding_model
            )[0]
            
            # Retrieve
            search_results = RetrievalMethods.vector_search(
                query_embedding, chunk_embeddings, doc_ids, top_k=10
            )
            
        elif method == "bm25":
            search_results = RetrievalMethods.bm25_search(
                query, chunked_docs, top_k=10
            )
            
        elif method == "hybrid":
            # Embed query
            query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                [query], model_name=embedding_model
            )[0]
            
            # Retrieve
            search_results = RetrievalMethods.hybrid_search(
                query, query_embedding, chunked_docs, chunk_embeddings, top_k=10
            )
            
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Get retrieved IDs
        retrieved_ids = [doc_id for doc_id, _ in search_results]
        
        # Calculate metrics
        precision_at_1 = RAGEvaluator.precision_at_k(relevant_chunks, retrieved_ids, k=1)
        precision_at_3 = RAGEvaluator.precision_at_k(relevant_chunks, retrieved_ids, k=3)
        precision_at_5 = RAGEvaluator.precision_at_k(relevant_chunks, retrieved_ids, k=5)
        recall_at_3 = RAGEvaluator.recall_at_k(relevant_chunks, retrieved_ids, k=3)
        recall_at_5 = RAGEvaluator.recall_at_k(relevant_chunks, retrieved_ids, k=5)
        mrr = RAGEvaluator.mean_reciprocal_rank(relevant_chunks, retrieved_ids)
        
        # Add to sums
        precision_at_1_sum += precision_at_1
        precision_at_3_sum += precision_at_3
        precision_at_5_sum += precision_at_5
        recall_at_3_sum += recall_at_3
        recall_at_5_sum += recall_at_5
        mrr_sum += mrr
    
    # Calculate averages
    num_queries = len(queries)
    metrics = {
        "precision_at_1": precision_at_1_sum / num_queries,
        "precision_at_3": precision_at_3_sum / num_queries,
        "precision_at_5": precision_at_5_sum / num_queries,
        "recall_at_3": recall_at_3_sum / num_queries,
        "recall_at_5": recall_at_5_sum / num_queries,
        "mrr": mrr_sum / num_queries
    }
    
    print(f"{method.capitalize()} search metrics:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")
        
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run chunking experiments")
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.pkl", help="Path to corpus file")
    parser.add_argument("--queries", type=str, default="data/sample_queries.pkl", help="Path to queries file")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of documents to use")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    run_chunking_experiment(
        corpus_file=args.corpus,
        queries_file=args.queries,
        sample_size=args.sample_size,
        num_queries=args.num_queries,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir
    )