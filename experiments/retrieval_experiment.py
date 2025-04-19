"""
Retrieval Experiment

This script runs experiments to compare different retrieval methods:
- Vector search
- BM25 keyword search
- Hybrid search with different weights
- Query expansion techniques

It evaluates retrieval performance across different metrics and configurations.
"""

import os
import sys
import pickle
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods, QueryProcessor
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker


# --- run_retrieval_experiment Function (With Type Conversion Fix) ---
def run_retrieval_experiment(
    corpus_file: str = "data/sample_corpus.pkl",
    queries_file: str = "data/sample_queries.pkl",
    sample_size: int = 50,
    num_queries: int = 5,
    chunk_strategy: str = "fixed",
    chunk_size: int = 128,      # Default is int
    chunk_overlap: int = 0,       # Default is int
    embedding_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "results"
):
    """
    Run experiments to compare different retrieval methods
    """
    # Ensure output directory exists (Keep as is)
    os.makedirs(output_dir, exist_ok=True)

    # Load data (Keep as is)
    try:
        with open(corpus_file, "rb") as f: corpus_data = pickle.load(f)
    except FileNotFoundError: print(f"Corpus file not found: {corpus_file}"); return None
    try:
        with open(queries_file, "rb") as f: query_data = pickle.load(f)
    except FileNotFoundError: print(f"Queries file not found: {queries_file}"); return None

    # Sample data (Keep as is)
    if sample_size < len(corpus_data): corpus_sample = corpus_data[:sample_size]
    else: corpus_sample = corpus_data
    if num_queries < len(query_data): queries_sample = query_data[:num_queries]
    else: queries_sample = query_data
    print(f"Running retrieval experiments with {len(corpus_sample)} documents and {len(queries_sample)} queries")

    # Create experiment tracker (Keep as is)
    tracker = ExperimentTracker("retrieval_experiment")

    # Load best config (Keep as is)
    best_config = load_best_configuration(output_dir)

    # --- Assign and FIX TYPES for chunk_size and chunk_overlap ---
    # Start with defaults (which are ints)
    current_chunk_strategy = chunk_strategy
    current_chunk_size = chunk_size
    current_chunk_overlap = chunk_overlap
    current_embedding_model = embedding_model

    # Overwrite with loaded config if available
    if best_config:
        current_chunk_strategy = best_config.get("chunking_strategy", current_chunk_strategy)
        # Get potential string/object values from loaded config
        loaded_chunk_size = best_config.get("chunk_size")
        loaded_chunk_overlap = best_config.get("chunk_overlap")
        current_embedding_model = best_config.get("embedding_model", current_embedding_model)

        # --- ADDED TYPE CONVERSION WITH ERROR HANDLING ---
        if loaded_chunk_size is not None:
            try:
                current_chunk_size = int(loaded_chunk_size) # Convert to int
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loaded chunk_size '{loaded_chunk_size}' to int. Using default: {chunk_size}")
                current_chunk_size = chunk_size # Fallback to default int
        else:
             current_chunk_size = chunk_size # Use default if not found in config

        if loaded_chunk_overlap is not None:
            try:
                current_chunk_overlap = int(loaded_chunk_overlap) # Convert to int
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loaded chunk_overlap '{loaded_chunk_overlap}' to int. Using default: {chunk_overlap}")
                current_chunk_overlap = chunk_overlap # Fallback to default int
        else:
             current_chunk_overlap = chunk_overlap # Use default if not found in config
        # --- END TYPE CONVERSION ---

    # Use the correctly typed variables from now on
    final_chunk_strategy = current_chunk_strategy
    final_chunk_size = current_chunk_size
    final_chunk_overlap = current_chunk_overlap
    final_embedding_model = current_embedding_model

    # Print the final values being used (and their types)
    print(f"Using chunking strategy: {final_chunk_strategy} (type: {type(final_chunk_strategy).__name__})")
    print(f"Using chunk size: {final_chunk_size} (type: {type(final_chunk_size).__name__})") # Should be int
    print(f"Using chunk overlap: {final_chunk_overlap} (type: {type(final_chunk_overlap).__name__})") # Should be int
    print(f"Using embedding model: {final_embedding_model} (type: {type(final_embedding_model).__name__})")


    # Log experiment configuration (use final values)
    tracker.log_experiment_config({
        "dataset": os.path.basename(corpus_file),
        "sample_size": len(corpus_sample),
        "num_queries": len(queries_sample),
        "chunk_strategy": final_chunk_strategy,
        "chunk_size": final_chunk_size,
        "chunk_overlap": final_chunk_overlap,
        "embedding_model": final_embedding_model
    })

    # Apply chunking strategy (use final values)
    # --- This call should now receive integers ---
    try:
        if final_chunk_strategy == "fixed":
            chunked_docs = DocumentChunker.chunk_by_fixed_size(
                corpus_sample, chunk_size=final_chunk_size, overlap=final_chunk_overlap
            )
        elif final_chunk_strategy == "paragraph":
            chunked_docs = DocumentChunker.chunk_by_paragraph(corpus_sample)
        elif final_chunk_strategy == "semantic":
            chunked_docs = DocumentChunker.chunk_by_semantic_units(corpus_sample)
        else:
            raise ValueError(f"Unknown chunking strategy: {final_chunk_strategy}")
    except TypeError as te:
         # Catch the specific error again JUST IN CASE conversion failed somehow
         print(f"\nCRITICAL ERROR during chunking: {te}")
         print(f"Values passed: size={final_chunk_size} (type {type(final_chunk_size).__name__}), overlap={final_chunk_overlap} (type {type(final_chunk_overlap).__name__})")
         raise te # Re-raise the error if it still happens
    except Exception as e:
         print(f"Error during chunking: {e}")
         return None # Exit if chunking fails

    print(f"Created {len(chunked_docs)} chunks")

    # --- Extract texts and IDs (Add validation loop for safety) ---
    validated_chunks = []
    for i, doc in enumerate(chunked_docs):
        if not isinstance(doc, dict): doc = {"text": str(doc)}
        if "text" not in doc: print(f"Warning: Chunk {i} missing 'text'. Skipping."); continue
        if "chunk_id" not in doc or not doc["chunk_id"]:
            # Generate ID (consider using final_chunk_strategy etc. for uniqueness)
            doc["chunk_id"] = f"{final_chunk_strategy}_{final_chunk_size}_{final_chunk_overlap}_{i}"
        validated_chunks.append(doc)
    chunked_docs = validated_chunks # Use validated list
    if not chunked_docs: print("Warning: No valid chunks after validation."); return None

    try:
        chunk_texts = [doc["text"] for doc in chunked_docs]
        doc_ids = [doc["chunk_id"] for doc in chunked_docs]
    except KeyError as ke:
        print(f"FATAL ERROR: KeyError '{ke}' accessing chunk keys AFTER validation.")
        return None
    # --- End Text/ID Extraction ---


    # Generate embeddings (use final model)
    print(f"Generating embeddings using {final_embedding_model}...")
    try:
        # This uses the ORIGINAL static EmbeddingProvider method
        chunk_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
            chunk_texts, model_name=final_embedding_model
        )
    except AttributeError as ae:
        print(f"\n\nCRITICAL ERROR generating embeddings: {ae}")
        print("Check EmbeddingProvider class in rag_components.py - requires ORIGINAL static version.")
        return None
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

    # Define retrieval methods to test (Keep as is)
    retrieval_methods = [
        {"name": "vector", "type": "vector"}, {"name": "bm25", "type": "bm25"},
        {"name": "hybrid_0.3", "type": "hybrid", "alpha": 0.3}, {"name": "hybrid_0.5", "type": "hybrid", "alpha": 0.5}, {"name": "hybrid_0.7", "type": "hybrid", "alpha": 0.7},
        {"name": "vector_expanded", "type": "vector", "expansion": "simple"}, {"name": "bm25_expanded", "type": "bm25", "expansion": "simple"}, {"name": "hybrid_expanded", "type": "hybrid", "alpha": 0.5, "expansion": "simple"}
    ]

    # Test each retrieval method (Keep loop structure)
    for method_config in retrieval_methods:
        method_name = method_config["name"]
        print(f"\nEvaluating retrieval method: {method_name}")
        try:
            start_time = time.time()
            # Pass final_embedding_model here
            retrieval_metrics = evaluate_retrieval_method(
                queries_sample, chunk_embeddings, chunked_docs, doc_ids,
                method_config, final_embedding_model
            )
            retrieval_time = time.time() - start_time

            # Log results (Keep structure, ensure sanitize function exists if needed)
            def sanitize_dict_for_json(d): # Define helper if not global
                 if not isinstance(d, dict): return d
                 return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}

            result_data = {
                "retrieval_method": method_name,
                "retrieval_type": method_config["type"],
                "query_expansion": method_config.get("expansion", "none"),
                "retrieval_time_sec": retrieval_time,
                "queries_per_second": len(queries_sample) / retrieval_time if retrieval_time > 0 else 0,
                # Use sanitize and .get for safety
                **sanitize_dict_for_json({f"metric_{k}": v for k, v in retrieval_metrics.items()})
            }
            if method_config["type"] == "hybrid": result_data["alpha"] = method_config.get("alpha", 0.5)
            tracker.log_iteration(result_data)

        except AttributeError as ae:
            # Catch specific error if EmbeddingProvider definition is wrong during eval
            print(f"\n\nCRITICAL ERROR evaluating method {method_name}: {ae}")
            print("Check EmbeddingProvider class definition - requires ORIGINAL static version.")
            failure_log_data = {"retrieval_method": method_name, "status": "Failed", "error": f"AttributeError: {ae}"}
            tracker.log_iteration(failure_log_data)
            continue # Continue to next method

        except Exception as e:
            print(f"Error evaluating method {method_name}: {e}")
            import traceback; traceback.print_exc()
            failure_log_data = {"retrieval_method": method_name, "status": "Failed", "error": str(e)}
            tracker.log_iteration(failure_log_data)
            continue # Continue to next method

    # Generate a report (Keep as is)
    print("Generating experiment report...")
    report_path = tracker.generate_report()
    print(f"Report generated at {report_path}")

    return report_path

def evaluate_retrieval_method(
    queries: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    method_config: Dict[str, Any],
    embedding_model: str
) -> Dict[str, float]:
    """
    Evaluate a specific retrieval method
    
    Args:
        queries: List of query objects with 'question' and 'answer'
        chunk_embeddings: Embeddings of all chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs corresponding to chunk_embeddings
        method_config: Configuration for the retrieval method
        embedding_model: Name of the embedding model
        
    Returns:
        Dictionary of evaluation metrics
    """
    method_type = method_config["type"]
    expansion_type = method_config.get("expansion", None)
    
    # Initialize metrics
    precision_at_1_sum = 0
    precision_at_3_sum = 0
    precision_at_5_sum = 0
    recall_at_3_sum = 0
    recall_at_5_sum = 0
    mrr_sum = 0
    
    # For each query, calculate retrieval metrics
    for query_obj in tqdm(queries, desc=f"Evaluating {method_config['name']}"):
        query = query_obj["question"]
        expected_answer = query_obj.get("answer", "")
        
        # Find relevant chunks (simplified approach)
        relevant_chunks = []
        
        query_words = set(query.lower().split())
        answer_words = set(expected_answer.lower().split())
        combined_words = query_words.union(answer_words)
        
        for i, doc_id in enumerate(doc_ids):
            # Get document text
            doc_idx = next((j for j, doc in enumerate(chunked_docs) if doc["chunk_id"] == doc_id), None)
            if doc_idx is None:
                continue
                
            doc_text = chunked_docs[doc_idx]["text"]
            doc_words = set(doc_text.lower().split())
            
            # Calculate word overlap
            overlap = len(combined_words.intersection(doc_words))
            if overlap >= min(3, len(combined_words)):
                relevant_chunks.append(doc_id)
        
        # If no relevant chunks found, use chunks with highest word overlap
        if not relevant_chunks:
            overlaps = []
            for i, doc_id in enumerate(doc_ids):
                doc_idx = next((j for j, doc in enumerate(chunked_docs) if doc["chunk_id"] == doc_id), None)
                if doc_idx is None:
                    continue
                    
                doc_text = chunked_docs[doc_idx]["text"]
                doc_words = set(doc_text.lower().split())
                overlap = len(combined_words.intersection(doc_words))
                overlaps.append((doc_id, overlap))
            
            # Sort by overlap and take top 2
            overlaps.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = [doc_id for doc_id, _ in overlaps[:2]]
        
        # Apply query expansion if specified
        all_retrieved_ids = []
        
        if expansion_type:
            expanded_queries = QueryProcessor.expand_query(query, method=expansion_type)
            
            # Retrieve for each expanded query
            for exp_query in expanded_queries:
                retrieved_for_query = retrieve_with_method(
                    exp_query, chunk_embeddings, chunked_docs, doc_ids,
                    method_type, method_config, embedding_model
                )
                
                all_retrieved_ids.extend([doc_id for doc_id, _ in retrieved_for_query])
                
            # Remove duplicates while preserving order (using dict.fromkeys trick)
            retrieved_ids = list(dict.fromkeys(all_retrieved_ids))[:10]  # Keep top 10
            
        else:
            # Regular retrieval without expansion
            search_results = retrieve_with_method(
                query, chunk_embeddings, chunked_docs, doc_ids,
                method_type, method_config, embedding_model
            )
            
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
    
    print(f"Retrieval metrics for {method_config['name']}:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")
        
    return metrics

def retrieve_with_method(
    query: str,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    method_type: str,
    method_config: Dict[str, Any],
    embedding_model: str,
    top_k: int = 10
) -> List[tuple]:
    """
    Retrieve documents using the specified method
    
    Args:
        query: The query string
        chunk_embeddings: Embeddings of chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs
        method_type: Type of retrieval method
        method_config: Configuration for the method
        embedding_model: Name of the embedding model
        top_k: Number of results to retrieve
        
    Returns:
        List of (doc_id, score) tuples
    """
    if method_type == "vector":
        # Embed query
        query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
            [query], model_name=embedding_model
        )[0]
        
        # Retrieve
        search_results = RetrievalMethods.vector_search(
            query_embedding, chunk_embeddings, doc_ids, top_k=top_k
        )
        
    elif method_type == "bm25":
        search_results = RetrievalMethods.bm25_search(
            query, chunked_docs, top_k=top_k
        )
        
    elif method_type == "hybrid":
        # Embed query
        query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
            [query], model_name=embedding_model
        )[0]
        
        # Retrieve with custom alpha
        alpha = method_config.get("alpha", 0.5)
        search_results = RetrievalMethods.hybrid_search(
            query, query_embedding, chunked_docs, chunk_embeddings,
            alpha=alpha, top_k=top_k
        )
        
    else:
        raise ValueError(f"Unknown retrieval method type: {method_type}")
        
    return search_results

def load_best_configuration(results_dir: str) -> Dict[str, Any]:
    """
    Load the best configuration from previous experiments
    
    Args:
        results_dir: Directory with experiment results
        
    Returns:
        Dictionary with best chunking and embedding settings
    """
    best_config = {}
    
    # Try to find previous experiment results
    if not os.path.exists(results_dir):
        return best_config
        
    # Look for chunking experiment results
    chunking_dirs = [d for d in os.listdir(results_dir) 
                    if d.startswith("chunking_experiment") and os.path.isdir(os.path.join(results_dir, d))]
                    
    if chunking_dirs:
        latest_chunking = max(chunking_dirs)
        chunking_results = os.path.join(results_dir, latest_chunking, "results.csv")
        
        if os.path.exists(chunking_results):
            # Load results and find best configuration by MRR
            try:
                df = pd.read_csv(chunking_results)
                
                if "metric_mrr" in df.columns:
                    best_idx = df["metric_mrr"].idxmax()
                    best_row = df.loc[best_idx]
                    
                    best_config["chunking_strategy"] = best_row.get("chunking_strategy")
                    
                    if best_row.get("chunking_strategy") == "fixed":
                        best_config["chunk_size"] = best_row.get("chunk_size")
                        best_config["chunk_overlap"] = best_row.get("chunk_overlap")
            except:
                pass
                
    # Look for embedding experiment results
    embedding_dirs = [d for d in os.listdir(results_dir) 
                     if d.startswith("embedding_experiment") and os.path.isdir(os.path.join(results_dir, d))]
                     
    if embedding_dirs:
        latest_embedding = max(embedding_dirs)
        embedding_results = os.path.join(results_dir, latest_embedding, "results.csv")
        
        if os.path.exists(embedding_results):
            # Load results and find best configuration by MRR
            try:
                df = pd.read_csv(embedding_results)
                
                if "metric_mrr" in df.columns:
                    best_idx = df["metric_mrr"].idxmax()
                    best_row = df.loc[best_idx]
                    
                    best_config["embedding_model"] = best_row.get("embedding_model")
            except:
                pass
                
    return best_config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run retrieval method experiments")
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.pkl", help="Path to corpus file")
    parser.add_argument("--queries", type=str, default="data/sample_queries.pkl", help="Path to queries file")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of documents to use")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate")
    parser.add_argument("--chunk-strategy", type=str, default="fixed", help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int, default=128, help="Size of chunks for fixed strategy")
    parser.add_argument("--chunk-overlap", type=int, default=0, help="Overlap for fixed chunking")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    run_retrieval_experiment(
        corpus_file=args.corpus,
        queries_file=args.queries,
        sample_size=args.sample_size,
        num_queries=args.num_queries,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir
    )