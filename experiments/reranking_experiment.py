"""
Reranking Experiment

This script runs experiments to compare different reranking methods:
- No reranking (baseline)
- Cross-encoder reranking
- LLM-based reranking
- Reciprocal Rank Fusion
- Diversity-aware reranking

It evaluates how these reranking techniques affect retrieval performance.
"""

import os
import sys
import pickle
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods
from src.components.reranking import RerankerModule
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker
from src.components.advanced_reranking import AdvancedReranker

# Mock LLM function for reranking
def mock_llm_reranker(prompt: str) -> str:
    """
    Mock LLM function that scores document relevance
    In real system, this would call an actual LLM
    
    Args:
        prompt: Prompt asking for relevance score
        
    Returns:
        A relevance score string
    """
    # For demo, generate a random score between 1-10
    import random
    
    # Extract query and document from prompt for analysis
    query = ""
    document = ""
    in_query = False
    in_document = False
    
    for line in prompt.split('\n'):
        line = line.strip()
        if line.startswith("Query:"):
            in_query = True
            in_document = False
            query = line[6:].strip()
        elif line.startswith("Document:"):
            in_query = False
            in_document = True
            document = ""
        elif in_document:
            document += line + " "
    
    # Simple heuristic: if query words appear in document, give higher score
    score = 5  # Default score
    
    if query and document:
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Count matching words
        matches = len(query_words.intersection(doc_words))
        
        # Adjust score based on matches (simple heuristic)
        if matches > 0:
            score = min(10, 5 + matches)
        else:
            score = max(1, 5 - len(query_words) // 2)
            
        # Add some randomness
        score = max(1, min(10, score + random.randint(-1, 1)))
    
    return str(score)
def run_reranking_experiment(
    corpus_file: str = "data/sample_corpus.pkl",
    queries_file: str = "data/sample_queries.pkl",
    sample_size: int = 50,
    num_queries: int = 5,
    chunk_strategy: str = "fixed",
    chunk_size: int = 128,      # Default is int
    chunk_overlap: int = 0,       # Default is int
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_function: Callable = mock_llm_reranker, # Corrected default from paste.txt
    output_dir: str = "results"
):
    """
    Run experiments to compare different reranking methods
    """
    # --- Keep os.makedirs, data loading, sampling exactly as in paste.txt ---
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(corpus_file, "rb") as f: corpus_data = pickle.load(f)
    except FileNotFoundError: print(f"Corpus file not found: {corpus_file}"); return None
    try:
        with open(queries_file, "rb") as f: query_data = pickle.load(f)
    except FileNotFoundError: print(f"Queries file not found: {queries_file}"); return None
    if sample_size < len(corpus_data): corpus_sample = corpus_data[:sample_size]
    else: corpus_sample = corpus_data
    if num_queries < len(query_data): queries_sample = query_data[:num_queries]
    else: queries_sample = query_data
    print(f"Running reranking experiments with {len(corpus_sample)} documents and {len(queries_sample)} queries")

    # --- Keep tracker initialization exactly as in paste.txt ---
    tracker = ExperimentTracker("reranking_experiment")

    # --- Load best configuration (Keep exactly as in paste.txt) ---
    best_config = load_best_configuration(output_dir)

    # --- Assign and FIX TYPES for chunk_size and chunk_overlap ---
    # Start with function defaults (which are ints)
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
        # No else needed, current_chunk_size already holds the default if loaded_chunk_size is None

        if loaded_chunk_overlap is not None:
            try:
                current_chunk_overlap = int(loaded_chunk_overlap) # Convert to int
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loaded chunk_overlap '{loaded_chunk_overlap}' to int. Using default: {chunk_overlap}")
                current_chunk_overlap = chunk_overlap # Fallback to default int
        # No else needed, current_chunk_overlap already holds the default if loaded_chunk_overlap is None
        # --- END ADDED TYPE CONVERSION ---

    # Use the correctly typed variables from now on
    # Renaming for clarity, using values determined above
    final_chunk_strategy = current_chunk_strategy
    final_chunk_size = current_chunk_size         # This is now guaranteed to be an int
    final_chunk_overlap = current_chunk_overlap     # This is now guaranteed to be an int
    final_embedding_model = current_embedding_model

    # Print the final values being used
    print(f"Using chunking strategy: {final_chunk_strategy}")
    print(f"Using chunk size: {final_chunk_size} (type: {type(final_chunk_size).__name__})") # Should be int
    print(f"Using chunk overlap: {final_chunk_overlap} (type: {type(final_chunk_overlap).__name__})") # Should be int
    print(f"Using embedding model: {final_embedding_model}")


    # --- Keep Log experiment configuration exactly as in paste.txt (using final_ variables) ---
    tracker.log_experiment_config({
        "dataset": os.path.basename(corpus_file),
        "sample_size": len(corpus_sample),
        "num_queries": len(queries_sample),
        "chunk_strategy": final_chunk_strategy,
        "chunk_size": final_chunk_size,         # Log the int value
        "chunk_overlap": final_chunk_overlap,     # Log the int value
        "embedding_model": final_embedding_model
    })

    # Apply chunking strategy (use final, correctly typed values)
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
         # Extra safety check
         print(f"\nCRITICAL ERROR during chunking: {te}")
         print(f"Values passed: size={final_chunk_size} (type {type(final_chunk_size).__name__}), overlap={final_chunk_overlap} (type {type(final_chunk_overlap).__name__})")
         raise te
    except Exception as e:
         print(f"Error during chunking: {e}"); return None

    print(f"Created {len(chunked_docs)} chunks")

    # --- Keep Extract texts and IDs exactly as in paste.txt ---
    # NOTE: Requires validation loop here if chunking methods don't guarantee structure
    try:
        chunk_texts = [doc["text"] for doc in chunked_docs]
        doc_ids = [doc["chunk_id"] for doc in chunked_docs]
    except KeyError as ke:
         print(f"ERROR: KeyError '{ke}' accessing chunk data. Ensure chunking returns dicts with 'text' and 'chunk_id'.")
         return None

    # --- Keep Generate embeddings exactly as in paste.txt ---
    print(f"Generating embeddings using {final_embedding_model}...")
    try:
        # Requires ORIGINAL static EmbeddingProvider
        chunk_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
            chunk_texts, model_name=final_embedding_model
        )
    except AttributeError as ae:
        print(f"\nERROR: {ae}. Check EmbeddingProvider class uses ORIGINAL static methods.")
        return None
    except Exception as e:
        print(f"Error generating embeddings: {e}"); return None


    # --- Keep Define reranking methods exactly as in paste.txt ---
    reranking_methods = [
        {"name": "no_reranking", "description": "Base retrieval without reranking"},
        {"name": "cross_encoder", "description": "Cross-encoder reranking", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        {"name": "llm_reranking", "description": "LLM-based reranking"},
        {"name": "hybrid_fusion", "description": "Fusion of vector and BM25 rankings"},
        {"name": "cross_encoder_fusion", "description": "Fusion of vector and cross-encoder rankings"},
        {"name": "multi_stage", "description": "Multi-stage reranking pipeline", "stages": ["semantic", "cross_encoder", "diversity"]},
        {"name": "keyword_fusion", "description": "Keyword and semantic fusion", "stages": ["semantic", "keyword", "diversity"]}
    ]

    # --- Keep Test each reranking method loop structure exactly as in paste.txt ---
    for method in reranking_methods:
        method_name = method["name"]
        print(f"\nEvaluating reranking method: {method_name}")
        try:
            start_time = time.time()
            # --- Keep call to evaluate_reranking_method exactly as in paste.txt ---
            reranking_metrics = evaluate_reranking_method(
                queries_sample,
                chunk_embeddings,
                chunked_docs,
                doc_ids,
                final_embedding_model, # Use final model name
                method, # Pass the whole method config dict
                llm_function # Pass the llm function
            )
            reranking_time = time.time() - start_time

            # --- Keep Log results exactly as in paste.txt ---
            # Define sanitize helper if needed locally
            def sanitize_dict_for_json(d):
                 if not isinstance(d, dict): return d
                 return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}

            tracker.log_iteration({
                "reranking_method": method_name,
                "description": method["description"],
                "reranking_time_sec": reranking_time,
                # Use .get for safety
                "queries_per_second": len(queries_sample) / reranking_time if reranking_time > 0 else 0,
                **sanitize_dict_for_json({f"metric_{k}": v for k, v in reranking_metrics.items()})
                # "metric_precision_at_1": reranking_metrics.get("precision_at_1"), # Using .get is safer
                # "metric_precision_at_3": reranking_metrics.get("precision_at_3"),
                # "metric_precision_at_5": reranking_metrics.get("precision_at_5"),
                # "metric_recall_at_3": reranking_metrics.get("recall_at_3"),
                # "metric_recall_at_5": reranking_metrics.get("recall_at_5"),
                # "metric_mrr": reranking_metrics.get("mrr")
            })

        except Exception as e:
            print(f"Error evaluating method {method_name}: {e}")
            import traceback; traceback.print_exc()
            # Log failure
            failure_log_data = {"reranking_method": method_name, "status": "Failed", "error": str(e)}
            tracker.log_iteration(failure_log_data)
            continue # Continue to next method
    # --- End reranker loop ---

    # --- Keep Generate report exactly as in paste.txt ---
    print("Generating experiment report...")
    try:
        report_path = tracker.generate_report()
        print(f"Report generated at {report_path}")
        return report_path
    except Exception as e:
        print(f"Error generating report: {e}"); return None
# --- End run_reranking_experiment ---


def evaluate_reranking_method(
    queries: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    embedding_model: str,
    method_config: Dict[str, Any],
    llm_function: Callable
) -> Dict[str, float]:
    """
    Evaluate a specific reranking method
    
    Args:
        queries: List of query objects with 'question' and 'answer'
        chunk_embeddings: Embeddings of all chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs corresponding to chunk_embeddings
        embedding_model: Name of the embedding model
        method_config: Configuration for the reranking method
        llm_function: Function to generate text with an LLM
        
    Returns:
        Dictionary of evaluation metrics
    """
    method_name = method_config["name"]
    
    # Initialize metrics
    precision_at_1_sum = 0
    precision_at_3_sum = 0
    precision_at_5_sum = 0
    recall_at_3_sum = 0
    recall_at_5_sum = 0
    mrr_sum = 0
    
    # For each query, calculate retrieval metrics
    for query_obj in tqdm(queries, desc=f"Evaluating {method_name}"):
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
        
        # Initial retrieval (for all methods)
        query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
            [query], model_name=embedding_model
        )[0]
        
        # Get top results from vector search
        vector_results = RetrievalMethods.vector_search(
            query_embedding, chunk_embeddings, doc_ids, top_k=20
        )
        
        # Get retrieved documents
        retrieved_doc_ids = [doc_id for doc_id, _ in vector_results]
        retrieved_docs = []
        
        for doc_id, score in vector_results:
            idx = doc_ids.index(doc_id)
            doc = chunked_docs[idx].copy()
            doc["score"] = score
            retrieved_docs.append(doc)
        
        # Apply reranking based on method
        if method_name == "no_reranking":
            # No reranking, use vector search results directly
            final_results = retrieved_doc_ids[:10]
        elif method_name == "cross_encoder":
            # Use cross-encoder for reranking
            model_name = method_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            try:
                # Get reranked documents
                reranked_pairs = RerankerModule.score_with_cross_encoder(
                    query, retrieved_docs, model_name=model_name
                )
                
                # Extract document IDs
                final_results = [doc["chunk_id"] for doc, _ in reranked_pairs[:10]]
            except Exception as e:
                print(f"Error in cross-encoder reranking: {e}")
                # Fallback to original order
                final_results = retrieved_doc_ids[:10]
        elif method_name == "llm_reranking":
            # Use LLM for reranking
            try:
                # Get reranked documents
                reranked_pairs = RerankerModule.score_with_llm_api(
                    query, retrieved_docs, llm_function=llm_function
                )
                
                # Extract document IDs
                final_results = [doc["chunk_id"] for doc, _ in reranked_pairs[:10]]
            except Exception as e:
                print(f"Error in LLM reranking: {e}")
                # Fallback to original order
                final_results = retrieved_doc_ids[:10]
        elif method_name == "hybrid_fusion":
            # Get BM25 results
            bm25_results = RetrievalMethods.bm25_search(
                query, chunked_docs, top_k=20
            )
            
            # Convert to doc objects for fusion
            bm25_docs = []
            for doc_id, score in bm25_results:
                idx = doc_ids.index(doc_id)
                doc = chunked_docs[idx].copy()
                doc["score"] = score
                bm25_docs.append((doc, score))
            
            # Apply RRF fusion
            vector_docs = [(doc, score) for doc, score in zip(retrieved_docs, [r[1] for r in vector_results])]
            fused_docs = RerankerModule.reciprocal_rank_fusion([vector_docs, bm25_docs])
            
            # Extract document IDs
            final_results = [doc["chunk_id"] for doc, _ in fused_docs[:10]]
        elif method_name == "cross_encoder_fusion":
            # Get cross-encoder results
            try:
                model_name = method_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                cross_encoder_docs = RerankerModule.score_with_cross_encoder(
                    query, retrieved_docs, model_name=model_name
                )
                
                # Convert vector results to format for fusion
                vector_docs = [(doc, score) for doc, score in zip(retrieved_docs, [r[1] for r in vector_results])]
                
                # Apply RRF fusion
                fused_docs = RerankerModule.reciprocal_rank_fusion([vector_docs, cross_encoder_docs])
                
                # Extract document IDs
                final_results = [doc["chunk_id"] for doc, _ in fused_docs[:10]]
            except Exception as e:
                print(f"Error in cross-encoder fusion: {e}")
                # Fallback to original order
                final_results = retrieved_doc_ids[:10]
        elif method_name == "multi_stage" or method_name == "keyword_fusion":
            # Create advanced reranker
            advanced_reranker = AdvancedReranker()
            
            # Get stages
            stages = method_config.get("stages", ["semantic", "cross_encoder", "diversity"])
            
            # Perform multi-stage reranking
            reranked_pairs = advanced_reranker.multi_stage_reranking(
                query, query_embedding, retrieved_docs, chunk_embeddings, top_k=10, stages=stages
            )
            
            # Extract document IDs
            final_results = [doc_id for doc, _ in reranked_pairs[:10]]
        else:
            raise ValueError(f"Unknown reranking method: {method_name}")
        
        # Calculate metrics
        precision_at_1 = RAGEvaluator.precision_at_k(relevant_chunks, final_results, k=1)
        precision_at_3 = RAGEvaluator.precision_at_k(relevant_chunks, final_results, k=3)
        precision_at_5 = RAGEvaluator.precision_at_k(relevant_chunks, final_results, k=5)
        recall_at_3 = RAGEvaluator.recall_at_k(relevant_chunks, final_results, k=3)
        recall_at_5 = RAGEvaluator.recall_at_k(relevant_chunks, final_results, k=5)
        mrr = RAGEvaluator.mean_reciprocal_rank(relevant_chunks, final_results)
        
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
    
    print(f"Metrics for {method_name}:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")
        
    return metrics

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
    
    parser = argparse.ArgumentParser(description="Run reranking experiments")
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
    
    run_reranking_experiment(
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