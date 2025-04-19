"""
Embedding Experiment

This script runs experiments to compare different embedding models:
- Various Sentence Transformer models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- HuggingFace models
- OpenAI embedding models (if API key is available)

It evaluates how different embedding models affect retrieval performance,
considering both effectiveness (precision, recall, MRR) and efficiency (speed).
"""

import os
import sys
import pickle
import time
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker

def run_embedding_experiment(
    corpus_file: str = "data/sample_corpus.pkl",
    queries_file: str = "data/sample_queries.pkl",
    sample_size: int = 50,
    num_queries: int = 5,
    chunk_strategy: str = "fixed",
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    output_dir: str = "results"
):
    """
    Run experiments to compare different embedding models
    
    Args:
        corpus_file: Path to corpus pickle file
        queries_file: Path to queries pickle file
        sample_size: Number of documents to use
        num_queries: Number of queries to evaluate
        chunk_strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        chunk_overlap: Overlap for fixed chunking
        output_dir: Directory to save results
        
    Returns:
        Path to the experiment report
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        with open(corpus_file, "rb") as f:
            corpus_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Corpus file not found: {corpus_file}")
        print("Please run test_framework.py first to generate sample data.")
        return None
        
    try:
        with open(queries_file, "rb") as f:
            query_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Queries file not found: {queries_file}")
        print("Please run test_framework.py first to generate sample data.")
        return None
    
    # Sample data for experiment
    if sample_size < len(corpus_data):
        corpus_sample = corpus_data[:sample_size]
    else:
        corpus_sample = corpus_data
        
    if num_queries < len(query_data):
        queries_sample = query_data[:num_queries]
    else:
        queries_sample = query_data
        
    print(f"Running embedding experiments with {len(corpus_sample)} documents and {len(queries_sample)} queries")
    
    # Create experiment tracker
    tracker = ExperimentTracker("embedding_experiment")
    
    # Log experiment configuration
    tracker.log_experiment_config({
        "dataset": os.path.basename(corpus_file),
        "sample_size": len(corpus_sample),
        "num_queries": len(queries_sample),
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    
    # Apply chunking strategy
    if chunk_strategy == "fixed":
        chunked_docs = DocumentChunker.chunk_by_fixed_size(
            corpus_sample, chunk_size=chunk_size, overlap=chunk_overlap
        )
    elif chunk_strategy == "paragraph":
        chunked_docs = DocumentChunker.chunk_by_paragraph(corpus_sample)
    elif chunk_strategy == "semantic":
        chunked_docs = DocumentChunker.chunk_by_semantic_units(corpus_sample)
    else:
        raise ValueError(f"Unknown chunking strategy: {chunk_strategy}")
        
    print(f"Created {len(chunked_docs)} chunks")
    
    # Extract texts and IDs
    chunk_texts = [doc["text"] for doc in chunked_docs]
    doc_ids = [doc["chunk_id"] for doc in chunked_docs]
    
    # Define embedding models to test
    embedding_models = [
        # SentenceTransformers models of varying sizes and capabilities
        {"name": "all-MiniLM-L6-v2", "type": "sentence_transformer", "dim": 384},
        {"name": "all-mpnet-base-v2", "type": "sentence_transformer", "dim": 768},
        {"name": "multi-qa-mpnet-base-dot-v1", "type": "sentence_transformer", "dim": 768},
        {"name": "BAAI/bge-small-en-v1.5", "type": "huggingface", "dim": 384}
    ]
    
    if os.environ.get("OPENAI_API_KEY"):
        embedding_models.append(
            {"name": "text-embedding-ada-002", "type": "openai", "dim": 1536}
        )
    
    # Test each embedding model
    for model_config in embedding_models:
        model_name = model_config["name"]
        model_type = model_config["type"]
        
        print(f"\nEvaluating embedding model: {model_name}")
        
        try:
            # Generate embeddings
            start_time = time.time()
            
            if model_type == "sentence_transformer":
                chunk_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
                    chunk_texts, model_name=model_name
                )
            elif model_type == "openai":
                chunk_embeddings = EmbeddingProvider.get_openai_embeddings(
                    chunk_texts, model_name=model_name
                )
            elif model_type == "huggingface":
                chunk_embeddings = EmbeddingProvider.get_huggingface_embeddings(
                    chunk_texts, model_name=model_name
                )
            else:
                raise ValueError(f"Unknown embedding type: {model_type}")
                
            embed_time = time.time() - start_time
            
            print(f"Generated embeddings in {embed_time:.2f} seconds")
            
            # Evaluate vector retrieval with this embedding model
            vector_metrics = evaluate_vector_retrieval(
                queries_sample, chunk_embeddings, doc_ids, chunked_docs, model_name, model_type
            )
            
            # Log results
            tracker.log_iteration({
                "embedding_model": model_name,
                "embedding_type": model_type,
                "embedding_dim": model_config["dim"],
                "embedding_time_sec": embed_time,
                "docs_per_second": len(chunk_texts) / embed_time,
                "metric_precision_at_1": vector_metrics["precision_at_1"],
                "metric_precision_at_3": vector_metrics["precision_at_3"],
                "metric_precision_at_5": vector_metrics["precision_at_5"],
                "metric_recall_at_3": vector_metrics["recall_at_3"],
                "metric_recall_at_5": vector_metrics["recall_at_5"],
                "metric_mrr": vector_metrics["mrr"]
            })
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate a report
    print("Generating experiment report...")
    report_path = tracker.generate_report()
    print(f"Report generated at {report_path}")
    
    return report_path

def evaluate_vector_retrieval(
    queries: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    doc_ids: List[str],
    chunked_docs: List[Dict[str, str]],
    model_name: str,
    model_type: str
) -> Dict[str, float]:
    """
    Evaluate retrieval performance with vector search
    
    Args:
        queries: List of query objects with 'question' and 'answer'
        chunk_embeddings: Embeddings of all chunks
        doc_ids: List of document IDs corresponding to chunk_embeddings
        chunked_docs: List of chunk documents (needed for relevance judgments)
        model_name: Name of the embedding model (for embedding queries)
        model_type: Type of embedding model (sentence_transformer, openai, etc.)
        
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
    for query_obj in tqdm(queries, desc=f"Evaluating {model_name}"):
        query = query_obj["question"]
        expected_answer = query_obj.get("answer", "")
        
        # Find relevant chunks (simplified approach for this experiment)
        # In a real-world scenario, you'd have ground truth relevance judgments
        # Here we'll use a simple word overlap heuristic
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
        
        # Embed query using the same model as the chunks
        try:
            if model_type == "sentence_transformer":
                query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                    [query], model_name=model_name
                )[0]
            elif model_type == "openai":
                query_embedding = EmbeddingProvider.get_openai_embeddings(
                    [query], model_name=model_name
                )[0]
            elif model_type == "huggingface":
                query_embedding = EmbeddingProvider.get_huggingface_embeddings(
                    [query], model_name=model_name
                )[0]
            else:
                raise ValueError(f"Unknown embedding type: {model_type}")
                
            # Retrieve with vector search
            search_results = RetrievalMethods.vector_search(
                query_embedding, chunk_embeddings, doc_ids, top_k=10
            )
            
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
            
        except Exception as e:
            print(f"Error evaluating query with {model_name}: {e}")
            # Skip this query
            continue
    
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
    
    print(f"Retrieval metrics for {model_name}:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")
        
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding model experiments")
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.pkl", help="Path to corpus file")
    parser.add_argument("--queries", type=str, default="data/sample_queries.pkl", help="Path to queries file")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of documents to use")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate")
    parser.add_argument("--chunk-strategy", type=str, default="fixed", help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int, default=128, help="Size of chunks for fixed strategy")
    parser.add_argument("--chunk-overlap", type=int, default=0, help="Overlap for fixed chunking")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    run_embedding_experiment(
        corpus_file=args.corpus,
        queries_file=args.queries,
        sample_size=args.sample_size,
        num_queries=args.num_queries,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir
    )