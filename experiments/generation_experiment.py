"""
Generation Experiment

This script runs experiments to compare different generation approaches:
- Different prompt templates
- Various retrieval configurations for context
- Reranking impact on generation quality

It evaluates metrics like answer precision, faithfulness, and citation accuracy.
"""

import os
import sys
import pickle
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods, QueryProcessor
from src.components.reranking import RerankerModule
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker

# Mock LLM generation function
def mock_llm_generation(prompt: str) -> str:
    """
    Mock LLM function that generates text based on prompt
    In a real system, this would call an API or run a local model
    
    Args:
        prompt: The prompt to generate from
        
    Returns:
        Generated text
    """
    # For demo purposes, use a simple template-based approach
    # Extract query and context from prompt for analysis
    # Parse prompt assuming a structure like:
    # System: Instructions...
    # Context: Retrieved documents...
    # User: The query...
    
    query = ""
    context = ""
    instructions = ""
    current_section = None
    
    for line in prompt.split('\n'):
        if line.startswith("System:"):
            current_section = "system"
            instructions = ""
        elif line.startswith("Context:"):
            current_section = "context"
            context = ""
        elif line.startswith("User:") or line.startswith("Question:"):
            current_section = "user"
            query = line.split(":", 1)[1].strip()
        elif current_section == "system":
            instructions += line + " "
        elif current_section == "context":
            context += line + " "
    
    # Simple response generation logic
    if not context or context.strip() == "":
        # No context provided
        return f"I don't have enough information to answer about '{query}'. Could you provide more context?"
    
    # Simple extractive approach: find sentences in context that might contain answer
    import re
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    sentences = nltk.sent_tokenize(context)
    
    # Look for sentences containing query terms
    query_terms = query.lower().split()
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(term in sentence_lower for term in query_terms):
            relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        # No relevant sentences found
        return f"Based on the information provided, I cannot find a specific answer to '{query}'."
    
    # Craft response based on relevant sentences
    response = f"Based on the provided information, "
    
    if len(relevant_sentences) == 1:
        response += relevant_sentences[0]
    else:
        # Add citations if instructed
        if "citation" in instructions.lower() or "cite" in instructions.lower():
            for i, sentence in enumerate(relevant_sentences[:3]):  # Limit to 3 sentences
                if i == 0:
                    response += f"{sentence} [1] "
                else:
                    response += f"Furthermore, {sentence.lower()} [2] "
        else:
            for i, sentence in enumerate(relevant_sentences[:3]):  # Limit to 3 sentences
                if i == 0:
                    response += sentence
                else:
                    response += f" Furthermore, {sentence.lower()}"
    
    return response


# --- run_generation_experiment Function (WITH TYPE CONVERSION FIX) ---
def run_generation_experiment(
    corpus_file: str = "data/sample_corpus.pkl",
    queries_file: str = "data/sample_queries.pkl",
    sample_size: int = 50,
    num_queries: int = 5,
    chunk_strategy: str = "fixed",
    chunk_size: int = 128,      # Default is int
    chunk_overlap: int = 0,       # Default is int
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_function: Callable = mock_llm_generation, # Default from paste.txt
    output_dir: str = "results"
):
    """
    Run experiments to compare different generation techniques
    """
    # --- Keep os.makedirs, data loading, sampling exactly as in paste-2.txt ---
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
    print(f"Running generation experiments with {len(corpus_sample)} documents and {len(queries_sample)} queries")

    # --- Keep tracker initialization exactly as in paste-2.txt ---
    tracker = ExperimentTracker("generation_experiment")

    # --- Load best configuration (Keep exactly as in paste-2.txt) ---
    best_config = load_best_configuration(output_dir)

    # --- Assign and FIX TYPES for chunk_size and chunk_overlap ---
    # Start with function defaults (which are ints)
    current_chunk_strategy = chunk_strategy
    current_chunk_size = chunk_size             # Holds default int
    current_chunk_overlap = chunk_overlap           # Holds default int
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
                # current_chunk_size already holds the default from initialization
        # If loaded_chunk_size was None, current_chunk_size retains its default int value

        if loaded_chunk_overlap is not None:
            try:
                current_chunk_overlap = int(loaded_chunk_overlap) # Convert to int
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loaded chunk_overlap '{loaded_chunk_overlap}' to int. Using default: {chunk_overlap}")
                # current_chunk_overlap already holds the default from initialization
        # If loaded_chunk_overlap was None, current_chunk_overlap retains its default int value
        # --- END ADDED TYPE CONVERSION ---

    # Use the final, correctly typed variables from now on
    print(f"Using chunking strategy: {current_chunk_strategy}")
    print(f"Using chunk size: {current_chunk_size} (type: {type(current_chunk_size).__name__})") # Should be int
    print(f"Using chunk overlap: {current_chunk_overlap} (type: {type(current_chunk_overlap).__name__})") # Should be int
    print(f"Using embedding model: {current_embedding_model}")
    # --- END OF FIX ---


    # --- Keep Log experiment configuration exactly as in paste-2.txt (using current_ variables) ---
    tracker.log_experiment_config({
        "dataset": os.path.basename(corpus_file),
        "sample_size": len(corpus_sample),
        "num_queries": len(queries_sample),
        "chunk_strategy": current_chunk_strategy,
        "chunk_size": current_chunk_size,         # Log the int value
        "chunk_overlap": current_chunk_overlap,     # Log the int value
        "embedding_model": current_embedding_model
    })

    # Apply chunking strategy (use current_, correctly typed values)
    # --- This call should now receive integers ---
    try:
        if current_chunk_strategy == "fixed":
            chunked_docs = DocumentChunker.chunk_by_fixed_size(
                corpus_sample, chunk_size=current_chunk_size, overlap=current_chunk_overlap
            )
        elif current_chunk_strategy == "paragraph":
            chunked_docs = DocumentChunker.chunk_by_paragraph(corpus_sample)
        elif current_chunk_strategy == "semantic":
            chunked_docs = DocumentChunker.chunk_by_semantic_units(corpus_sample)
        else:
            raise ValueError(f"Unknown chunking strategy: {current_chunk_strategy}")
    except TypeError as te:
         # Extra safety check
         print(f"\nCRITICAL ERROR during chunking: {te}")
         print(f"Values passed: size={current_chunk_size} (type {type(current_chunk_size).__name__}), overlap={current_chunk_overlap} (type {type(current_chunk_overlap).__name__})")
         raise te # Re-raise the error
    except Exception as e:
         print(f"Error during chunking: {e}"); return None # Exit if chunking fails

    print(f"Created {len(chunked_docs)} chunks")

    # --- Keep Extract texts and IDs exactly as in paste-2.txt ---
    # NOTE: Assumes chunking returns dicts with 'text'/'chunk_id'. Add validation if needed.
    try:
        chunk_texts = [doc["text"] for doc in chunked_docs]
        doc_ids = [doc["chunk_id"] for doc in chunked_docs]
    except KeyError as ke:
        print(f"ERROR: KeyError '{ke}' accessing chunk data. Ensure strategy '{current_chunk_strategy}' returns dicts with 'text'/'chunk_id'.")
        return None

    # --- Keep Generate embeddings exactly as in paste-2.txt ---
    print(f"Generating embeddings using {current_embedding_model}...")
    try:
        # Requires ORIGINAL static EmbeddingProvider
        chunk_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(
            chunk_texts, model_name=current_embedding_model
        )
    except AttributeError as ae:
        print(f"\nERROR: {ae}. Check EmbeddingProvider class uses ORIGINAL static methods.")
        return None
    except Exception as e:
        print(f"Error generating embeddings: {e}"); return None


    # --- Keep Define prompt templates exactly as in paste-2.txt ---
    prompt_templates = [
        {"name": "basic", "description": "Basic prompt", "template": "Context: {context}\n\nQuestion: {query}\n\nAnswer:"},
        {"name": "instructional", "description": "Detailed instructions", "template": "System: You are a helpful assistant... Use ONLY the following context... If the context doesn't contain the answer, say you don't know.\n\nContext: {context}\n\nUser: {query}"}, # Removed ellipsis from original paste
        {"name": "extractive", "description": "Extract answer", "template": "System: You must extract the answer... Quote the relevant parts...\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"},
        {"name": "citation", "description": "Request citations", "template": "System: You are a helpful assistant... Include citations...\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer with citations:"}
    ]

    # --- Keep Define retrieval configs exactly as in paste-2.txt ---
    retrieval_configs = [
        {"name": "top_3", "description": "Top 3 documents", "k": 3},
        {"name": "top_5", "description": "Top 5 documents", "k": 5},
        {"name": "reranked_3", "description": "Top 10, reranked, top 3 used", "k": 3, "rerank": True}
    ]

    # --- Keep Test combinations loop exactly as in paste-2.txt ---
    for prompt_config in prompt_templates:
        for retrieval_config in retrieval_configs:
            config_name = f"{prompt_config['name']}_{retrieval_config['name']}"
            print(f"\nEvaluating generation with: {config_name}")
            try:
                start_time = time.time()
                # Keep call to evaluate_generation exactly as in paste-2.txt
                generation_metrics = evaluate_generation(
                    queries_sample, chunk_embeddings, chunked_docs, doc_ids,
                    current_embedding_model, # Use final model name
                    prompt_config, retrieval_config, llm_function
                )
                generation_time = time.time() - start_time

                # Keep Log results exactly as in paste-2.txt
                # Define sanitize helper if not global/imported
                def sanitize_dict_for_json(d):
                     if not isinstance(d, dict): return d
                     return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}
                log_data = {
                    "prompt_template": prompt_config["name"], "prompt_description": prompt_config["description"],
                    "retrieval_config": retrieval_config["name"], "retrieval_description": retrieval_config["description"],
                    "generation_time_sec": generation_time,
                    "queries_per_second": len(queries_sample) / generation_time if generation_time > 0 else 0,
                    # Using .get for safety, assuming keys match evaluate_answer output
                    **sanitize_dict_for_json({f"metric_{k}": v for k, v in generation_metrics.items()})
                    # "metric_answer_precision": generation_metrics.get("answer_precision"),
                    # "metric_answer_recall": generation_metrics.get("answer_recall"),
                    # "metric_faithfulness": generation_metrics.get("faithfulness"),
                    # "metric_citation_rate": generation_metrics.get("citation_rate"),
                    # "metric_answer_correctness": generation_metrics.get("answer_correctness")
                }
                tracker.log_iteration(log_data)

            except Exception as e:
                print(f"Error evaluating configuration {config_name}: {e}")
                import traceback; traceback.print_exc()
                # Log failure
                failure_log_data = {
                    "prompt_template": prompt_config["name"], "retrieval_config": retrieval_config["name"],
                    "status": "Failed", "error": str(e)
                }
                tracker.log_iteration(failure_log_data)
                continue # Continue to next config combination
    # --- End combination loop ---

    # --- Keep Generate report exactly as in paste-2.txt ---
    print("Generating experiment report...")
    try:
        report_path = tracker.generate_report()
        print(f"Report generated at {report_path}")
        return report_path
    except Exception as e:
        print(f"Error generating report: {e}"); return None
# --- End run_generation_experiment ---

def evaluate_generation(
    queries: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    embedding_model: str,
    prompt_config: Dict[str, str],
    retrieval_config: Dict[str, Any],
    llm_function: Callable
) -> Dict[str, float]:
    """
    Evaluate generation with specific prompt and retrieval configurations
    
    Args:
        queries: List of query objects with 'question' and 'answer'
        chunk_embeddings: Embeddings of all chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs
        embedding_model: Name of embedding model
        prompt_config: Configuration for prompt template
        retrieval_config: Configuration for retrieval
        llm_function: Function to generate text
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize metrics
    answer_precision_sum = 0
    answer_recall_sum = 0
    faithfulness_sum = 0
    citation_rate_sum = 0
    answer_correctness_sum = 0
    
    # Process each query
    for query_obj in tqdm(queries, desc=f"Generating answers"):
        query = query_obj["question"]
        expected_answer = query_obj.get("answer", "")
        
        # Retrieve relevant context
        context_chunks, retrieved_texts = retrieve_context(
            query,
            chunk_embeddings,
            chunked_docs,
            doc_ids,
            embedding_model,
            retrieval_config
        )
        
        # Format prompt
        prompt_template = prompt_config["template"]
        context_text = "\n\n".join(retrieved_texts)
        prompt = prompt_template.format(context=context_text, query=query)
        
        # Generate answer
        generated_answer = llm_function(prompt)
        
        # Evaluate answer
        # In a real system, you might use more sophisticated evaluation metrics
        metrics = evaluate_answer(
            generated_answer,
            expected_answer,
            retrieved_texts,
            context_chunks
        )
        
        # Add to sums
        answer_precision_sum += metrics["answer_precision"]
        answer_recall_sum += metrics["answer_recall"]
        faithfulness_sum += metrics["faithfulness"]
        citation_rate_sum += metrics["citation_rate"]
        answer_correctness_sum += metrics["answer_correctness"]
    
    # Calculate averages
    num_queries = len(queries)
    avg_metrics = {
        "answer_precision": answer_precision_sum / num_queries,
        "answer_recall": answer_recall_sum / num_queries,
        "faithfulness": faithfulness_sum / num_queries,
        "citation_rate": citation_rate_sum / num_queries,
        "answer_correctness": answer_correctness_sum / num_queries
    }
    
    print(f"Generation metrics for {prompt_config['name']} with {retrieval_config['name']}:")
    for name, value in avg_metrics.items():
        print(f" {name}: {value:.4f}")
        
    return avg_metrics

def retrieve_context(
    query: str,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, str]],
    doc_ids: List[str],
    embedding_model: str,
    retrieval_config: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Retrieve context for generation
    
    Args:
        query: The query string
        chunk_embeddings: Embeddings of chunks
        chunked_docs: List of chunk documents
        doc_ids: List of document IDs
        embedding_model: Name of embedding model
        retrieval_config: Configuration for retrieval
        
    Returns:
        Tuple of (retrieved_chunks, retrieved_texts)
    """
    k = retrieval_config.get("k", 3)
    should_rerank = retrieval_config.get("rerank", False)
    
    # Embed query
    query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
        [query], model_name=embedding_model
    )[0]
    
    # Initial retrieval
    initial_k = 10 if should_rerank else k
    
    results = RetrievalMethods.vector_search(
        query_embedding, chunk_embeddings, doc_ids, top_k=initial_k
    )
    
    # Get retrieved documents
    retrieved_chunks = []
    for doc_id, score in results:
        idx = doc_ids.index(doc_id)
        doc = chunked_docs[idx].copy()
        doc["score"] = score
        retrieved_chunks.append(doc)
    
    # Apply reranking if needed
    if should_rerank:
        try:
            reranked_docs = RerankerModule.score_with_cross_encoder(
                query, retrieved_chunks
            )
            
            retrieved_chunks = [doc for doc, _ in reranked_docs[:k]]
        except:
            # Fallback to original results if reranking fails
            retrieved_chunks = retrieved_chunks[:k]
    
    # Extract text from chunks
    retrieved_texts = [chunk["text"] for chunk in retrieved_chunks]
    
    return retrieved_chunks, retrieved_texts

def evaluate_answer(
    generated_answer: str,
    expected_answer: str,
    retrieved_texts: List[str],
    context_chunks: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Evaluate the quality of the generated answer
    
    Args:
        generated_answer: The answer generated by the LLM
        expected_answer: The expected answer
        retrieved_texts: The texts retrieved as context
        context_chunks: The chunks retrieved as context
        
    Returns:
        Dictionary of evaluation metrics
    """
    # In a real system, you would use more sophisticated metrics
    # This is a simplified implementation
    
    # Answer precision/recall using ROUGE scores
    metrics = RAGEvaluator.rouge_scores(generated_answer, expected_answer)
    answer_precision = metrics.get("rouge1", 0.0)
    answer_recall = metrics.get("rouge1", 0.0)
    
    # Faithfulness (how well the answer sticks to the retrieved context)
    faithfulness = RAGEvaluator.faithfulness_score(generated_answer, retrieved_texts)
    
    # Check for citations (simplified)
    citation_rate = 0.0
    if "[" in generated_answer and "]" in generated_answer:
        # Looks like citation brackets
        citation_rate = 1.0
    elif "according to" in generated_answer.lower() or "mentioned in" in generated_answer.lower():
        citation_rate = 0.8
    
    # Answer correctness (simplified)
    # In a real system, you might use an LLM to judge correctness
    answer_correctness = 0.5  # Default middle value
    
    # Simple heuristic: if there's significant overlap with expected answer, rate higher
    if expected_answer:
        # Calculate word overlap
        generated_words = set(generated_answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        overlap = len(generated_words.intersection(expected_words))
        
        if overlap > 5:
            answer_correctness = 0.8
        elif overlap > 2:
            answer_correctness = 0.6
    
    return {
        "answer_precision": answer_precision,
        "answer_recall": answer_recall,
        "faithfulness": faithfulness,
        "citation_rate": citation_rate,
        "answer_correctness": answer_correctness
    }

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
    
    parser = argparse.ArgumentParser(description="Run generation experiments")
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
    
    run_generation_experiment(
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