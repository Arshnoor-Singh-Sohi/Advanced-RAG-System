"""
Test Framework

This script tests the core components of the RAG system to ensure they work correctly.
It performs simple tests of chunking, embedding, retrieval, and evaluation.
"""

import os
import sys
import pickle
import time
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' data...")
    nltk.download('punkt')
# Explicitly download punkt_tab as it seems missing
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: # Use LookupError as indicated by the traceback
    print("Downloading NLTK 'punkt_tab' data...")
    nltk.download('punkt_tab') # This is the crucial line


# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods, QueryProcessor
from src.components.evaluation import RAGEvaluator
from src.utils.experiment_tracker import ExperimentTracker

def run_sample_test():
    """Run a simple test of the framework components"""
    # Create sample corpus
    sample_corpus = create_sample_corpus()
    
    # Create test queries
    sample_queries = create_sample_queries()
    
    print("Testing chunking strategies...")
    
    # Test fixed size chunking
    fixed_chunks = DocumentChunker.chunk_by_fixed_size(sample_corpus, chunk_size=100, overlap=20)
    print(f"Fixed size chunking created {len(fixed_chunks)} chunks")
    
    # Test paragraph chunking
    para_chunks = DocumentChunker.chunk_by_paragraph(sample_corpus)
    print(f"Paragraph chunking created {len(para_chunks)} chunks")
    
    # Test semantic chunking
    semantic_chunks = DocumentChunker.chunk_by_semantic_units(sample_corpus)
    print(f"Semantic chunking created {len(semantic_chunks)} chunks")
    
    print("\nTesting embedding generation...")
    
    # Test embedding a few documents
    sample_texts = [chunk['text'] for chunk in fixed_chunks[:5]]
    try:
        # Try SentenceTransformer embeddings (should work offline)
        embeddings = EmbeddingProvider.get_sentence_transformer_embeddings(sample_texts)
        print(f"Generated embeddings of shape {embeddings.shape}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
    
    print("\nTesting retrieval methods...")
    
    # Create a small test set
    test_query = sample_queries[0]['question']
    test_docs = fixed_chunks[:20]
    test_embeddings = EmbeddingProvider.get_sentence_transformer_embeddings([d['text'] for d in test_docs])
    query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings([test_query])[0]
    
    # Test vector search
    vector_results = RetrievalMethods.vector_search(
        query_embedding, test_embeddings, 
        [d['chunk_id'] for d in test_docs],
        top_k=3
    )
    
    print("Vector search results:")
    for doc_id, score in vector_results:
        print(f" {doc_id}: {score:.4f}")
    
    # Test BM25 search
    bm25_results = RetrievalMethods.bm25_search(test_query, test_docs, top_k=3)
    
    print("\nBM25 search results:")
    for doc_id, score in bm25_results:
        print(f" {doc_id}: {score:.4f}")
    
    # Test hybrid search
    hybrid_results = RetrievalMethods.hybrid_search(
        test_query, query_embedding, test_docs, test_embeddings, top_k=3
    )
    
    print("\nHybrid search results:")
    for doc_id, score in hybrid_results:
        print(f" {doc_id}: {score:.4f}")
    
    print("\nTesting evaluation metrics...")
    
    # Simulate known relevant documents (using top vector results as mock ground truth)
    relevant_docs = [doc_id for doc_id, _ in vector_results]
    retrieved_docs = [doc_id for doc_id, _ in hybrid_results]
    
    precision = RAGEvaluator.precision_at_k(relevant_docs, retrieved_docs, k=3)
    recall = RAGEvaluator.recall_at_k(relevant_docs, retrieved_docs, k=3)
    mrr = RAGEvaluator.mean_reciprocal_rank(relevant_docs, retrieved_docs)
    
    print(f"Precision@3: {precision:.4f}")
    print(f"Recall@3: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    print("\nTesting experiment tracking...")
    
    # Create a test experiment
    tracker = ExperimentTracker("test_experiment")
    
    # Log configuration
    tracker.log_experiment_config({
        "dataset": "Sample data",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_methods": ["vector", "bm25", "hybrid"]
    })
    
    # Log a few iterations
    tracker.log_iteration({
        "chunking_strategy": "fixed",
        "chunk_size": 100,
        "retrieval_method": "vector",
        "metric_precision": precision,
        "metric_recall": recall,
        "metric_mrr": mrr
    })
    
    tracker.log_iteration({
        "chunking_strategy": "paragraph",
        "chunk_size": "auto",
        "retrieval_method": "bm25",
        "metric_precision": precision * 0.9,  # Simulated worse results
        "metric_recall": recall * 0.8,
        "metric_mrr": mrr * 0.85
    })
    
    # Generate a report
    report_path = tracker.generate_report()
    print(f"Generated report at {report_path}")
    
    print("\nAll framework tests completed!")
    
    # Save sample data for other experiments
    os.makedirs("data", exist_ok=True)
    with open("data/sample_corpus.pkl", "wb") as f:
        pickle.dump(sample_corpus, f)
    with open("data/sample_queries.pkl", "wb") as f:
        pickle.dump(sample_queries, f)
    
    print("Saved sample data to data/ directory")
    
    return True

def create_sample_corpus(num_docs=50):
    """Create a sample corpus for testing"""
    corpus = []
    
    # Add some documents about RAG
    rag_docs = [
        {
            "title": "RAG Introduction",
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with information retrieval systems. It enhances the quality of generated text by retrieving relevant information from a knowledge base."
        },
        {
            "title": "RAG Components",
            "text": "A RAG system typically consists of three main components: 1) A Retriever that finds relevant information, 2) A Generator that creates responses, and 3) A Knowledge Base of documents or facts."
        },
        {
            "title": "Vector Search",
            "text": "Vector search is a key technique in RAG systems. It converts text into numerical vectors and finds documents with similar vector representations. This allows for semantic searching beyond simple keyword matching."
        },
        {
            "title": "Embedding Models",
            "text": "Embedding models transform text into numerical vectors that capture semantic meaning. Popular embedding models include models from OpenAI, Sentence Transformers, and various models available on Hugging Face."
        },
        {
            "title": "Document Chunking",
            "text": "Effective document chunking is crucial for RAG systems. Documents are split into smaller pieces to enable more precise retrieval. Common strategies include fixed-size chunks, paragraph-based, and semantic chunking."
        },
        {
            "title": "Retrieval Methods",
            "text": "Various retrieval methods can be used in RAG systems. Vector search uses embedding similarity, BM25 uses term frequency statistics, and hybrid methods combine both approaches. Each has different strengths for different types of queries."
        },
        {
            "title": "Reranking",
            "text": "Reranking is a technique used to improve retrieval results. After initial retrieval, documents are reranked using more expensive but accurate methods. Cross-encoders are commonly used for reranking in RAG systems."
        },
        {
            "title": "Prompt Engineering",
            "text": "Prompt engineering is important for effective RAG systems. The prompt tells the language model how to use the retrieved context. Well-designed prompts include clear instructions and properly formatted context."
        }
    ]
    
    corpus.extend(rag_docs)
    
    # Add some documents about machine learning (for diversity)
    ml_docs = [
        {
            "title": "Neural Networks",
            "text": "Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of artificial neurons organized in layers and can learn complex patterns from data through training."
        },
        {
            "title": "Supervised Learning",
            "text": "Supervised learning is a machine learning paradigm where models are trained on labeled data. The model learns to map inputs to correct outputs based on example input-output pairs provided during training."
        },
        {
            "title": "Unsupervised Learning",
            "text": "Unsupervised learning is a type of machine learning where models are trained on unlabeled data. These algorithms attempt to find patterns and structure in the data without explicit guidance."
        },
        {
            "title": "Reinforcement Learning",
            "text": "Reinforcement learning is a machine learning approach where agents learn to make decisions by taking actions in an environment to maximize cumulative rewards. It is inspired by behavioral psychology."
        }
    ]
    
    corpus.extend(ml_docs)
    
    # Add some longer documents to test chunking
    long_docs = [
        {
            "title": "RAG Research",
            "text": """Retrieval-Augmented Generation (RAG) has been an active area of research since its introduction. 
            The original RAG paper by Lewis et al. (2020) proposed combining neural retrieval with sequence generation.
            Since then, many improvements have been proposed, such as better retrieval mechanisms, more efficient indexing,
            and enhanced integration of retrieved information into the generation process.
            
            Some notable advancements include query expansion techniques to improve retrieval recall,
            reranking approaches to enhance retrieval precision, and more sophisticated generation methods
            that better leverage the retrieved context. There has also been work on adaptive retrieval,
            where the model decides when to retrieve information based on its confidence in answering a query.
            
            Recent work has also focused on evaluating RAG systems more effectively, developing better
            metrics for retrieval quality, faithfulness to retrieved information, and overall response quality.
            Evaluating RAG systems is challenging because it requires assessing both the retrieval and generation components,
            as well as how well they work together.
            
            Another important research direction is making RAG systems more efficient, reducing the computational
            cost of retrieval and enabling faster responses. This includes work on more efficient embedding models,
            better indexing strategies, and optimized retrieval algorithms."""
        },
        {
            "title": "Deep Learning Architectures",
            "text": """Deep learning has revolutionized machine learning and artificial intelligence through its powerful architectures.
            Convolutional Neural Networks (CNNs) have transformed computer vision by efficiently processing grid-like data.
            They use convolutional layers to detect local patterns like edges and shapes, pooling layers to reduce dimensionality,
            and fully connected layers for high-level reasoning.
            
            Recurrent Neural Networks (RNNs) were designed to handle sequential data like text and time series.
            Traditional RNNs suffered from vanishing gradient problems, leading to the development of Long Short-Term Memory (LSTM)
            and Gated Recurrent Unit (GRU) architectures. These architectures use gating mechanisms to control information flow
            and maintain long-term dependencies.
            
            Transformer architectures, introduced in the paper "Attention is All You Need," have largely replaced RNNs
            for many sequence modeling tasks. Transformers use self-attention mechanisms to weigh the importance of different
            parts of the input sequence. They have enabled significant advances in natural language processing and beyond.
            
            Graph Neural Networks (GNNs) process data represented as graphs, making them suitable for applications like
            social network analysis, molecular structure prediction, and recommendation systems. GNNs learn representations
            of nodes by aggregating information from their neighbors.
            
            Generative Adversarial Networks (GANs) consist of a generator and a discriminator trained in an adversarial manner.
            The generator creates synthetic data samples, while the discriminator tries to distinguish between real and synthetic samples.
            GANs have been highly successful in generating realistic images, video, and other types of content."""
        }
    ]
    
    corpus.extend(long_docs)
    
    # Generate more documents if needed to reach num_docs
    while len(corpus) < num_docs:
        title_num = len(corpus) + 1
        corpus.append({
            "title": f"Document {title_num}",
            "text": f"This is sample document {title_num} for testing the RAG system. It contains some text that can be retrieved and used for generating responses."
        })
    
    return corpus[:num_docs]

def create_sample_queries():
    """Create sample queries with ground truth answers"""
    return [
        {
            "question": "What is RAG?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines language models with information retrieval systems to enhance the quality of generated text by retrieving relevant information from a knowledge base."
        },
        {
            "question": "What are the main components of a RAG system?",
            "answer": "A RAG system typically consists of three main components: 1) A Retriever that finds relevant information, 2) A Generator that creates responses, and 3) A Knowledge Base of documents or facts."
        },
        {
            "question": "How does vector search work?",
            "answer": "Vector search converts text into numerical vectors and finds documents with similar vector representations, allowing for semantic searching beyond simple keyword matching."
        },
        {
            "question": "What is document chunking in RAG?",
            "answer": "Document chunking in RAG involves splitting documents into smaller pieces to enable more precise retrieval. Common strategies include fixed-size chunks, paragraph-based, and semantic chunking."
        },
        {
            "question": "What is neural network?",
            "answer": "Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of artificial neurons organized in layers and can learn complex patterns from data through training."
        }
    ]

if __name__ == "__main__":
    run_sample_test()