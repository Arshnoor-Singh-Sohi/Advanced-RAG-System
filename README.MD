# Advanced RAG-Based Knowledge Management System

This repository contains a comprehensive Retrieval-Augmented Generation (RAG) system designed for both practical knowledge management applications and as an experimental platform for advanced RAG research.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Research Paper Generation](#research-paper-generation)
- [Extending the System](#extending-the-system)

## Overview

Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generative approaches to enhance the quality, accuracy, and factual grounding of AI-generated responses. This project provides:

1. A complete end-to-end RAG system for knowledge management
2. An experimentation platform for evaluating different RAG components 
3. Analysis tools for research and paper writing

The system allows for extensive experimentation with different RAG approaches including various document chunking strategies, embedding models, retrieval methods, query processing techniques, reranking mechanisms, and generation prompt engineering.

## Key Features

- **Modular architecture** for easy experimentation with different RAG components
- **Multiple chunking strategies**: fixed-size, paragraph-based, and semantic chunking
- **Various embedding models**: SentenceTransformers, OpenAI, and Hugging Face models
- **Multiple retrieval methods**: Vector search, BM25, and hybrid approaches
- **Advanced query processing**: Query expansion, reformulation, and HyDE
- **Reranking mechanisms**: Cross-encoder, LLM-based, and fusion approaches
- **Comprehensive evaluation metrics**: Precision, recall, MRR, faithfulness, etc.
- **Experiment tracking and visualization tools**
- **Web interface** for interactive usage (Streamlit)
- **Analysis tools** for research paper generation

## Directory Structure

```
advanced_rag_system/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore file
├── run_experiments.py          # End-to-end experiment runner
├── paper_template.md           # IEEE paper template
├── src/
│   ├── components/
│   │   ├── rag_components.py   # Core RAG functionality
│   │   ├── advanced_chunking.py # Enhanced semantic chunking
│   │   ├── enhanced_hyde.py    # Enhanced HyDE implementation
│   │   ├── advanced_reranking.py # Multi-stage reranking
│   │   ├── reranking.py        # Basic reranking implementations
│   │   └── evaluation.py       # Evaluation metrics
│   ├── utils/
│   │   └── experiment_tracker.py # Experiment tracking
│   └── app/
│       ├── rag_app.py          # Demo application
│       └── streamlit_app.py    # Web interface
├── experiments/
│   ├── test_framework.py       # Initial framework test
│   ├── chunking_experiment.py  # Chunking comparisons
│   ├── embedding_experiment.py # Embedding comparisons
│   ├── retrieval_experiment.py # Retrieval method comparisons
│   ├── query_processing_experiment.py
│   ├── reranking_experiment.py # Reranking comparisons
│   └── generation_experiment.py # Generation comparisons
├── analysis/
│   ├── analyze_results.py      # Results analysis
│   └── paper_figures.py        # Paper figure generation
├── data/                       # Directory for data files
│   └── .gitkeep                # Empty file to keep directory in git
└── results/                    # Experiment results
    └── .gitkeep                # Empty file to keep directory in git
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced-rag-system.git
cd advanced-rag-system
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up API keys for external services:
If you want to use OpenAI embeddings or other paid services, set up your API keys:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Demo Application

To run the command-line demo application:

```bash
python -m src.app.rag_app --corpus path/to/your/corpus.pkl
```

### Running the Web Interface

To launch the Streamlit web interface:

```bash
streamlit run src.app.streamlit_app
```

This will open a browser window with the RAG web interface where you can:
- Upload your own documents
- Configure the RAG pipeline
- Ask questions and get answers
- View performance metrics

### Creating a Corpus

You can create a corpus in several ways:

1. **Upload files through the Streamlit interface**
2. **Convert files programmatically**:

```python
from src.components.rag_components import DocumentChunker

# Create corpus from text files
corpus = []
for file_path in text_files:
    with open(file_path, 'r') as f:
        text = f.read()
        corpus.append({
            "title": os.path.basename(file_path),
            "text": text
        })

# Save corpus
import pickle
with open("data/my_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
```

## Experiments

The system includes several experiment scripts for evaluating different aspects of RAG systems:

### Running Individual Experiments

```bash
# Test the framework
python -m experiments.test_framework

# Run chunking experiments
python -m experiments.chunking_experiment --corpus data/my_corpus.pkl --output-dir results

# Run embedding experiments
python -m experiments.embedding_experiment --corpus data/my_corpus.pkl --output-dir results

# Run retrieval experiments
python -m experiments.retrieval_experiment --corpus data/my_corpus.pkl --output-dir results

# Run query processing experiments
python -m experiments.query_processing_experiment --corpus data/my_corpus.pkl --output-dir results

# Run reranking experiments
python -m experiments.reranking_experiment --corpus data/my_corpus.pkl --output-dir results

# Run generation experiments
python -m experiments.generation_experiment --corpus data/my_corpus.pkl --output-dir results
```

### Running All Experiments

To run a comprehensive suite of experiments:

```bash
python run_experiments.py --corpus data/my_corpus.pkl --output-dir results
```

## Research Paper Generation

After running experiments, you can analyze the results and generate figures for your research paper:

```bash
# Analyze results
python -m analysis.analyze_results --results-dir results --output-dir figures

# Generate paper figures
python -m analysis.paper_figures --results-dir results --output-dir paper_figures
```

This will create:
1. Analysis of each component's performance
2. Comparison of different configurations
3. High-quality figures and tables for an IEEE-format paper
4. JSON files with recommendations and insights

## Extending the System

### Adding New Chunking Strategies

Extend the `DocumentChunker` class in `src/components/rag_components.py`:

```python
@staticmethod
def chunk_by_your_method(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Your new chunking strategy"""
    chunked_docs = []
    # Implement your chunking logic
    return chunked_docs
```

### Adding New Embedding Models

Extend the `EmbeddingProvider` class in `src/components/rag_components.py`:

```python
@staticmethod
def get_your_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Get embeddings using your model"""
    # Implement your embedding logic
    return embeddings
```

### Adding New Retrieval Methods

Extend the `RetrievalMethods` class in `src/components/rag_components.py`:

```python
@staticmethod
def your_search(query: str, ...) -> List[Tuple[str, float]]:
    """Your retrieval method"""
    # Implement your retrieval logic
    return results
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```
@article{your_name2025advanced,
  title={Advanced RAG-Based Knowledge Management System and Medical Diagnostic Support Application},
  author={Your Name},
  journal={IEEE Conference on Neural Networks and Deep Learning},
  year={2025}
}
```

## Acknowledgments

This project leverages the following open-source libraries:
- LangChain
- Sentence-Transformers
- FAISS
- HuggingFace Transformers
- Streamlit