"""
Streamlit Web Interface for RAG Application

This module provides a web interface for the RAG application using Streamlit.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import sys
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import base64
from io import BytesIO
import traceback
import matplotlib as mpl

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import RAG components
try:
    from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods, QueryProcessor
    from src.components.reranking import RerankerModule
    from src.components.evaluation import RAGEvaluator
    from src.app.rag_app import RAGApplication
except ImportError as e:
    st.error(f"Failed to import necessary RAG components: {e}. Please ensure src is in the Python path.")
    st.stop() # Stop execution if core components can't be imported



# Set page config
st.set_page_config(
    page_title="Advanced RAG Knowledge Management System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS styles - simplified and compatible with dark theme
st.markdown("""
<style>
    /* Basic styles */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Content boxes */
    .content-box {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(250, 250, 250, 0.1);
    }
    .info-box {
        background-color: rgba(66, 165, 245, 0.1);
        border-left: 3px solid #42A5F5;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 3px solid #FF9800;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 3px solid #F44336;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Chat elements */
    .user-message {
        background-color: rgba(66, 165, 245, 0.1);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .assistant-message {
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 3px solid #FFC107;
        border-radius: 4px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    /* Navigation */
    .nav-item {
        padding: 0.5rem;
        margin-bottom: 0.3rem;
        border-radius: 5px;
    }
    .nav-active {
        background-color: rgba(66, 165, 245, 0.2);
        font-weight: 600;
    }
    
    /* Separator */
    .separator {
        margin: 1.5rem 0;
        border-top: 1px solid rgba(250, 250, 250, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_app" not in st.session_state:
    st.session_state.rag_app = None
    
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"
    
if "corpus_uploaded" not in st.session_state:
    st.session_state.corpus_uploaded = False
    
if "retrieval_metrics" not in st.session_state:
    st.session_state.retrieval_metrics = None
    
LOADED_CONFIG = None
CONFIG_SAVE_PATH = "user_config.json"
# Add os import if not already at the top
import os
if os.path.exists(CONFIG_SAVE_PATH):
    try:
        # Add json import if not already at the top
        import json
        with open(CONFIG_SAVE_PATH, 'r') as f:
            LOADED_CONFIG = json.load(f)
        print(f"Loaded configuration from {CONFIG_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading configuration from {CONFIG_SAVE_PATH}: {e}")
        LOADED_CONFIG = None
# --- End Load ---

# Define Default Config Structure (with updated defaults from prompt)
DEFAULT_CONFIG = {
    "corpus_path": "data/corpus.pkl", "index_path": "data/faiss_index",
    "chunking_strategy": "recursive", # Default changed
    "chunk_size": 1000,               # Default changed
    "chunk_overlap": 200,             # Default changed
    "embedding_model": "all-MiniLM-L6-v2", # Default "MiniLM" equivalent
    "retrieval_method": "hybrid", "retrieval_alpha": 0.7,
    "top_k": 4,                       # Default changed (assuming retrieval_k=4)
    "query_expansion": False, "expansion_method": "simple",
    "use_hyde": False, "hyde_method": "basic", "hyde_num_docs": 3,
    "use_reranking": True, "reranking_method": "cross_encoder",
    "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "reranking_stages": ["semantic", "cross_encoder", "diversity"],
    "diversity_alpha": 0.7,
    # Ensure prompt template is correctly formatted
    "prompt_template": "Answer the question based ONLY on the following context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
    "temperature": 0.5,
    # --- MMR Defaults ---
    "use_mmr": False,                 # Default changed
    "mmr_fetch_k": 20,                # Default changed
    "mmr_lambda": 0.5,                # Default changed (key is mmr_lambda)
    # Add other keys like bm25_k1, bm25_b if used in UI
    "bm25_k1": 1.5, # Example default
    "bm25_b": 0.75   # Example default
}

# Initialize session state config (Merge loaded with defaults)
if "config" not in st.session_state:
    temp_config = DEFAULT_CONFIG.copy() # Start with defaults
    if LOADED_CONFIG:
        # Update defaults with any values found in the loaded file
        # This merges, keeping defaults for keys missing in the loaded file
        temp_config.update(LOADED_CONFIG)
        print("Applied loaded configuration over defaults.")
    st.session_state.config = temp_config # Assign merged config

# --- Ensure ALL keys from DEFAULT_CONFIG exist in session_state.config ---
# This handles cases where a saved config is loaded but is missing newer default keys
for key, default_value in DEFAULT_CONFIG.items():
    if key not in st.session_state.config:
        print(f"Adding missing key '{key}' to config with default value.")
        st.session_state.config[key] = default_value

# Initialize previous_config based on the final initial config
if "previous_config" not in st.session_state:
    st.session_state.previous_config = st.session_state.config.copy()

if 'response_mode' not in st.session_state:
    # Default value might depend on loaded config or standard default
    st.session_state.response_mode = st.session_state.config.get("response_mode", "LLM-Enhanced (OpenAI)") # Use LLM as new default?

if 'enable_comparison' not in st.session_state:
    st.session_state.enable_comparison = st.session_state.config.get("enable_comparison", False)

if 'temperature' not in st.session_state:
    # Sync sidebar temperature slider state with config
    st.session_state.temperature = st.session_state.config.get("temperature", 0.5)

# --- START OF ADDED HELPER FUNCTION ---
def calculate_lexical_diversity(text):
    """Calculate lexical diversity (ratio of unique words to total words)"""
    words = text.lower().split()
    if not words: # Handle empty text
        return 0.0
    return len(set(words)) / len(words) # Simplified from max(len(words), 1) as len(words) > 0 here
# --- END OF ADDED HELPER FUNCTION ---

# Functions for file operations
def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk"""
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def convert_file_to_corpus(file_path):
    """Convert an uploaded file to a corpus format"""
    _, file_ext = os.path.splitext(file_path)
    corpus = []
    
    # Process based on file type
    if file_ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split by double newlines for paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            corpus.append({
                "title": f"Paragraph_{i+1}",
                "text": para
            })
    
    elif file_ext.lower() == ".pdf":
        try:
            import PyPDF2
            
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Process each page
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        # Split long pages into manageable chunks
                        if len(text) > 2000:
                            # Split by paragraphs or similar logical breaks
                            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                            for j, chunk in enumerate(chunks):
                                corpus.append({
                                    "title": f"Page_{i+1}_Section_{j+1}",
                                    "text": chunk
                                })
                        else:
                            corpus.append({
                                "title": f"Page_{i+1}",
                                "text": text
                            })
            
            # If no text was extracted, inform the user
            if not corpus:
                st.warning("No text could be extracted from the PDF. It may be image-based or protected.")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None, []
    
    elif file_ext.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        # Assume the CSV has at least two columns: title and text
        if "title" in df.columns and "text" in df.columns:
            for _, row in df.iterrows():
                corpus.append({
                    "title": row["title"],
                    "text": row["text"]
                })
        else:
            # Use the first column as title and second as text
            cols = df.columns
            for _, row in df.iterrows():
                corpus.append({
                    "title": str(row[cols[0]]),
                    "text": str(row[cols[1]])
                })
                
    elif file_ext.lower() in [".json", ".jsonl"]:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_ext.lower() == ".json":
                # Assume JSON is a list of documents
                documents = json.load(f)
                if isinstance(documents, list):
                    for doc in documents:
                        if isinstance(doc, dict) and "text" in doc:
                            corpus.append({
                                "title": doc.get("title", f"Document_{len(corpus)+1}"),
                                "text": doc["text"]
                            })
            else:
                # JSONL: each line is a JSON document
                for line in f:
                    try:
                        doc = json.loads(line)
                        if isinstance(doc, dict) and "text" in doc:
                            corpus.append({
                                "title": doc.get("title", f"Document_{len(corpus)+1}"),
                                "text": doc["text"]
                            })
                    except:
                        continue
    
    # Save corpus to pickle file
    corpus_path = os.path.join("data", "corpus.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    
    return corpus_path, corpus

# Initialize the RAG Application
def initialize_rag_app():
    """Initialize or update the RAG application with current configuration"""
    if st.session_state.rag_app is None:
        st.session_state.rag_app = RAGApplication()
        
    # Update configuration
    st.session_state.rag_app.config = st.session_state.config

# Function to process a query
def process_query(query):
    """Process a query and update conversation history"""
    if st.session_state.rag_app is None:
        initialize_rag_app()
        
    # Measure time for performance metrics
    start_time = time.time()
    
    # Process the query
    answer, contexts = st.session_state.rag_app.process_query(query)
    
    # Calculate retrieval time
    retrieval_time = time.time() - start_time
    
    # Update conversation history
    st.session_state.conversation_history.append({
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "time": retrieval_time
    })
    
    # Calculate and save retrieval metrics
    if len(contexts) > 0:
        # For demo purposes, we'll use a simple relevance score
        relevance_scores = []
        for doc in contexts:
            # Simple word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / (len(query_words) + len(doc_words) - overlap) if doc_words else 0
            relevance_scores.append(score)
            
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        st.session_state.retrieval_metrics = {
            "retrieval_time": retrieval_time,
            "num_documents": len(contexts),
            "avg_relevance": avg_relevance
        }
    
    return answer, contexts

# Sidebar navigation
def sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.markdown("## 🧠 Advanced RAG System")
        
        # Navigation
        st.markdown("### Navigation")
        
        # Create custom navigation with icons
        nav_options = {
            "Chat": "💬 Chat",
            "Configuration": "⚙️ Configuration",
            "Metrics": "📊 Metrics",
            "Experiment Lab": "🧪 Experiment Lab",
            "About": "ℹ️ About"
        }
        
        # Highlight the current page in the navigation
        for page, label in nav_options.items():
            if st.session_state.current_page == page:
                st.markdown(f'<div class="nav-item nav-active">{label}</div>', unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Show additional settings based on current page
        if st.session_state.current_page == "Chat":
            st.markdown("---")
            st.markdown("### Knowledge Base")
            
            corpus_option = st.radio(
                "Knowledge Source",
                ["Upload Documents", "Use Example Dataset"],
                key="corpus_option_radio",
                horizontal=True
            )
            
            if corpus_option == "Upload Documents":
                uploaded_file = st.file_uploader(
                    "Upload your knowledge base file",
                    type=["txt", "csv", "json", "jsonl", "pkl", "pdf"]
                )
                
                if uploaded_file is not None:
                    if st.button("Process Uploaded File", key="process_file_button"):
                        with st.spinner("Processing file..."):
                            # Save the uploaded file
                            file_path = save_uploaded_file(uploaded_file)
                            
                            # Convert to corpus if needed
                            if file_path.endswith(".pkl"):
                                corpus_path = file_path
                            else:
                                corpus_path, corpus = convert_file_to_corpus(file_path)
                                
                            # Update configuration
                            st.session_state.config["corpus_path"] = corpus_path
                            
                            # Clear any existing RAG app
                            st.session_state.rag_app = None
                            
                            # Initialize new RAG app with the updated config
                            initialize_rag_app()
                            
                            # Prepare the knowledge base
                            st.session_state.rag_app.prepare_knowledge_base(force_rebuild=True)
                            
                            st.session_state.corpus_uploaded = True
                            st.success("File processed successfully!")
                            
            else:
                if st.button("Load Example Dataset", key="load_example_button"):
                    with st.spinner("Loading example dataset..."):
                        # Create example dataset if it doesn't exist
                        os.makedirs("data", exist_ok=True)
                        example_path = os.path.join("data", "example_corpus.pkl")
                        
                        if not os.path.exists(example_path):
                            # Create a simple example corpus
                            example_corpus = [
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
                                }
                            ]
                            
                            with open(example_path, "wb") as f:
                                pickle.dump(example_corpus, f)
                                
                        # Update configuration
                        st.session_state.config["corpus_path"] = example_path
                        
                        # Clear any existing RAG app
                        st.session_state.rag_app = None
                        
                        # Initialize new RAG app with the updated config
                        initialize_rag_app()
                        
                        # Prepare the knowledge base
                        st.session_state.rag_app.prepare_knowledge_base(force_rebuild=True)
                        
                        st.session_state.corpus_uploaded = True
                        st.success("Example dataset loaded!")
                        st.rerun()

            st.markdown("---")
            st.markdown("### Response Settings")
            
            # Get current value from radio button
            response_mode_options = ["Extractive (Basic Retrieval)", "LLM-Enhanced (OpenAI)"]
            current_response_mode_index = response_mode_options.index(st.session_state.response_mode)
            
            selected_response_mode = st.radio(
                "Response Generation Method",
                response_mode_options,
                index=current_response_mode_index,
                key='response_mode_radio',
                horizontal=True
            )
            # Update session state ONLY if the selection changed
            if selected_response_mode != st.session_state.response_mode:
                st.session_state.response_mode = selected_response_mode
                st.rerun()
            
            enable_comparison_checkbox = st.checkbox(
                "Enable Side-by-Side Comparison",
                value=st.session_state.enable_comparison,
                key='comparison_checkbox',
                help="Generate responses using both methods for comparison"
            )
            # Update session state if checkbox value changes
            if enable_comparison_checkbox != st.session_state.enable_comparison:
                st.session_state.enable_comparison = enable_comparison_checkbox
                st.rerun()

            # --- LLM SETTINGS SECTION ---
            if st.session_state.response_mode == "LLM-Enhanced (OpenAI)" or st.session_state.enable_comparison:
                st.markdown("---")
                st.markdown("### LLM Settings")

                # Temperature Slider
                current_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.05,
                    key='temperature_slider',
                    help="Controls randomness. Higher values = more creative/random, Lower values = more deterministic/focused."
                )

                # Update session state if the slider value changes
                if current_temperature != st.session_state.temperature:
                    st.session_state.temperature = current_temperature
            # --- END LLM SETTINGS SECTION ---
            
        elif st.session_state.current_page == "Configuration":
            st.markdown("---")
            st.markdown("### Help")
            st.info(
                "Configure your RAG system settings in the main panel. "
                "Changes will apply to new queries."
            )
            
        elif st.session_state.current_page == "Experiment Lab":
            st.markdown("---")
            st.markdown("### Experiment Controls")
            st.info(
                "Run experiments to compare different RAG configurations. "
                "Results will be shown in the main panel."
            )

# Page: Chat Interface
def chat_page():
    """Chat interface page with input fixed at the bottom."""
    st.markdown(
        """
        <p style="
            font-size: 3rem;         /* Increase font size (adjust '3rem' as needed) */
            font-weight: 600;        /* Make it bolder */
            text-align: center;      /* Center the text */
            margin-bottom: 1rem;     /* Add some space below */
            padding-bottom: 0.5rem;  /* Keep padding if desired */
            /* Optional: Keep border if you liked it from 'main-header' */
            /* border-bottom: 1px solid rgba(250, 250, 250, 0.2); */
        ">
            💬 Advanced RAG Chat Interface
        </p>
        """,
        unsafe_allow_html=True
    )

    # --- Initial Checks ---
    # Check if corpus is loaded
    if not st.session_state.get("corpus_uploaded", False):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ No knowledge base loaded. Please upload documents or load the example dataset from the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome!
        To get started:
        1.  Use the sidebar to **Upload Documents** or **Load Example Dataset**.
        2.  Once processed, you can ask questions below!
        """)
        return # Stop rendering the rest of the chat page

    # Initialize RAG app if needed
    if "rag_app" not in st.session_state or st.session_state.rag_app is None:
        st.markdown('<div class="info-box">Initializing RAG application...</div>', unsafe_allow_html=True)
        try:
            initialize_rag_app() # Assuming this function initializes and stores in st.session_state.rag_app
            if st.session_state.rag_app is None: # Check if initialization actually failed
                 st.error("Failed to initialize RAG Application.")
                 return
            # We need to rerun *after* initialization for the page to proceed correctly
            st.rerun()
        except Exception as e:
            st.error(f"Fatal error during RAG initialization: {e}")
            st.exception(e)
            return


    # --- Display existing chat messages FIRST ---
    # Loop through the history stored in session state
    for i, message in enumerate(st.session_state.get("conversation_history", [])):
        role = message.get("role")

        # Infer role for backward compatibility if 'role' key is missing
        if not role:
            if "query" in message and ("answer" in message or "extractive_answer" in message):
                 # If it has query AND answer fields, display user query then assistant response separately
                 with st.chat_message("user"):
                      st.markdown(message.get("query", "*Query missing*"))
                 role = "assistant" # Now process the assistant part
            elif "query" in message:
                 role = "user"
            elif "answer" in message or "extractive_answer" in message:
                 role = "assistant"
            else:
                 role = "system" # Or skip if format is unknown

        # Display message based on inferred or stored role
        if role == "user":
             with st.chat_message("user"):
                  st.markdown(message.get("query", "*Query missing*"))
        elif role == "assistant":
            with st.chat_message("assistant"):
                is_error = message.get("is_error", False)
                if is_error:
                    st.error(f"{message.get('answer', 'An unspecified error occurred.')}")

                # --- Comparison View ---
                elif "extractive_answer" in message and "llm_answer" in message:
                    st.markdown("**Responses:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("###### Extractive")
                        st.markdown(message.get('extractive_answer', 'N/A'))
                    with col2:
                        st.markdown("###### LLM-Enhanced")
                        st.markdown(message.get('llm_answer', 'N/A'))

                    # --- Metrics for Comparison ---
                    st.markdown("---")
                    st.markdown("###### Response Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    extractive_ans_text = message.get('extractive_answer', '')
                    llm_ans_text = message.get('llm_answer', '')
                    extractive_len = len(extractive_ans_text.split())
                    llm_len = len(llm_ans_text.split())
                    lex_div = calculate_lexical_diversity(llm_ans_text)
                    try: diff = ((llm_len - extractive_len) / extractive_len) * 100 if extractive_len > 0 else 0; delta_val = f"{diff:.1f}%"
                    except ZeroDivisionError: delta_val = None

                    metric_col1.metric("Extractive Words", extractive_len)
                    metric_col2.metric("LLM Words", llm_len, delta=delta_val if delta_val else None, help="Change vs Extractive")
                    metric_col3.metric("LLM Diversity", f"{lex_div:.3f}", help="Unique/Total words (LLM)")
                    # --- End Metrics ---

                # --- Single Answer View ---
                elif "answer" in message:
                    st.markdown(message["answer"])
                else:
                    st.warning("Could not display assistant response (unknown format).")

                # --- Sources Display ---
                if not is_error and message.get("contexts"):
                    with st.expander(f"Show {len(message['contexts'])} Sources", expanded=False):
                        for j, doc in enumerate(message["contexts"]):
                            source_title = doc.get("title", f"Source {j+1}")
                            text_preview = doc.get('text', 'N/A')
                            score = doc.get('score') # If score is available
                            st.markdown('<div class="source-box">', unsafe_allow_html=True)
                            st.markdown(f"**{source_title}**" + (f" (Score: {score:.3f})" if score else ""))
                            st.caption(f"{text_preview[:250]}..." if len(text_preview) > 250 else text_preview)
                            st.markdown('</div>', unsafe_allow_html=True)
                elif not is_error:
                    st.caption("_No sources were retrieved._")

                # --- Timestamp and Time ---
                time_taken = message.get("time", 0)
                timestamp = message.get("timestamp")
                if timestamp: st.caption(f"_Processed in {time_taken:.2f}s on {pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')}_")
                elif time_taken > 0: st.caption(f"_Processed in {time_taken:.2f}s_")

        elif role == "system": # Handle system messages or unknown formats if needed
             with st.chat_message("system"):
                  st.warning("Unknown message format in history.")


    # --- Chat Input Widget (Stays at the bottom) ---
    if prompt := st.chat_input("Ask your question here...", key="chat_main_input"):

        # 1. Add user message to history and display it immediately
        st.session_state.conversation_history.append({"role": "user", "query": prompt})
        # Rerun isn't strictly needed here just to show the user message,
        # but the whole block will rerun after processing anyway.

        # 2. Process the query (using a spinner for feedback)
        with st.spinner("Thinking..."):
            if st.session_state.rag_app:
                start_time = time.time()
                try:
                    # Determine response modes needed
                    modes_to_run = []
                    if st.session_state.get("enable_comparison", False):
                        modes_to_run = ["extractive", "llm"]
                    elif st.session_state.get("response_mode") == "Extractive (Basic Retrieval)":
                        modes_to_run = ["extractive"]
                    else: # Default to LLM-Enhanced
                        modes_to_run = ["llm"]

                    results = {}
                    contexts = None
                    # llm_temp = st.session_state.get("temperature", 0.5) # Get current temp

                    for mode in modes_to_run:
                        st.session_state.rag_app.response_mode = mode
                        llm_kwargs = {}
                        # if mode == "llm": llm_kwargs['temperature'] = llm_temp

                        # --- THE CORE RAG CALL ---
                        answer, current_contexts = st.session_state.rag_app.process_query(prompt)
                        # --- END CORE RAG CALL ---

                        if mode == "extractive": results["extractive_answer"] = answer
                        else: results["llm_answer"] = answer
                        if contexts is None: contexts = current_contexts

                    total_time = time.time() - start_time

                    # 3. Construct assistant response for history
                    assistant_entry = {
                        "role": "assistant", "contexts": contexts or [],
                        "time": total_time, "timestamp": time.time()
                    }
                    if st.session_state.get("enable_comparison", False):
                        assistant_entry["extractive_answer"] = results.get("extractive_answer", "*Error*")
                        assistant_entry["llm_answer"] = results.get("llm_answer", "*Error*")
                    elif "extractive_answer" in results: assistant_entry["answer"] = results["extractive_answer"]
                    elif "llm_answer" in results: assistant_entry["answer"] = results["llm_answer"]
                    else: assistant_entry["answer"] = "*Error: No response generated.*"; assistant_entry["is_error"] = True

                    st.session_state.conversation_history.append(assistant_entry)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.conversation_history.append({
                        "role": "assistant", "answer": f"Error: {e}",
                        "is_error": True, "time": time.time() - start_time, "timestamp": time.time()
                    })
                    traceback.print_exc()
            else:
                st.error("RAG system not ready.")
                st.session_state.conversation_history.append({
                    "role": "assistant", "answer": "Error: RAG system not ready.", "is_error": True
                })

        # 4. Rerun to display the new user message and the processed assistant response
        st.rerun()

# Page: Configuration
def configuration_page():
    """Configuration page with enhanced RAG parameter controls"""
    st.markdown('<p class="main-header">⚙️ RAG System Configuration</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("Configure your RAG system with the settings below. Changes will apply to new queries.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different configuration categories
    tabs = st.tabs([
        "📚 Knowledge Base", 
        "✂️ Chunking", 
        "🔢 Embedding", 
        "🔍 Retrieval", 
        "❓ Query Processing", 
        "🔄 Reranking",
        "💬 Generation"
    ])
    
    # Tab: Knowledge Base
    with tabs[0]:
        st.markdown('<p class="sub-header">Knowledge Base Configuration</p>', unsafe_allow_html=True)
        
        # Display current corpus info
        if st.session_state.corpus_uploaded:
            try:
                with open(st.session_state.config["corpus_path"], "rb") as f:
                    corpus = pickle.load(f)
                    
                st.markdown('<div class="content-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Current Knowledge Base:** {os.path.basename(st.session_state.config['corpus_path'])}")
                with col2:
                    st.markdown(f"**Number of Documents:** {len(corpus)}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show sample documents
                st.markdown("**Sample Documents:**")
                for i, doc in enumerate(corpus[:3]):
                    st.markdown('<div class="source-box">', unsafe_allow_html=True)
                    st.markdown(f"**{doc.get('title', f'Document {i+1}')}**")
                    st.markdown(f"{doc['text'][:100]}..." if len(doc['text']) > 100 else doc['text'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except:
                st.warning("Error loading corpus information.")
        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("No knowledge base loaded. Please upload documents or load the example dataset from the sidebar.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Advanced options
        st.markdown("### Advanced Options")
        
        # Index path
        index_path = st.text_input(
            "Index Path", 
            value=st.session_state.config["index_path"],
            help="Path where the vector index will be stored"
        )
        st.session_state.config["index_path"] = index_path
    
    # Tab: Chunking - ENHANCED
    with tabs[1]:
        st.markdown('<p class="sub-header">Chunking Configuration</p>', unsafe_allow_html=True)

        # --- Unified Strategy Selection ---
        all_strategies = [
            "fixed",          # From paste.txt
            "recursive",      # From request
            "token",          # From request
            "sentence",       # From request
            "paragraph",      # From paste.txt
            "semantic"        # From paste.txt
        ]
        # Get current strategy from config, default safely
        current_strategy = st.session_state.config.get("chunking_strategy", all_strategies[0])
        try:
            current_index = all_strategies.index(current_strategy)
        except ValueError:
            st.warning(f"Configured strategy '{current_strategy}' not in list. Defaulting.")
            current_index = 0
            current_strategy = all_strategies[0] # Reset to default

        # Combined selectbox for all strategies
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_strategy = st.selectbox(
                "Chunking Strategy",
                all_strategies,
                index=current_index,
                key="chunking_strategy_select", # Added key
                help="Method used to split documents into chunks. Affects how size/overlap are used."
            )
            # Update config immediately
            st.session_state.config["chunking_strategy"] = selected_strategy

        # --- Unified Help Expander ---
        with col2:
            st.markdown('<div style="padding-top: 25px">', unsafe_allow_html=True)
            with st.expander("About Strategies"):
                st.markdown("""
                - **Fixed**: Splits by token count with optional overlap. Simple but can break sentences.
                - **Recursive**: Splits recursively by characters (e.g., `\\n\\n`, `\\n`, `.`, ` `), trying to keep paragraphs/sentences intact first. Often a good default.
                - **Token**: Splits based on token count using a specific tokenizer (e.g., for an LLM). Consistent chunk sizes for model context.
                - **Sentence**: Splits by sentence boundaries (`.`, `!`, `?`). Preserves sentence meaning.
                - **Paragraph**: Splits by paragraph boundaries (often `\\n\\n`). Preserves document structure.
                - **Semantic**: Uses AI models (e.g., embeddings) to split based on meaning/topic shifts. Potentially better coherence but slower.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Conditional Size/Overlap Sliders ---
        # Show for strategies where size/overlap are primary parameters
        if selected_strategy in ["fixed", "recursive", "token"]:
            st.markdown("### Chunk Size & Overlap")
            col1, col2 = st.columns(2)

            with col1:
                # Use wider range from the second request
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=100,   # From request
                    max_value=2000,  # From request
                    # Read from config, use request's default if not set
                    value=st.session_state.config.get("chunk_size", 1000),
                    step=100,        # From request
                    key="chunk_size_slider", # Added key
                    help="Target maximum size for each text chunk (often in tokens)."
                )
                st.session_state.config["chunk_size"] = chunk_size # Update config

            with col2:
                # Ensure overlap max value depends on current chunk_size
                max_overlap_value = chunk_size // 2 # Sensible limit
                # Read from config, use request's default if not set, ensure within bounds
                default_overlap = min(st.session_state.config.get("chunk_overlap", 200), max_overlap_value)

                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=max_overlap_value, # Dynamic max based on chunk size
                    value=default_overlap,
                    step=50,         # From request
                    key="chunk_overlap_slider", # Added key
                    help="Number of overlapping units (e.g., tokens) between consecutive chunks."
                )
                st.session_state.config["chunk_overlap"] = chunk_overlap # Update config

            # --- Visualization (Conditional) ---
            st.markdown("### Chunk Size and Overlap Visualization")
            try:
                visualization_cols = st.columns([1, 6, 1])
                with visualization_cols[1]:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    doc_length = max(2000, chunk_size * 2.5) # Adjust viz length based on chunk size
                    step = chunk_size - chunk_overlap if chunk_size > chunk_overlap else chunk_size # Prevent division by zero/negative step
                    if step <= 0: step = 1 # Ensure step is positive
                    num_chunks = int((doc_length - chunk_overlap) // step) + 1
                    num_chunks = min(num_chunks, 10) # Limit chunks shown for clarity

                    for i in range(num_chunks):
                        start = i * step
                        end = start + chunk_size
                        rect = plt.Rectangle((start, 0), chunk_size, 1, fill=True, alpha=0.5,
                                            color='blue', linewidth=1, edgecolor='black')
                        ax.add_patch(rect)

                    ax.set_xlim(0, start + chunk_size if num_chunks > 0 else chunk_size) # Adjust xlim based on drawn chunks
                    ax.set_ylim(0, 1.5)
                    ax.set_yticks([])
                    ax.set_xlabel("Document position (units)")
                    ax.set_title("Chunking Visualization")

                    # Add labels only if chunks are drawn
                    if num_chunks > 0:
                        plt.annotate(f"Size: {chunk_size}", (ax.get_xlim()[1]*0.5, 1.3), ha='center', va='center', fontsize=10)
                        if chunk_overlap > 0 and num_chunks > 1: # Only show overlap if >1 chunk and overlap > 0
                            overlap_pos_x = step + chunk_overlap / 2
                            # Adjust annotation pos if it goes off chart
                            if overlap_pos_x > ax.get_xlim()[1]: overlap_pos_x = ax.get_xlim()[1] * 0.9
                            plt.annotate(f"Overlap: {chunk_overlap}", (overlap_pos_x, 1.1), ha='center', va='center', fontsize=9, arrowprops=dict(arrowstyle='->'))

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig) # Close plot
            except Exception as e:
                st.warning(f"Could not generate chunking visualization: {e}")

        # --- Conditional Strategy Information ---
        st.markdown("### Strategy Details")
        if selected_strategy == "fixed":
            st.info("Splits text based strictly on the 'Chunk Size' (e.g., token count). Simple but can cut sentences mid-way.")
        elif selected_strategy == "recursive":
            st.info("Recursively splits text using a list of separators (e.g., paragraph, sentence, word). Aims to keep meaningful blocks together. Often a good starting point.")
        elif selected_strategy == "token":
            st.info("Splits text based on token count according to a specific model's tokenizer. Ensures chunks fit model context windows well.")
        elif selected_strategy == "sentence":
            st.info("Splits text at sentence boundaries (e.g., '.', '!', '?'). Preserves sentence integrity but chunk sizes can vary greatly.")
        elif selected_strategy == "paragraph":
            st.info("Splits text at paragraph breaks (usually '\\n\\n'). Preserves document structure well if formatting is consistent.")
        elif selected_strategy == "semantic":
            st.info("Uses AI (e.g., embedding similarities) to find semantic boundaries. Aims for coherent chunks but is computationally more expensive and experimental.")

            # Keep Advanced Semantic Chunking Options
            st.markdown("#### Advanced Semantic Options")
            advanced_semantic = st.checkbox("Enable advanced semantic options", value=st.session_state.config.get("advanced_semantic", False), key="advanced_semantic_chk")
            st.session_state.config["advanced_semantic"] = advanced_semantic

            if advanced_semantic:
                semantic_methods = ["basic_similarity", "semantic_segmentation", "topic_modeling"] # Example methods
                default_sem_method = st.session_state.config.get("semantic_method", semantic_methods[0])
                try: sem_idx = semantic_methods.index(default_sem_method)
                except ValueError: sem_idx = 0

                semantic_method = st.selectbox(
                    "Semantic Chunking Method",
                    semantic_methods,
                    index=sem_idx,
                    key="semantic_method_select",
                    help="Specific algorithm for semantic chunking."
                )
                st.session_state.config["semantic_method"] = semantic_method

                if semantic_method in ["semantic_segmentation", "topic_modeling"]:
                    st.warning("These advanced methods may significantly slow down processing.")

    # Tab: Embedding - ENHANCED
    with tabs[2]:
        st.markdown('<p class="sub-header">Embedding Configuration</p>', unsafe_allow_html=True)

        # --- Embedding Model Selection (Retained Categorized Approach) ---
        st.markdown("### Embedding Model Selection")

        # Group embedding models by type (Keep your existing categories)
        embedding_categories = {
            "Lightweight (Fast)": ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
            "Balanced": ["all-mpnet-base-v2"],
            "Specialized": ["multi-qa-mpnet-base-dot-v1"],
            "Commercial API": ["text-embedding-ada-002"]
            # Add other models your RAGApplication supports here
        }

        # Category selection using st.radio (Keep)
        embedding_category = st.radio(
            "Model Category",
            list(embedding_categories.keys()),
            horizontal=True,
            key="embedding_category_radio" # Added key
        )

        # Get current model from config
        current_model = st.session_state.config.get("embedding_model", embedding_categories[embedding_category][0]) # Safer default

        # Find default selection index within the selected category's list
        default_index = 0
        try:
            # Find the index of the current_model within the list for the selected category
            default_index = embedding_categories[embedding_category].index(current_model)
        except ValueError:
            # If current_model is not in the selected category (e.g., category just changed),
            # default to the first model in the new category.
            default_index = 0
            # Optionally, you could update the config here, but it will update below anyway.
            # st.session_state.config["embedding_model"] = embedding_categories[embedding_category][0]

        # Model selection using st.selectbox (Keep, ensure correct options/index)
        selected_model = st.selectbox(
            "Select Embedding Model:", # Changed label slightly
            embedding_categories[embedding_category],
            index=default_index, # Use the correctly calculated index
            key="embedding_model_select", # Added key
            help="Model used to generate vector embeddings. Changing this requires rebuilding the KB."
        )

        # --- Update Configuration ---
        # Update the main config dictionary. The "Apply" button handles the rest.
        st.session_state.config["embedding_model"] = selected_model

        # --- Display Model Information (Retained Detailed Approach) ---
        model_info = {
            "all-MiniLM-L6-v2": { "dimensions": 384, "speed": "Fast", "quality": "Good", "description": "Lightweight general-purpose model..." },
            "all-mpnet-base-v2": { "dimensions": 768, "speed": "Medium", "quality": "Excellent", "description": "Higher quality general-purpose model..." },
            "multi-qa-mpnet-base-dot-v1": { "dimensions": 768, "speed": "Medium", "quality": "Excellent for QA", "description": "Optimized for question-answering..." },
            "BAAI/bge-small-en-v1.5": { "dimensions": 384, "speed": "Fast", "quality": "Very Good", "description": "Efficient model for retrieval tasks..." },
            "text-embedding-ada-002": { "dimensions": 1536, "speed": "Varies (API)", "quality": "Very Good", "description": "OpenAI model, requires API key & costs." }
            # Add info for any other models in embedding_categories
        }

        st.markdown("### Model Information") # Add sub-header
        if selected_model in model_info:
            info = model_info[selected_model]
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Dimensions", info["dimensions"])
            with col2: st.metric("Speed", info["speed"])
            with col3: st.metric("Quality", info["quality"])
            st.markdown(f"**Description:** {info['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("*(Select a model to see its information)*")

        # --- Handle OpenAI API Key Input (Merged Logic) ---
        if selected_model == "text-embedding-ada-002":
            st.markdown("#### OpenAI API Key")
            # Added warning about environment variables
            st.warning("Recommended: Set the `OPENAI_API_KEY` environment variable instead of entering the key here.", icon="⚠️")

            # Use config dictionary for value and storage, matching the second request's pattern
            openai_api_key = st.text_input(
                "OpenAI API Key (Optional)",
                type="password",
                # Read from config dict, provide empty string if not found
                value=st.session_state.config.get("openai_api_key", ""),
                key="openai_api_key_input", # Added key
                help="Required if env var not set. Stored temporarily in session config."
            )

            # Update the config dict if a value is provided in the UI
            if openai_api_key:
                st.session_state.config["openai_api_key"] = openai_api_key
            elif "openai_api_key" in st.session_state.config and not openai_api_key:
                # If the field is cleared, remove the key from the config dict
                # This prevents accidentally using an old key if the user clears the input
                del st.session_state.config["openai_api_key"]
    
    # Tab: Retrieval - ENHANCED
    with tabs[3]:
        st.markdown('<p class="sub-header">Retrieval Configuration</p>', unsafe_allow_html=True)
        
        # Main retrieval settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Retrieval method
            retrieval_method = st.selectbox(
                "Retrieval Method",
                ["vector", "bm25", "hybrid"],
                index=["vector", "bm25", "hybrid"].index(st.session_state.config["retrieval_method"]),
                help="Method used to retrieve documents"
            )
            st.session_state.config["retrieval_method"] = retrieval_method
            
        with col2:
            # Top-k
            top_k = st.slider(
                "Number of Retrieved Documents (Top-K)",
                min_value=1,
                max_value=20,
                value=st.session_state.config["top_k"],
                step=1,
                help="Number of documents to retrieve for each query"
            )
            st.session_state.config["top_k"] = top_k
        
        # Advanced retrieval settings
        st.markdown("### Advanced Retrieval Settings")
        
        # Hybrid alpha (only for hybrid method)
        if retrieval_method == "hybrid":
            col1, col2 = st.columns([3, 2])
            
            with col1:
                retrieval_alpha = st.slider(
                    "Vector Search Weight (Alpha)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.config["retrieval_alpha"],
                    step=0.1,
                    help="Weight of vector search (1-Alpha for BM25)"
                )
                st.session_state.config["retrieval_alpha"] = retrieval_alpha
            
            with col2:
                st.markdown('<div style="padding-top: 40px">', unsafe_allow_html=True)
                st.info(f"BM25 Weight: {1-retrieval_alpha:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a visualization of weight distribution
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh(["Weights"], [retrieval_alpha], color='blue', alpha=0.6, label='Vector')
            ax.barh(["Weights"], [1-retrieval_alpha], left=[retrieval_alpha], color='green', alpha=0.6, label='BM25')
            
            # Add labels
            ax.text(retrieval_alpha/2, 0, f"Vector: {retrieval_alpha:.1f}", 
                    ha='center', va='center', color='white' if retrieval_alpha > 0.3 else 'black')
            ax.text(retrieval_alpha + (1-retrieval_alpha)/2, 0, f"BM25: {1-retrieval_alpha:.1f}", 
                    ha='center', va='center', color='white' if (1-retrieval_alpha) > 0.3 else 'black')
            
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Advanced BM25 options    
        if retrieval_method == "bm25":
            st.markdown("### BM25 Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                bm25_k1 = st.slider(
                    "k1 Parameter",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Controls term frequency saturation"
                )
                st.session_state.config["bm25_k1"] = bm25_k1
                
            with col2:
                bm25_b = st.slider(
                    "b Parameter",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    help="Controls document length normalization"
                )
                st.session_state.config["bm25_b"] = bm25_b
            
            st.info("Higher k1 values increase the impact of term frequency. Higher b values increase normalization for document length.")
            
        st.markdown("### Maximal Marginal Relevance (MMR)")
        # Only relevant for methods using the vector store
        if retrieval_method in ["vector", "hybrid"]:
            use_mmr = st.checkbox(
                "Use MMR for Vector Store Retrieval",
                value=st.session_state.config.get("use_mmr", False), # Default False
                key="config_use_mmr",
                help="Re-ranks vector search results to maximize relevance and diversity."
            )
            st.session_state.config["use_mmr"] = use_mmr

            if use_mmr:
                mmr_col1, mmr_col2 = st.columns(2)
                with mmr_col1:
                    # Ensure fetch_k is at least top_k, ideally more
                    min_fetch_k = max(top_k, 5) # Ensure fetch_k >= top_k
                    default_fetch_k = max(st.session_state.config.get("mmr_fetch_k", 20), min_fetch_k) # Default 20, but ensure >= min
                    mmr_fetch_k = st.slider(
                        "MMR Fetch K", min_value=min_fetch_k, max_value=50, # Example max
                        value=default_fetch_k,
                        step=1, key="config_mmr_fetch_k",
                        help=f"Fetch N docs initially before MMR (must be >= Top-K={top_k})."
                    )
                    st.session_state.config["mmr_fetch_k"] = mmr_fetch_k
                with mmr_col2:
                    # Renamed key from prompt to match code example `mmr_lambda_mult`
                    mmr_lambda = st.slider(
                        "MMR Lambda", 0.0, 1.0, # Renamed slider label
                        value=st.session_state.config.get("mmr_lambda", 0.5), # Use 'mmr_lambda' key
                        step=0.1, key="config_mmr_lambda", # Use 'mmr_lambda' key
                        help="Balance diversity (0.0) vs relevance (1.0)."
                    )
                    st.session_state.config["mmr_lambda"] = mmr_lambda # Use 'mmr_lambda' key
        else:
            st.info("MMR option is only applicable for 'vector' or 'hybrid' retrieval.")
            # Ensure MMR config is off if not applicable
            if st.session_state.config.get("use_mmr"):
                 st.session_state.config["use_mmr"] = False
        # Compare retrieval methods
        with st.expander("Retrieval Method Comparison"):
            st.markdown("""
            | Method | Strengths | Weaknesses |
            | ------ | --------- | ---------- |
            | **Vector** | Semantic understanding, handles concepts | Misses exact matches, needs good embeddings |
            | **BM25** | Fast, exact match capability, no training | Limited semantic understanding |
            | **Hybrid** | Balances semantic and keyword matching | Requires tuning alpha parameter |
            """)
    
    # Tab: Query Processing
    with tabs[4]:
        st.markdown('<p class="sub-header">Query Processing Configuration</p>', unsafe_allow_html=True)
        
        # Query expansion
        query_expansion = st.checkbox(
            "Enable Query Expansion",
            value=st.session_state.config["query_expansion"],
            help="Generate variations of the query to improve recall"
        )
        st.session_state.config["query_expansion"] = query_expansion
        
        # Expansion method (only if query expansion is enabled)
        if query_expansion:
            expansion_method = st.selectbox(
                "Expansion Method",
                ["simple", "llm", "hybrid"],
                index=["simple", "llm", "hybrid"].index(st.session_state.config["expansion_method"]) 
                      if st.session_state.config["expansion_method"] in ["simple", "llm", "hybrid"] else 0,
                help="Method used for query expansion"
            )
            st.session_state.config["expansion_method"] = expansion_method
            
            # Expansion visualization
            st.markdown("### Example Query Expansion")
            example_query = st.text_input("Try an example query", value="How does chunking work in RAG?")
            
            if example_query:
                st.markdown("**Original Query:** " + example_query)
                st.markdown("**Expanded Queries:**")
                
                if expansion_method == "simple":
                    expansions = [
                        example_query,
                        example_query.replace("chunking", "segmentation"),
                        example_query.replace("work", "function")
                    ]
                elif expansion_method == "llm":
                    expansions = [
                        example_query,
                        "What are the mechanisms of document chunking in RAG?",
                        "Explain document segmentation techniques for retrieval augmented generation"
                    ]
                else:  # hybrid
                    expansions = [
                        example_query,
                        example_query.replace("chunking", "segmentation"),
                        "Explain document segmentation techniques for retrieval augmented generation"
                    ]
                
                for i, exp in enumerate(expansions):
                    st.markdown(f"{i+1}. *{exp}*")
            
        # HyDE options
        st.markdown("### Hypothetical Document Embeddings (HyDE)")
        
        use_hyde = st.checkbox(
            "Enable HyDE",
            value=st.session_state.config.get("use_hyde", False),
            help="Use HyDE to improve semantic search"
        )
        st.session_state.config["use_hyde"] = use_hyde
        
        if use_hyde:
            hyde_method = st.selectbox(
                "HyDE Method",
                ["basic", "enhanced", "multi_document"],
                index=0,
                help="Method for generating hypothetical documents"
            )
            st.session_state.config["hyde_method"] = hyde_method
            
            if hyde_method == "multi_document":
                hyde_num_docs = st.slider(
                    "Number of Hypothetical Documents",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1
                )
                st.session_state.config["hyde_num_docs"] = hyde_num_docs
                
            # HyDE explanation
            with st.expander("How HyDE Works"):
                st.markdown("""
                **Hypothetical Document Embeddings (HyDE)** works by:
                
                1. Generating a hypothetical document that might answer the query
                2. Embedding this document instead of the query itself
                3. Using the document embedding for retrieval
                
                This improves semantic search by bridging the gap between questions and answers.
                """)
    
    # Tab: Reranking
    with tabs[5]:
        st.markdown('<p class="sub-header">Reranking Configuration</p>', unsafe_allow_html=True)
        
        # Use reranking
        use_reranking = st.checkbox(
            "Enable Reranking",
            value=st.session_state.config["use_reranking"],
            help="Rerank initially retrieved documents for better precision"
        )
        st.session_state.config["use_reranking"] = use_reranking
        
        # Reranking method (only if reranking is enabled)
        if use_reranking:
            reranking_method = st.selectbox(
                "Reranking Method",
                ["cross_encoder", "contextual", "diversity", "multi_stage"],
                index=["cross_encoder", "contextual", "diversity", "multi_stage"].index(
                    st.session_state.config["reranking_method"]
                ) if st.session_state.config["reranking_method"] in ["cross_encoder", "contextual", "diversity", "multi_stage"] else 0,
                help="Method used for reranking"
            )
            st.session_state.config["reranking_method"] = reranking_method
            
            # Cross-encoder model (only for cross_encoder method)
            if reranking_method == "cross_encoder":
                reranking_model = st.selectbox(
                    "Cross-Encoder Model",
                    [
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        "cross-encoder/ms-marco-TinyBERT-L-2",
                        "cross-encoder/nli-deberta-v3-base"
                    ],
                    index=0,
                    help="Model used for cross-encoder reranking"
                )
                st.session_state.config["reranking_model"] = reranking_model
                
            # Multi-stage reranking
            if reranking_method == "multi_stage":
                st.markdown("### Multi-Stage Reranking Pipeline")
                
                stage_options = ["semantic", "cross_encoder", "keyword", "diversity"]
                default_stages = ["semantic", "cross_encoder", "diversity"]
                
                # Get current stages or use default
                current_stages = st.session_state.config.get("reranking_stages", default_stages)
                
                # Convert to multiselect
                selected_stages = st.multiselect(
                    "Reranking Stages",
                    options=stage_options,
                    default=current_stages,
                    help="Stages to include in the reranking pipeline"
                )
                
                # Ensure at least one stage is selected
                if not selected_stages:
                    st.warning("Please select at least one reranking stage.")
                    selected_stages = default_stages
                    
                st.session_state.config["reranking_stages"] = selected_stages
                
                # Visualize the pipeline
                st.markdown("### Pipeline Visualization")
                pipeline_cols = st.columns(len(selected_stages))
                
                for i, stage in enumerate(selected_stages):
                    with pipeline_cols[i]:
                        st.markdown(f"**Stage {i+1}**")
                        st.markdown(f"*{stage.replace('_', ' ').title()}*")
                        
                        # Add stage description
                        if stage == "semantic":
                            st.caption("Vector similarity scoring")
                        elif stage == "cross_encoder":
                            st.caption("Cross-encoder relevance scoring")
                        elif stage == "keyword":
                            st.caption("Keyword matching")
                        elif stage == "diversity":
                            st.caption("Maximize result diversity")
                
            # Diversity reranking options
            if reranking_method == "diversity" or "diversity" in st.session_state.config.get("reranking_stages", []):
                st.markdown("### Diversity Parameters")
                
                diversity_alpha = st.slider(
                    "Diversity-Relevance Balance (Alpha)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.config.get("diversity_alpha", 0.7),
                    step=0.1,
                    help="Higher values prioritize relevance, lower values prioritize diversity"
                )
                st.session_state.config["diversity_alpha"] = diversity_alpha
                
                # Visualization of the trade-off
                fig, ax = plt.subplots(figsize=(6, 1))
                ax.barh(["Balance"], [diversity_alpha], color='blue', alpha=0.6, label='Relevance')
                ax.barh(["Balance"], [1-diversity_alpha], left=[diversity_alpha], color='green', alpha=0.6, label='Diversity')
                
                # Add labels
                ax.text(diversity_alpha/2, 0, f"Relevance: {diversity_alpha:.1f}", 
                        ha='center', va='center', color='white' if diversity_alpha > 0.3 else 'black')
                ax.text(diversity_alpha + (1-diversity_alpha)/2, 0, f"Diversity: {1-diversity_alpha:.1f}", 
                        ha='center', va='center', color='white' if (1-diversity_alpha) > 0.3 else 'black')
                
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                
        # Explanation of reranking methods
        with st.expander("About Reranking Methods"):
            st.markdown("""
            **Reranking** improves retrieval precision by reordering initially retrieved documents. Methods include:
            
            - **Cross-Encoder**: Uses a cross-encoder model to score query-document pairs (more accurate but slower)
            - **Contextual**: Considers conversation history when reranking
            - **Diversity**: Balances relevance with diversity to avoid redundant information
            - **Multi-Stage**: Combines multiple reranking approaches in a pipeline
            """)
    
    # Tab: Generation
    with tabs[6]:
        st.markdown('<p class="sub-header">Generation Configuration</p>', unsafe_allow_html=True)

        # --- Response style selection ---
        st.markdown("### Response Style")

        response_style = st.selectbox(
            "Response Style",
            ["Concise", "Balanced", "Detailed", "Custom"],
            index=1,  # Default to balanced
            help="Select a preset style or define a custom prompt template."
        )
        # Store the selected style itself if needed elsewhere
        # st.session_state.config["response_style"] = response_style # Optional: Store the style name

        # Preset prompt templates based on style
        prompt_templates = {
            "Concise": "Answer the question briefly based on this context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
            "Balanced": "Answer the question based ONLY on the following context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
            "Detailed": "Provide a detailed answer to the question using only the information from the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer with a comprehensive explanation:"
        }

        # --- Prompt Template Handling ---
        st.markdown("### Prompt Template")
        # Set prompt template based on style or allow custom input
        if response_style != "Custom":
            # Use preset template
            prompt_template = prompt_templates[response_style]
            st.session_state.config["prompt_template"] = prompt_template # Update config

            # Show the selected preset template (read-only)
            st.markdown("**Selected Preset Template:**")
            st.code(prompt_template, language='text') # Use st.code for better display
        else:
            # Custom prompt template input
            st.markdown("**Define Custom Template:**")
            custom_prompt_template = st.text_area(
                "Custom Prompt Template",
                value=st.session_state.config.get("prompt_template", prompt_templates["Balanced"]), # Default to balanced if no custom exists
                height=150,
                key="custom_prompt_input", # Add key for stability
                help="Template for generation prompt. Use {query} and {context} placeholders."
            )
            st.session_state.config["prompt_template"] = custom_prompt_template # Update config

        # --- Model parameters ---
        st.markdown("### Language Model Parameters")

        # Temperature control
        # Ensure default value is retrieved safely
        default_temp = st.session_state.config.get("temperature", 0.5) # Use config value or default
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5, # Allow slightly higher temp if desired
            value=float(default_temp), # Ensure it's a float
            step=0.05,
            key="temperature_slider_config", # Add key
            help="Controls randomness (0=deterministic, >1=more creative). Affects LLM-Enhanced mode."
        )
        # Update session state AND config simultaneously
        st.session_state.temperature = temperature # For immediate use in Chat sidebar potentially
        st.session_state.config["temperature"] = temperature # For saving the config

        # --- Temperature visualization ---
        st.markdown("###### Temperature Scale:")
        temp_cols = st.columns([1, 3, 1]) # Use columns to center the small plot
        with temp_cols[1]:
            try:
                # Create a temperature visualization
                fig, ax = plt.subplots(figsize=(6, 1)) # Make it shorter

                # Create a gradient bar
                x = np.linspace(0, 1.5, 150) # Range for the visual bar (0 to 1.5)

                # --- CORRECTED Colormap Definition ---
                # Define transition points normalized to the [0, 1] range
                normalized_gradient_points = [
                    (0.0, 'blue'),          # Start point (0.0 maps to 0.0)
                    (0.7 / 1.5, 'purple'), # Intermediate point (0.7 maps proportionally)
                    (1.0, 'red')           # End point (1.5 maps to 1.0)
                ]
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    'temperature', normalized_gradient_points
                )
                # --- End Correction ---

                # Plot gradient bar using the actual range (0 to 1.5)
                # and apply the colormap by normalizing the x values back to [0, 1]
                for i in range(len(x)-1):
                    # Normalize x[i] to the [0, 1] range for the colormap lookup
                    normalized_x = x[i] / 1.5
                    ax.axvspan(x[i], x[i+1], color=cmap(normalized_x), alpha=0.8)

                # Add current temperature marker (uses the actual temperature value)
                ax.axvline(x=temperature, color='black', linewidth=3, linestyle='--', label=f'Current: {temperature:.2f}')

                # Customize the plot appearance (ensure xlim matches slider range)
                ax.set_xlim(0, 1.5)
                ax.set_yticks([]) # Remove y-axis ticks
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlabel("Temperature", fontsize=9)

                plt.tight_layout(pad=0.5) # Adjust padding

                # Display the plot
                st.pyplot(fig, use_container_width=True)
                plt.close(fig) # Close the plot to free memory

            except Exception as e:
                st.error(f"Failed to create temperature plot: {e}")


    # --- Apply Button Logic (Placed AFTER all tabs) ---
    st.markdown("---") # Separator before the button

    # Initialize previous_config if it doesn't exist, before the button logic
    if "previous_config" not in st.session_state:
        st.session_state.previous_config = st.session_state.config.copy()

    # --- Define parameter lists BEFORE the button block ---
    critical_params = [
        "chunking_strategy", "chunk_size", "chunk_overlap",
        "embedding_model"
    ]
    # Define these here too for clarity, though not the cause of this specific error
    retrieval_related_params = ["retrieval_method", "retrieval_alpha"]
    reranking_related_params = ["use_reranking", "reranking_method", "reranking_model"]

    # --- Initialize flags BEFORE the button block ---
    config_changed = False
    force_rebuild_needed = False

    # Button to apply configuration changes
    if st.button("✅ Apply Configuration Changes", use_container_width=True, key="apply_config_main_btn"):

        # --- Reset flags INSIDE the button block ---
        # This ensures the button logic starts fresh each time it's clicked
        config_changed = False
        force_rebuild_needed = False
        # --- End Reset ---

        # Check critical params first
        for param in critical_params:
            if st.session_state.config.get(param) != st.session_state.previous_config.get(param):
                st.info(f"Detected change in critical parameter: {param}. Knowledge base rebuild required.")
                force_rebuild_needed = True
                config_changed = True # Mark that config changed
                break # No need to check further critical params

        # Check other params only if no critical change detected yet
        # 'force_rebuild_needed' now has a clearly defined scope from BEFORE the 'if st.button'
        if not force_rebuild_needed:
            # Check if retrieval/reranking changes require rebuild (customize this check)
            if st.session_state.config.get("retrieval_method") != st.session_state.previous_config.get("retrieval_method"):
                # Add more specific logic here if only certain transitions require rebuild
                st.info(f"Detected change in retrieval method. Rebuild might be needed (depending on implementation). Forcing rebuild.")
                # Assignment within nested block is now safe
                force_rebuild_needed = True

        # Always mark config_changed if ANY parameter differs (even non-critical like top_k or prompt)
        if not config_changed:
            if st.session_state.config != st.session_state.previous_config:
                config_changed = True
                st.info("Detected changes in non-critical parameters (e.g., top_k, temperature, prompt). Applying without rebuild.")

        # --- Apply changes logic (uses the flags) ---
        if config_changed:
            # Store current config as the new 'previous' config for the *next* comparison
            st.session_state.previous_config = st.session_state.config.copy()

            # Clear any existing RAG app instance to force re-initialization with new config
            st.session_state.rag_app = None

            # Use a spinner during re-initialization and potential rebuild
            spinner_message = "Applying configuration..."
            if force_rebuild_needed and st.session_state.corpus_uploaded:
                spinner_message = "Applying configuration and rebuilding knowledge base..."

            with st.spinner(spinner_message):
                try:
                    # Initialize new RAG app - it will pick up the latest config from session_state
                    initialize_rag_app() # Ensure this function uses st.session_state.config

                    # Prepare the knowledge base only if corpus is present
                    if st.session_state.corpus_uploaded:
                        # Call prepare_knowledge_base, forcing rebuild only if needed
                        st.session_state.rag_app.prepare_knowledge_base(force_rebuild=force_rebuild_needed)

                        if force_rebuild_needed:
                            st.success("Configuration applied and knowledge base rebuilt successfully!")
                        else:
                            st.success("Configuration applied successfully!")
                    else:
                        st.warning("Configuration saved. Load a knowledge base on the Chat page to build the index.")

                    try:
                        config_save_path = "user_config.json" # Or path from config
                        # Define config_to_save: exclude temporary/sensitive keys if needed
                        config_to_save = {k: v for k, v in st.session_state.config.items() if k != "openai_api_key"} # Example

                        with open(config_save_path, 'w') as f:
                            # Need to import json at the top of streamlit_app.py
                            import json
                            json.dump(config_to_save, f, indent=4)
                        print(f"Configuration saved to {config_save_path}")
                        st.caption("Configuration saved.") # User feedback in UI
                    except Exception as e:
                        print(f"Warning: Failed to save configuration: {e}")
                        st.caption("Failed to save configuration.") # User feedback in UI

                    # Short delay before potential rerun can make success message more visible
                    time.sleep(1.5)
                    # Consider if a rerun is necessary - often it is to reflect changes elsewhere
                    # st.rerun()

                except Exception as e:
                    st.error(f"Failed to apply configuration: {e}")
                    traceback.print_exc() # Show full error for debugging
        else:
            # This else block now correctly corresponds to the outer 'if config_changed:'
            st.info("No changes detected in the configuration.")



# Page: Metrics
def metrics_page():
    """Metrics and analytics page"""
    st.markdown('<p class="main-header">📊 RAG System Metrics & Analytics</p>', unsafe_allow_html=True)
    
    # Check if there's conversation history
    if not st.session_state.conversation_history:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("No conversation data available. Use the Chat interface to generate metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # --- Filter history to include only assistant messages (which have processing time) ---
    assistant_exchanges = [
        exchange for exchange in st.session_state.conversation_history
        if exchange.get("role") == "assistant" and "time" in exchange
    ]

    # Check if there are any assistant responses to calculate metrics from
    if not assistant_exchanges:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("No assistant responses recorded yet to calculate metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
        # Optionally display total queries even if no assistant responses
        st.metric("Total Conversation Turns (User+Assistant)", len(st.session_state.conversation_history))
        return
    # --- End Filtering and Check ---
    # Performance metrics
    st.markdown('<p class="sub-header">Performance Metrics</p>', unsafe_allow_html=True)
    
    # Create metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        # --- Calculate avg_time using the FILTERED list ---
        avg_time = sum(exchange["time"] for exchange in assistant_exchanges) / len(assistant_exchanges)
        st.metric("Average Assistant Response Time", f"{avg_time:.2f} sec")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        total_queries = len(st.session_state.conversation_history) # Can show total turns
        user_queries = sum(1 for ex in st.session_state.conversation_history if ex.get("role") == "user")
        st.metric("User Queries", user_queries)
        # Optionally show total turns: st.metric("Total Turns", total_queries)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        # --- Calculate avg_context_len using the FILTERED list ---
        # Check contexts exist before calculating average
        contexts_present = [len(exchange.get("contexts", [])) for exchange in assistant_exchanges if exchange.get("contexts") is not None]
        avg_context_len = sum(contexts_present) / len(contexts_present) if contexts_present else 0
        st.metric("Avg. Retrieved Documents (per Assistant Response)", f"{avg_context_len:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Query history visualization (Adjust data preparation) ---
    st.markdown("### Query History & Performance")

    # Create DataFrame from conversation history - focusing on assistant responses for time/docs
    history_data = []
    query_map = {} # Store user query text mapped to assistant response index

    # First pass: get user queries
    for i, exchange in enumerate(st.session_state.conversation_history):
        if exchange.get("role") == "user":
            # Map the index of the *next* assistant message to this query text
            query_map[i + 1] = exchange.get("query", "N/A")

    # Second pass: get assistant data and match with query
    for i, exchange in enumerate(st.session_state.conversation_history):
        if exchange.get("role") == "assistant" and "time" in exchange:
            query_text = query_map.get(i, "N/A (Query missing?)") # Get query text using index
            history_data.append({
                # "Turn ID": i + 1, # Or use a different ID system
                "User Query": query_text,
                "Response Time (sec)": exchange["time"],
                "Retrieved Documents": len(exchange.get("contexts", []))
            })

    history_df = pd.DataFrame(history_data)
    history_df.index += 1 # Start index from 1 for display
    history_df.index.name = "Assistant Response #"

    # Display as a table
    st.dataframe(history_df, use_container_width=True)

    # --- Visualizations (Plot using the prepared history_df) ---
    st.markdown("### Response Time Analysis")
    if not history_df.empty:
        try:
            fig_time, ax_time = plt.subplots(figsize=(10, 4))
            # Use the index for the x-axis if it represents sequence
            sns.lineplot(x=history_df.index, y="Response Time (sec)", data=history_df, marker='o', ax=ax_time)
            ax_time.set_title("Response Time per Assistant Response")
            ax_time.set_xlabel("Assistant Response Sequence")
            ax_time.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig_time, use_container_width=True)
            plt.close(fig_time) # Close plot
        except Exception as e:
            st.error(f"Failed to plot response time: {e}")
    else:
        st.info("No data available for response time plot.")


    st.markdown("### Document Retrieval Analysis")
    if not history_df.empty and "Retrieved Documents" in history_df.columns:
        try:
            fig_docs, ax_docs = plt.subplots(figsize=(10, 4))
            sns.barplot(x=history_df.index, y="Retrieved Documents", data=history_df, ax=ax_docs, palette="viridis")
            ax_docs.set_title("Number of Retrieved Documents per Assistant Response")
            ax_docs.set_xlabel("Assistant Response Sequence")
            ax_docs.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_docs, use_container_width=True)
            plt.close(fig_docs) # Close plot
        except Exception as e:
            st.error(f"Failed to plot retrieved documents: {e}")
    else:
        st.info("No data available for retrieved documents plot.")


    # --- Export metrics (Uses the prepared history_df) ---
    st.markdown("### Export Metrics")
    if not history_df.empty:
        if st.button("Export Metrics to CSV", key="export_metrics_btn"): # Add key
            try:
                csv = history_df.to_csv(index_label="Assistant Response #").encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="rag_metrics.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to generate CSV for download: {e}")

# Page: Experiment Lab
def experiment_lab_page():
    """Experiment laboratory page"""
    st.markdown('<p class="main-header">🧪 RAG Experiment Laboratory</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    Compare different RAG configurations to find the optimal setup for your specific use case.
    Run experiments across different chunking strategies, retrieval methods, and more.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if corpus is loaded
    if not st.session_state.corpus_uploaded:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ No knowledge base loaded. Please upload documents or load the example dataset from the Chat page.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Initialize RAG app if needed
    if st.session_state.rag_app is None:
        initialize_rag_app()
    
    # Create experiment setup tabs
    tabs = st.tabs([
        "🧩 Chunking Experiment", 
        "🔍 Retrieval Experiment", 
        "🔄 Combined Experiment", 
        "📊 Results"
    ])
    
    # Tab: Chunking Experiment
    with tabs[0]:
        st.markdown('<p class="sub-header">Chunking Strategy Experiment</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        Compare different document chunking strategies to find which works best for your knowledge base.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Test query
        test_query = st.text_input(
            "Test Query",
            value="What is vector search in RAG?",
            help="Query to use for testing chunking strategies",
            key="chunking_test_query"
        )
        
        # Chunking strategies to test
        st.markdown("### Select Chunking Strategies to Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_fixed = st.checkbox("Fixed Size Chunking", value=True)
            test_paragraph = st.checkbox("Paragraph Chunking", value=True)
            test_semantic = st.checkbox("Semantic Chunking", value=True)
            
        with col2:
            # Fixed size options (only if fixed is selected)
            if test_fixed:
                fixed_size = st.slider(
                    "Fixed Chunk Size",
                    min_value=64,
                    max_value=512,
                    value=128,
                    step=64,
                    help="Size of fixed chunks in tokens"
                )
                
                fixed_overlap = st.slider(
                    "Fixed Chunk Overlap",
                    min_value=0,
                    max_value=fixed_size // 2,
                    value=32,
                    step=16,
                    help="Overlap between fixed chunks in tokens"
                )
                
        # Run experiment button
        if st.button("Run Chunking Experiment"):
            if not test_query:
                st.error("Please enter a test query.")
                return
                
            if not (test_fixed or test_paragraph or test_semantic):
                st.error("Please select at least one chunking strategy to test.")
                return
                
            # Run the experiment
            with st.spinner("Running chunking experiment..."):
                # List to store experiment results
                chunking_results = []
                
                # Test fixed strategy
                if test_fixed:
                    # Create a temporary copy of the configuration
                    temp_config = st.session_state.config.copy()
                    temp_config["chunking_strategy"] = "fixed"
                    temp_config["chunk_size"] = fixed_size
                    temp_config["chunk_overlap"] = fixed_overlap
                    
                    # Create a RAG app with this configuration
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.load_corpus(st.session_state.config["corpus_path"])
                    
                    # Process the query and time it
                    start_time = time.time()
                    temp_app.prepare_knowledge_base(force_rebuild=True)
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    chunking_results.append({
                        "Strategy": "Fixed",
                        "Parameters": f"Size: {fixed_size}, Overlap: {fixed_overlap}",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Test paragraph strategy
                if test_paragraph:
                    # Create a temporary copy of the configuration
                    temp_config = st.session_state.config.copy()
                    temp_config["chunking_strategy"] = "paragraph"
                    
                    # Create a RAG app with this configuration
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.load_corpus(st.session_state.config["corpus_path"])
                    
                    # Process the query and time it
                    start_time = time.time()
                    temp_app.prepare_knowledge_base(force_rebuild=True)
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    chunking_results.append({
                        "Strategy": "Paragraph",
                        "Parameters": "N/A",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Test semantic strategy
                if test_semantic:
                    # Create a temporary copy of the configuration
                    temp_config = st.session_state.config.copy()
                    temp_config["chunking_strategy"] = "semantic"
                    
                    # Create a RAG app with this configuration
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.load_corpus(st.session_state.config["corpus_path"])
                    
                    # Process the query and time it
                    start_time = time.time()
                    temp_app.prepare_knowledge_base(force_rebuild=True)
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    chunking_results.append({
                        "Strategy": "Semantic",
                        "Parameters": "N/A",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Store results in session state
                st.session_state.chunking_results = chunking_results
                
                # Show success message
                st.success("Chunking experiment completed!")
                
            # Display results table
            if hasattr(st.session_state, "chunking_results"):
                st.markdown("### Chunking Experiment Results")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(st.session_state.chunking_results)
                
                # Display table
                st.dataframe(results_df[["Strategy", "Parameters", "Docs Retrieved", "Processing Time"]])
                
                # Display answers
                st.markdown("### Generated Answers")
                
                for result in st.session_state.chunking_results:
                    st.markdown('<div class="content-box">', unsafe_allow_html=True)
                    st.markdown(f"**{result['Strategy']} Strategy:**")
                    st.markdown(result["Answer"])
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab: Retrieval Experiment
    with tabs[1]:
        st.markdown('<p class="sub-header">Retrieval Method Experiment</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        Compare different retrieval methods to find which works best for your knowledge base and queries.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Test query
        test_query = st.text_input(
            "Test Query",
            value="What is vector search in RAG?",
            help="Query to use for testing retrieval methods",
            key="retrieval_test_query"
        )
        
        # Retrieval methods to test
        st.markdown("### Select Retrieval Methods to Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_vector = st.checkbox("Vector Search", value=True)
            test_bm25 = st.checkbox("BM25 Search", value=True)
            test_hybrid = st.checkbox("Hybrid Search", value=True)
            
        with col2:
            # Hybrid alpha (only if hybrid is selected)
            if test_hybrid:
                hybrid_alpha = st.slider(
                    "Hybrid Alpha (Vector Weight)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Weight of vector search in hybrid retrieval"
                )
                
            # Top-k for all methods
            top_k = st.slider(
                "Number of Documents to Retrieve (Top-K)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Number of documents to retrieve for each method"
            )
                
        # Run experiment button
        if st.button("Run Retrieval Experiment"):
            if not test_query:
                st.error("Please enter a test query.")
                return
                
            if not (test_vector or test_bm25 or test_hybrid):
                st.error("Please select at least one retrieval method to test.")
                return
                
            # Run the experiment
            with st.spinner("Running retrieval experiment..."):
                # Use the current chunking strategy
                current_chunking = st.session_state.config["chunking_strategy"]
                current_size = st.session_state.config["chunk_size"]
                current_overlap = st.session_state.config["chunk_overlap"]
                
                # List to store experiment results
                retrieval_results = []
                
                # Create a base RAG app for chunking
                base_app = RAGApplication()
                base_app.config = st.session_state.config.copy()
                base_app.load_corpus(st.session_state.config["corpus_path"])
                base_app.prepare_knowledge_base(force_rebuild=True)
                
                # Get embeddings once
                # query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                #     [test_query], model_name=base_app.config["embedding_model"]
                # )[0]
                
                # Test vector search
                if test_vector:
                    # Create a temporary app config
                    temp_config = st.session_state.config.copy()
                    temp_config["retrieval_method"] = "vector"
                    temp_config["top_k"] = top_k
                    
                    # Update app config
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.corpus = base_app.corpus
                    # temp_app.chunked_docs = base_app.chunked_docs
                    # temp_app.doc_embeddings = base_app.doc_embeddings
                    # temp_app.doc_ids = base_app.doc_ids
                    temp_app.vector_store = base_app.vector_store
                    temp_app._create_retriever()
                    
                    # Process the query and time it
                    start_time = time.time()
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    retrieval_results.append({
                        "Method": "Vector",
                        "Parameters": f"Top-K: {top_k}",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Test BM25 search
                if test_bm25:
                    # Create a temporary app config
                    temp_config = st.session_state.config.copy()
                    temp_config["retrieval_method"] = "bm25"
                    temp_config["top_k"] = top_k
                    
                    # Update app config
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.corpus = base_app.corpus
                    # temp_app.chunked_docs = base_app.chunked_docs
                    # temp_app.doc_embeddings = base_app.doc_embeddings
                    # temp_app.doc_ids = base_app.doc_ids
                    temp_app.vector_store = base_app.vector_store
                    temp_app._create_retriever()
                    
                    # Process the query and time it
                    start_time = time.time()
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    retrieval_results.append({
                        "Method": "BM25",
                        "Parameters": f"Top-K: {top_k}",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Test hybrid search
                if test_hybrid:
                    # Create a temporary app config
                    temp_config = st.session_state.config.copy()
                    temp_config["retrieval_method"] = "hybrid"
                    temp_config["retrieval_alpha"] = hybrid_alpha
                    temp_config["top_k"] = top_k
                    
                    # Update app config
                    temp_app = RAGApplication()
                    temp_app.config = temp_config
                    temp_app.corpus = base_app.corpus
                    # temp_app.chunked_docs = base_app.chunked_docs
                    # temp_app.doc_embeddings = base_app.doc_embeddings
                    # temp_app.doc_ids = base_app.doc_ids
                    temp_app.vector_store = base_app.vector_store
                    temp_app._create_retriever()
                    
                    # Process the query and time it
                    start_time = time.time()
                    answer, contexts = temp_app.process_query(test_query)
                    process_time = time.time() - start_time
                    
                    # Add to results
                    retrieval_results.append({
                        "Method": "Hybrid",
                        "Parameters": f"Alpha: {hybrid_alpha}, Top-K: {top_k}",
                        "Docs Retrieved": len(contexts),
                        "Processing Time": process_time,
                        "Answer": answer
                    })
                    
                # Store results in session state
                st.session_state.retrieval_results = retrieval_results
                
                # Show success message
                st.success("Retrieval experiment completed!")
                
            # Display results table
            if hasattr(st.session_state, "retrieval_results"):
                st.markdown("### Retrieval Experiment Results")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(st.session_state.retrieval_results)
                
                # Display table
                st.dataframe(results_df[["Method", "Parameters", "Docs Retrieved", "Processing Time"]])
                
                # Display answers
                st.markdown("### Generated Answers")
                
                for result in st.session_state.retrieval_results:
                    st.markdown('<div class="content-box">', unsafe_allow_html=True)
                    st.markdown(f"**{result['Method']} Method:**")
                    st.markdown(result["Answer"])
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab: Combined Experiment
    with tabs[2]:
        st.markdown('<p class="sub-header">Combined Experiment</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ This experiment may take several minutes to complete.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Test queries
        st.markdown("### Test Queries")
        st.markdown("Enter multiple test queries separated by new lines:")
        
        test_queries = st.text_area(
            "Test Queries",
            value="What is vector search in RAG?\nHow does document chunking work?\nWhat are embedding models?",
            height=100,
            help="Queries to use for testing configurations"
        ).strip().split("\n")
        
        # Configurations to test
        st.markdown("### Select Configurations to Test")
        
        # Chunking strategies
        st.markdown("**Chunking Strategies:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_fixed = st.checkbox("Fixed Size", value=True, key="combined_fixed")
        with col2:
            test_paragraph = st.checkbox("Paragraph", value=True, key="combined_paragraph")
        with col3:
            test_semantic = st.checkbox("Semantic", value=False, key="combined_semantic")
            
        # Retrieval methods
        st.markdown("**Retrieval Methods:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_vector = st.checkbox("Vector", value=True, key="combined_vector")
        with col2:
            test_bm25 = st.checkbox("BM25", value=False, key="combined_bm25")
        with col3:
            test_hybrid = st.checkbox("Hybrid", value=True, key="combined_hybrid")
            
        # Reranking options
        st.markdown("**Reranking Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            test_no_reranking = st.checkbox("No Reranking", value=True, key="combined_no_reranking")
        with col2:
            test_cross_encoder = st.checkbox("Cross-Encoder", value=True, key="combined_cross_encoder")
            
        # Run button
        if st.button("Run Combined Experiment"):
            if not test_queries or not test_queries[0]:
                st.error("Please enter at least one test query.")
                return
                
            if not (test_fixed or test_paragraph or test_semantic):
                st.error("Please select at least one chunking strategy.")
                return
                
            if not (test_vector or test_bm25 or test_hybrid):
                st.error("Please select at least one retrieval method.")
                return
                
            if not (test_no_reranking or test_cross_encoder):
                st.error("Please select at least one reranking option.")
                return
                
            # Run the experiment
            with st.spinner("Running comprehensive experiment... This may take a while."):
                progress_bar = st.progress(0)
                progress_status = st.empty()
                
                # List to store all configurations to test
                configs_to_test = []
                
                # Generate all combinations to test
                for chunking in ["fixed", "paragraph", "semantic"]:
                    if not ((chunking == "fixed" and test_fixed) or
                            (chunking == "paragraph" and test_paragraph) or
                            (chunking == "semantic" and test_semantic)):
                        continue
                        
                    for retrieval in ["vector", "bm25", "hybrid"]:
                        if not ((retrieval == "vector" and test_vector) or
                                (retrieval == "bm25" and test_bm25) or
                                (retrieval == "hybrid" and test_hybrid)):
                            continue
                            
                        for reranking in [False, True]:
                            if not ((not reranking and test_no_reranking) or
                                    (reranking and test_cross_encoder)):
                                continue
                                
                            # Create configuration
                            config = st.session_state.config.copy()
                            config["chunking_strategy"] = chunking
                            config["retrieval_method"] = retrieval
                            config["use_reranking"] = reranking
                            
                            if reranking:
                                config["reranking_method"] = "cross_encoder"
                                
                            # Add to list
                            configs_to_test.append(config)
                
                # List to store experiment results
                combined_results = []
                
                # Calculate total iterations for progress bar
                total_iterations = len(configs_to_test) * len(test_queries)
                current_iteration = 0
                
                # Test each configuration with each query
                for config in configs_to_test:
                    for query in test_queries:
                        # Update progress
                        current_iteration += 1
                        progress = current_iteration / total_iterations
                        progress_bar.progress(progress)
                        progress_status.text(f"Processing configuration {current_iteration} of {total_iterations}...")
                        
                        # Create a RAG app with this configuration
                        app = RAGApplication()
                        app.config = config
                        app.load_corpus(st.session_state.config["corpus_path"])
                        
                        # Process the query and time it
                        start_time = time.time()
                        app.prepare_knowledge_base(force_rebuild=False)
                        answer, contexts = app.process_query(query)
                        process_time = time.time() - start_time
                        
                        # Add to results
                        combined_results.append({
                            "Query": query,
                            "Chunking": config["chunking_strategy"],
                            "Retrieval": config["retrieval_method"],
                            "Reranking": "Yes" if config["use_reranking"] else "No",
                            "Docs Retrieved": len(contexts),
                            "Time (sec)": process_time,
                            "Answer": answer
                        })
                        
                # Store results in session state
                st.session_state.combined_results = combined_results
                
                # Clean up progress indicators
                progress_bar.empty()
                progress_status.empty()
                
                # Show success message
                st.success(f"Combined experiment completed with {len(combined_results)} configurations!")
                
            # Display results table
            if hasattr(st.session_state, "combined_results"):
                st.markdown("### Combined Experiment Results")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(st.session_state.combined_results)
                
                # Display table
                st.dataframe(results_df[["Query", "Chunking", "Retrieval", "Reranking", "Docs Retrieved", "Time (sec)"]])
                
                # Create a pivot table for analysis
                st.markdown("### Performance Analysis")
                
                pivot = pd.pivot_table(
                    results_df,
                    values="Time (sec)",
                    index=["Chunking", "Retrieval"],
                    columns=["Reranking"],
                    aggfunc="mean"
                )
                
                st.dataframe(pivot)
                
                # Create a bar chart of average times
                st.markdown("### Average Processing Time by Configuration")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                avg_times = results_df.groupby(["Chunking", "Retrieval", "Reranking"])["Time (sec)"].mean().reset_index()
                avg_times["Configuration"] = avg_times.apply(
                    lambda x: f"{x['Chunking']}/{x['Retrieval']}/{x['Reranking']}", axis=1
                )
                
                sns.barplot(x="Configuration", y="Time (sec)", data=avg_times, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_title("Average Processing Time by Configuration")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                # Convert plot to image for Streamlit
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.image(buf)
    
    # Tab: Results
    with tabs[3]:
        st.markdown('<p class="sub-header">Experiment Results & Analysis</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        View and analyze results from all your experiments. Export data for further analysis or for your research paper.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Check if there are any results
        has_chunking_results = hasattr(st.session_state, "chunking_results")
        has_retrieval_results = hasattr(st.session_state, "retrieval_results")
        has_combined_results = hasattr(st.session_state, "combined_results")
        
        if not (has_chunking_results or has_retrieval_results or has_combined_results):
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("No experiment results available. Run experiments to generate results.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
            
        # Select experiment to view
        experiment_type = st.selectbox(
            "Select Experiment Results",
            ["Chunking Experiment", "Retrieval Experiment", "Combined Experiment"],
            key="results_experiment_type"
        )
        
        # Display selected experiment results
        if experiment_type == "Chunking Experiment" and has_chunking_results:
            # Convert to DataFrame
            chunking_df = pd.DataFrame(st.session_state.chunking_results)
            
            # Display table
            st.dataframe(chunking_df)
            
            # Create visualizations
            st.markdown("### Chunking Strategy Performance")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Processing time comparison
            sns.barplot(x="Strategy", y="Processing Time", data=chunking_df, ax=ax1)
            ax1.set_title("Processing Time by Chunking Strategy")
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Documents retrieved comparison
            sns.barplot(x="Strategy", y="Docs Retrieved", data=chunking_df, ax=ax2)
            ax2.set_title("Documents Retrieved by Chunking Strategy")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Convert plot to image for Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            
            # Export button
            if st.button("Export Chunking Results"):
                csv = chunking_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="chunking_experiment.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        elif experiment_type == "Retrieval Experiment" and has_retrieval_results:
            # Convert to DataFrame
            retrieval_df = pd.DataFrame(st.session_state.retrieval_results)
            
            # Display table
            st.dataframe(retrieval_df)
            
            # Create visualizations
            st.markdown("### Retrieval Method Performance")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Processing time comparison
            sns.barplot(x="Method", y="Processing Time", data=retrieval_df, ax=ax1)
            ax1.set_title("Processing Time by Retrieval Method")
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Documents retrieved comparison
            sns.barplot(x="Method", y="Docs Retrieved", data=retrieval_df, ax=ax2)
            ax2.set_title("Documents Retrieved by Retrieval Method")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Convert plot to image for Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            
            # Export button
            if st.button("Export Retrieval Results"):
                csv = retrieval_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="retrieval_experiment.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
        elif experiment_type == "Combined Experiment" and has_combined_results:
            # Convert to DataFrame
            combined_df = pd.DataFrame(st.session_state.combined_results)
            
            # Display table
            st.dataframe(combined_df)
            
            # Create visualizations
            st.markdown("### Combined Experiment Analysis")
            
            # Query filter
            if "Query" in combined_df.columns:
                selected_query = st.selectbox(
                    "Filter by Query",
                    ["All Queries"] + list(combined_df["Query"].unique())
                )
                
                if selected_query != "All Queries":
                    filtered_df = combined_df[combined_df["Query"] == selected_query]
                else:
                    filtered_df = combined_df
            else:
                filtered_df = combined_df
                
            # Heat map of configuration performance
            st.markdown("### Configuration Performance Heat Map")
            
            # Create a pivot table
            pivot_data = filtered_df.pivot_table(
                values="Time (sec)",
                index=["Chunking", "Retrieval"],
                columns="Reranking",
                aggfunc="mean"
            )
            
            # Plot heat map
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_title("Average Processing Time (sec) by Configuration")
            
            # Convert plot to image for Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            st.image(buf)
            
            # Best configuration analysis
            st.markdown("### Best Configurations")
            
            # Find best configuration by time
            best_time_idx = filtered_df["Time (sec)"].idxmin()
            best_time_config = filtered_df.loc[best_time_idx]
            
            st.markdown("**Fastest Configuration:**")
            st.markdown(f"""
            - Chunking: {best_time_config["Chunking"]}
            - Retrieval: {best_time_config["Retrieval"]}
            - Reranking: {best_time_config["Reranking"]}
            - Time: {best_time_config["Time (sec)"]:.2f} seconds
            """)
            
            # Export button
            if st.button("Export Combined Results"):
                csv = combined_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="combined_experiment.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

# Page: About
def about_page():
    """About page with information about the system"""
    st.markdown('<p class="main-header">ℹ️ About Advanced RAG Knowledge Management System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This Advanced RAG Knowledge Management System is a comprehensive platform for implementing, 
    testing, and optimizing Retrieval-Augmented Generation (RAG) systems.
    """)
    
    # System architecture
    st.markdown("## System Architecture")
    
    st.markdown("""
    The system is built with a modular architecture that separates core components for flexibility 
    and extensibility:
    
    1. **Document Processing**: Multiple chunking strategies for optimal information segmentation
    2. **Embedding & Indexing**: Various embedding models and efficient vector storage
    3. **Retrieval Engine**: Multiple retrieval methods including vector, keyword, and hybrid search
    4. **Query Processing**: Query expansion and reformulation techniques
    5. **Reranking**: Enhanced precision through cross-encoder reranking and diversity optimization
    6. **Answer Generation**: Contextual response generation with different prompt engineering strategies
    7. **Evaluation Framework**: Comprehensive metrics and evaluation tools
    """)
    
    # Features
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Knowledge Management**
        - Document ingestion and processing
        - Multiple chunking strategies
        - Efficient vector indexing
        - Customizable knowledge sources
        """)
        
    with col2:
        st.markdown("""
        **Retrieval Capabilities**
        - Semantic vector search
        - BM25 keyword search
        - Hybrid search strategies
        - Multi-stage retrieval pipeline
        """)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Advanced Techniques**
        - Query expansion
        - Context-aware reranking
        - Diversity optimization
        - Prompt engineering
        """)
        
    with col2:
        st.markdown("""
        **Experimentation**
        - Comparative analysis
        - Performance metrics
        - Configuration testing
        - Results visualization
        """)
    
    # Implementation details
    st.markdown("## Implementation Details")
    
    st.markdown("""
    The system is implemented in Python with the following key libraries:
    
    - **Sentence Transformers**: For embedding generation
    - **FAISS**: For efficient vector search
    - **HuggingFace Transformers**: For reranking and embedding models
    - **NLTK**: For text processing and chunking
    - **Streamlit**: For the web interface
    - **Pandas & Matplotlib**: For data analysis and visualization
    """)
    
    # Research applications
    st.markdown("## Research Applications")
    
    st.markdown("""
    This system is designed to support research in several key areas:
    
    1. **Chunking Strategy Comparison**: Evaluate different document segmentation approaches
    2. **Embedding Model Evaluation**: Compare embedding models for retrieval performance
    3. **Retrieval Method Analysis**: Measure effectiveness of different retrieval techniques
    4. **Reranking Impact Assessment**: Quantify the benefits of reranking stages
    5. **RAG Pipeline Optimization**: Find optimal configurations for specific use cases
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    Developed for Advanced Research in Retrieval-Augmented Generation
    
    Version 1.0
    """)

# Main function to run the app
def main():
    # Navigation sidebar
    sidebar()
    
    # Display the selected page
    if st.session_state.current_page == "Chat":
        chat_page()
    elif st.session_state.current_page == "Configuration":
        configuration_page()
    elif st.session_state.current_page == "Metrics":
        metrics_page()
    elif st.session_state.current_page == "Experiment Lab":
        experiment_lab_page()
    elif st.session_state.current_page == "About":
        about_page()

if __name__ == "__main__":
    main()