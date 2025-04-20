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
    
if "config" not in st.session_state:
    st.session_state.config = {
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
        "retrieval_alpha": 0.7,
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

if 'response_mode' not in st.session_state:
    st.session_state.response_mode = "Extractive (Basic Retrieval)" # Default value

if 'enable_comparison' not in st.session_state:
    st.session_state.enable_comparison = False # Default to comparison being off

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.3 # Default temperature value

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
    """Chat interface page"""
    st.markdown('<p class="main-header">💬 Advanced RAG Chat Interface</p>', unsafe_allow_html=True)
    
    # Check if corpus is loaded
    if not st.session_state.corpus_uploaded:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ No knowledge base loaded. Please upload documents or load the example dataset from the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    # Initialize RAG app if needed
    if st.session_state.rag_app is None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("Initializing RAG application...")
        st.markdown('</div>', unsafe_allow_html=True)
        initialize_rag_app()
        st.rerun() 
    
    # Chat input at the top for better UX
    st.markdown("### Ask a question")
    
    query = st.text_input("Enter your question:", key="query_input", placeholder="Type your question here...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send", key="send_button", use_container_width=True):
            if query and st.session_state.rag_app:
                start_time = time.time()
                with st.spinner("Processing query..."):
                    try:
                        # Use the comparison flag from session state
                        if st.session_state.enable_comparison:
                            # --- Comparison Path ---
                            # Get Extractive Response
                            st.session_state.rag_app.response_mode = "extractive"
                            extractive_answer, contexts = st.session_state.rag_app.process_query(query)

                            # Get LLM Response
                            st.session_state.rag_app.response_mode = "llm"
                            llm_answer, _ = st.session_state.rag_app.process_query(query)

                            # Store both answers in history
                            st.session_state.conversation_history.append({
                                "query": query,
                                "extractive_answer": extractive_answer,
                                "llm_answer": llm_answer,
                                "contexts": contexts,
                                "time": time.time() - start_time
                            })
                            # --- End Comparison Path ---

                        else:
                            # --- Single Response Path ---
                            # Set RAG app mode based on sidebar radio button
                            if st.session_state.response_mode == "Extractive (Basic Retrieval)":
                                st.session_state.rag_app.response_mode = "extractive"
                            else:
                                st.session_state.rag_app.response_mode = "llm"

                            # Process query once
                            answer, contexts = st.session_state.rag_app.process_query(query)

                            # Store single answer in history
                            st.session_state.conversation_history.append({
                                "query": query,
                                "answer": answer,
                                "contexts": contexts,
                                "time": time.time() - start_time
                            })
                            # --- End Single Response Path ---

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                # Rerun Streamlit to update the display with the new history entry
                st.rerun()

            elif not query:
                st.warning("Please enter a question.")
            elif not st.session_state.rag_app:
                st.error("RAG application not initialized.")
    
    with col2:
        if st.button("Clear Chat History", key="clear_button"):
            st.session_state.conversation_history = []
            st.rerun()
                
    # Chat history display
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history):
            # User query
            st.markdown('<div class="user-message">', unsafe_allow_html=True)
            st.markdown('<span class="message-header">🧑 User</span>', unsafe_allow_html=True)
            st.markdown(f"{exchange['query']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Assistant response
            if "extractive_answer" in exchange and "llm_answer" in exchange:
                # Comparison view with two columns
                st.markdown("### 🤖 Assistant")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Extractive Response")
                    st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
                    st.markdown(f"{exchange['extractive_answer']}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("#### LLM-Enhanced Response")
                    st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
                    st.markdown(f"{exchange['llm_answer']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add a metrics section below the comparison columns
                st.markdown("#### Response Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                # Calculate lengths safely
                extractive_ans_text = exchange.get('extractive_answer', '')
                llm_ans_text = exchange.get('llm_answer', '')
                extractive_len = len(extractive_ans_text.split())
                llm_len = len(llm_ans_text.split())

                with metric_col1:
                    st.metric("Extractive Word Count", extractive_len)

                with metric_col2:
                    st.metric("LLM Word Count", llm_len)

                with metric_col3:
                    # Word count difference as percentage
                    if extractive_len > 0:
                        difference = ((llm_len - extractive_len) / extractive_len) * 100
                        st.metric("Length Difference", f"{difference:.1f}%",
                                delta=f"{difference:.1f}%",
                                help="Percentage difference relative to extractive length. Positive means LLM is longer.")
                    else:
                        st.metric("Length Difference", "N/A", help="Extractive answer is empty.")
                
                # Lexical Diversity Calculation
                lexical_diversity = calculate_lexical_diversity(llm_ans_text)
                st.metric("Lexical Diversity (LLM)", f"{lexical_diversity:.3f}", 
                        help="Ratio of unique words to total words in LLM response. Higher is more varied.")
                
            elif "answer" in exchange:
                # Regular single response view
                st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
                st.markdown('<span class="message-header">🤖 Assistant</span>', unsafe_allow_html=True)
                st.markdown(f"{exchange['answer']}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Handle potential malformed history entry
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.warning("Could not display assistant response for this entry.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sources
            if exchange.get("contexts"):
                st.markdown("**Sources:**")
                for j, doc in enumerate(exchange["contexts"][:3]):
                    st.markdown('<div class="source-box">', unsafe_allow_html=True)
                    source = doc.get("title", doc.get("chunk_id", f"Source {j+1}"))
                    text_preview = doc.get('text','N/A')
                    st.markdown(f"**{source}**")
                    st.markdown(f"{text_preview[:200]}..." if len(text_preview) > 200 else text_preview)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                if "time" in exchange:
                    st.markdown(f"*Processed in {exchange['time']:.2f} seconds*")

            # Separator between conversations
            if i < len(st.session_state.conversation_history) - 1:
                st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                
    else:
        # Show welcome message when no conversation exists
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        👋 **Welcome to the RAG Chat Interface!**
        
        Ask questions about your knowledge base to see how the system retrieves and generates answers.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Configuration
def configuration_page():
    """Configuration page"""
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
    
    # Tab: Chunking
    with tabs[1]:
        st.markdown('<p class="sub-header">Chunking Configuration</p>', unsafe_allow_html=True)
        
        # Chunking strategy
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["fixed", "paragraph", "semantic"],
            index=["fixed", "paragraph", "semantic"].index(st.session_state.config["chunking_strategy"]),
            help="Method used to split documents into chunks"
        )
        st.session_state.config["chunking_strategy"] = chunking_strategy
        
        # Chunk size and overlap (only for fixed strategy)
        if chunking_strategy == "fixed":
            col1, col2 = st.columns(2)
            
            with col1:
                chunk_size = st.slider(
                    "Chunk Size",
                    min_value=32,
                    max_value=512,
                    value=st.session_state.config["chunk_size"],
                    step=32,
                    help="Number of tokens per chunk"
                )
                st.session_state.config["chunk_size"] = chunk_size
                
            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=chunk_size // 2,
                    value=min(st.session_state.config["chunk_overlap"], chunk_size // 2),
                    step=8,
                    help="Number of overlapping tokens between chunks"
                )
                st.session_state.config["chunk_overlap"] = chunk_overlap
                
        # Explanation of chunking strategies
        with st.expander("About Chunking Strategies"):
            st.markdown("""
            - **Fixed**: Splits documents into chunks of a fixed token size with optional overlap
            - **Paragraph**: Splits documents at paragraph boundaries
            - **Semantic**: Attempts to split documents while preserving semantic units and coherence
            """)
    
    # Tab: Embedding
    with tabs[2]:
        st.markdown('<p class="sub-header">Embedding Configuration</p>', unsafe_allow_html=True)
        
        # Embedding model
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "multi-qa-mpnet-base-dot-v1",
                "BAAI/bge-small-en-v1.5",
                "text-embedding-ada-002"
            ],
            index=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", 
                   "BAAI/bge-small-en-v1.5", "text-embedding-ada-002"].index(
                st.session_state.config["embedding_model"]
            ) if st.session_state.config["embedding_model"] in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", 
                                                              "multi-qa-mpnet-base-dot-v1", "BAAI/bge-small-en-v1.5", 
                                                              "text-embedding-ada-002"] else 0,
            help="Model used to generate vector embeddings"
        )
        st.session_state.config["embedding_model"] = embedding_model
        
        # Warning for OpenAI model
        if embedding_model == "text-embedding-ada-002":
            st.warning("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            
        # Explanation of embedding models
        with st.expander("About Embedding Models"):
            st.markdown("""
            - **all-MiniLM-L6-v2**: Lightweight, fast, general-purpose embedding model (384 dimensions)
            - **all-mpnet-base-v2**: Higher quality, general-purpose embedding model (768 dimensions)
            - **multi-qa-mpnet-base-dot-v1**: Optimized for question-answering tasks (768 dimensions)
            - **BAAI/bge-small-en-v1.5**: BGE small English model, good performance/speed trade-off
            - **text-embedding-ada-002**: OpenAI's embedding model, requires API key (1536 dimensions)
            """)
    
    # Tab: Retrieval
    with tabs[3]:
        st.markdown('<p class="sub-header">Retrieval Configuration</p>', unsafe_allow_html=True)
        
        # Retrieval method
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["vector", "bm25", "hybrid"],
            index=["vector", "bm25", "hybrid"].index(st.session_state.config["retrieval_method"]),
            help="Method used to retrieve documents"
        )
        st.session_state.config["retrieval_method"] = retrieval_method
        
        # Hybrid alpha (only for hybrid method)
        if retrieval_method == "hybrid":
            retrieval_alpha = st.slider(
                "Vector Search Weight (Alpha)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config["retrieval_alpha"],
                step=0.1,
                help="Weight of vector search (1-Alpha for BM25)"
            )
            st.session_state.config["retrieval_alpha"] = retrieval_alpha
            
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
        
        # Explanation of retrieval methods
        with st.expander("About Retrieval Methods"):
            st.markdown("""
            - **Vector**: Uses semantic similarity between query and document embeddings
            - **BM25**: Uses keyword-based statistics (similar to TF-IDF)
            - **Hybrid**: Combines vector and BM25 scores for balanced retrieval
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
                ["simple", "llm"],
                index=["simple", "llm"].index(st.session_state.config["expansion_method"]),
                help="Method used for query expansion"
            )
            st.session_state.config["expansion_method"] = expansion_method
            
            # Warning for LLM method
            if expansion_method == "llm":
                st.warning("LLM expansion requires an LLM API. Currently using a simple fallback.")
                
        # Explanation of query processing
        with st.expander("About Query Processing"):
            st.markdown("""
            **Query Expansion** generates variations of the original query to improve retrieval recall. Methods include:
            
            - **Simple**: Uses WordNet to find synonyms for query terms
            - **LLM**: Uses a language model to generate query variations (requires LLM API)
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
                ["cross_encoder", "contextual", "diversity"],
                index=["cross_encoder", "contextual", "diversity"].index(
                    st.session_state.config["reranking_method"]
                ) if st.session_state.config["reranking_method"] in ["cross_encoder", "contextual", "diversity"] else 0,
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
                
        # Explanation of reranking methods
        with st.expander("About Reranking Methods"):
            st.markdown("""
            **Reranking** improves retrieval precision by reordering initially retrieved documents. Methods include:
            
            - **Cross-Encoder**: Uses a cross-encoder model to score query-document pairs (more accurate but slower)
            - **Contextual**: Considers conversation history when reranking
            - **Diversity**: Balances relevance with diversity to avoid redundant information
            """)
    
    # Tab: Generation
    with tabs[6]:
        st.markdown('<p class="sub-header">Generation Configuration</p>', unsafe_allow_html=True)
        
        # Prompt template
        prompt_template = st.text_area(
            "Prompt Template",
            value=st.session_state.config["prompt_template"],
            height=150,
            help="Template for generation prompt (use {query} and {context} placeholders)"
        )
        st.session_state.config["prompt_template"] = prompt_template
        
        # Explanation of prompt templates
        with st.expander("About Prompt Templates"):
            st.markdown("""
            **Prompt Templates** guide the generation process. Use these placeholders:
            
            - **{query}**: Will be replaced with the user's question
            - **{context}**: Will be replaced with retrieved documents
            
            Examples:
            - Simple: "Answer based on this context: {context}\\n\\nQuestion: {query}"
            - With instruction: "You are a helpful assistant. Use ONLY the provided context to answer. Context: {context}\\n\\nQuestion: {query}"
            - Citation request: "Answer with citations: {context}\\n\\nQuestion: {query}"
            """)
    
    # Button to apply configuration changes
    if st.button("Apply Configuration Changes", use_container_width=True):
        # Clear any existing RAG app
        st.session_state.rag_app = None
        
        # Initialize new RAG app with the updated config
        initialize_rag_app()
        
        if st.session_state.corpus_uploaded:
            # Prepare the knowledge base with new configuration
            with st.spinner("Rebuilding knowledge base with new configuration..."):
                st.session_state.rag_app.prepare_knowledge_base(force_rebuild=True)
                
            st.success("Configuration applied successfully!")
        else:
            st.warning("Configuration saved, but no knowledge base loaded.")

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
        
    # Performance metrics
    st.markdown('<p class="sub-header">Performance Metrics</p>', unsafe_allow_html=True)
    
    # Create metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        avg_time = sum(exchange["time"] for exchange in st.session_state.conversation_history) / len(st.session_state.conversation_history)
        st.metric("Average Response Time", f"{avg_time:.2f} sec")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        total_queries = len(st.session_state.conversation_history)
        st.metric("Total Queries", total_queries)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        avg_context_len = sum(len(exchange["contexts"]) for exchange in st.session_state.conversation_history) / len(st.session_state.conversation_history)
        st.metric("Avg. Retrieved Documents", f"{avg_context_len:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Query history visualization
    st.markdown("### Query History")
    
    # Create DataFrame from conversation history
    history_data = []
    for i, exchange in enumerate(st.session_state.conversation_history):
        history_data.append({
            "Query ID": i + 1,
            "Query": exchange["query"],
            "Response Time (sec)": exchange["time"],
            "Retrieved Documents": len(exchange["contexts"])
        })
        
    history_df = pd.DataFrame(history_data)
    
    # Display as a table
    st.dataframe(history_df, use_container_width=True)
    
    # Response time visualization
    st.markdown("### Response Time Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x="Query ID", y="Response Time (sec)", data=history_df, marker='o', ax=ax)
    ax.set_title("Response Time per Query")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Convert plot to image for Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.image(buf)
    
    # Document retrieval visualization
    st.markdown("### Document Retrieval Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Query ID", y="Retrieved Documents", data=history_df, ax=ax)
    ax.set_title("Number of Retrieved Documents per Query")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Convert plot to image for Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.image(buf)
    
    # Export metrics
    st.markdown("### Export Metrics")
    
    if st.button("Export Metrics to CSV"):
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="rag_metrics.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

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
                query_embedding = EmbeddingProvider.get_sentence_transformer_embeddings(
                    [test_query], model_name=base_app.config["embedding_model"]
                )[0]
                
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
                    temp_app.chunked_docs = base_app.chunked_docs
                    temp_app.doc_embeddings = base_app.doc_embeddings
                    temp_app.doc_ids = base_app.doc_ids
                    
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
                    temp_app.chunked_docs = base_app.chunked_docs
                    temp_app.doc_embeddings = base_app.doc_embeddings
                    temp_app.doc_ids = base_app.doc_ids
                    
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
                    temp_app.chunked_docs = base_app.chunked_docs
                    temp_app.doc_embeddings = base_app.doc_embeddings
                    temp_app.doc_ids = base_app.doc_ids
                    
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