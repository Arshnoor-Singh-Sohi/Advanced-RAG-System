"""
LLM integration module for the RAG system.

This module provides functions and classes for:
- Interacting with different LLMs (OpenAI, Hugging Face, etc.)
- Creating and managing conversation chains
- Prompting and response generation
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# src/components/llm_integrations.py
import os
import streamlit as st # For error messages potentially
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    # Newer Langchain path
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    # Older Langchain path
    from langchain.embeddings import OpenAIEmbeddings

# Add torch import if you want GPU/MPS detection, otherwise default to CPU
# import torch

# <<< ADD THE FUNCTION HERE (Outside LLMProvider class) >>>
@st.cache_resource(show_spinner="Loading Embedding Model...") # Cache the loaded model
def get_embedding_model(model_identifier: str, config: dict = None):
    """
    Returns the appropriate LangChain embedding object based on identifier.
    Caches the loaded model resource.

    Args:
        model_identifier (str): The identifier of the embedding model (e.g., 'all-MiniLM-L6-v2').
        config (dict, optional): The configuration dictionary, potentially containing API keys.

    Returns:
        A LangChain embedding object (or None if invalid/error).
    """
    config = config or {}

    # --- Define supported models and their types ---
    # Ensure these identifiers match EXACTLY what's in your Streamlit selectbox
    model_map = {
        "all-MiniLM-L6-v2": "huggingface",
        "BAAI/bge-small-en-v1.5": "huggingface",
        "all-mpnet-base-v2": "huggingface",
        "multi-qa-mpnet-base-dot-v1": "huggingface",
        "text-embedding-ada-002": "openai",
        # Add other models from your UI here if needed
        "BAAI/bge-large-en-v1.5": "huggingface" # Added from your simple request example
    }

    model_type = model_map.get(model_identifier)

    if model_type == "huggingface":
        # device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        model_kwargs = {'device': 'cpu'} # Default to CPU
        encode_kwargs = {'normalize_embeddings': True}
        try:
            print(f"Loading HuggingFace Embedding Model: {model_identifier}")
            # Directly use the identifier which is the HF model name
            return HuggingFaceEmbeddings(
                model_name=model_identifier,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except Exception as e:
            st.error(f"Error loading HuggingFace model {model_identifier}: {e}")
            return None

    elif model_type == "openai":
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        if not api_key:
            # Use st.error for visibility in Streamlit if run from there
            st.error("OpenAI API Key required for 'text-embedding-ada-002' but not found in environment variables or config.")
            # Optionally, print for console visibility
            print("ERROR: OpenAI API Key required for 'text-embedding-ada-002' but not found.")
            return None
        try:
            print(f"Loading OpenAI Embedding Model: {model_identifier}")
            # Model name is passed during OpenAIEmbeddings initialization
            return OpenAIEmbeddings(openai_api_key=api_key, model=model_identifier)
        except Exception as e:
            st.error(f"Error loading OpenAI model {model_identifier}: {e}")
            return None

    else:
        st.error(f"Unknown or unsupported embedding model identifier: {model_identifier}")
        return None

def get_conversation_chain(llm, vectorstore):
        """
        Create a conversation chain with retrieval
        
        Args:
            llm: Language model
            vectorstore: Vector store for retrieval
            
        Returns:
            Conversation chain
        """
        try:
            from langchain.chains import ConversationalRetrievalChain
            from langchain.memory import ConversationBufferMemory
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create conversation chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            
            return chain
        except ImportError:
            raise ImportError("langchain package not installed. Install with 'pip install langchain'")
        except Exception as e:
            raise RuntimeError(f"Error creating conversation chain: {str(e)}")

def generate_response(prompt: str, llm) -> str:
    """
    Generate a response using an LLM
    
    Args:
        prompt: Prompt text
        llm: Language model
        
    Returns:
        Generated response
    """
    try:
        # If llm is an OpenAI or similar object with a predict method
        if hasattr(llm, "predict"):
            return llm.predict(prompt)
        # If llm is a function
        elif callable(llm):
            return llm(prompt)
        # Other LLM types
        else:
            return str(llm(prompt))
    except Exception as e:
        raise RuntimeError(f"Error generating response: {str(e)}")

def extract_answer_from_context(query: str, contexts: List[str]) -> str:
    """
    Extract an answer from context using simple heuristics
    
    Args:
        query: Query string
        contexts: List of context strings
        
    Returns:
        Extracted answer
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

class LLMProvider:
    """Provider for different LLM backends"""
    
    @staticmethod
    def get_openai_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Get an OpenAI language model
        
        Args:
            model_name: Name of the OpenAI model
            temperature: Sampling temperature
            
        Returns:
            OpenAI LLM instance
        """
        try:
            from langchain.llms import OpenAI
            from langchain.chat_models import ChatOpenAI
            
            # Check if API key is set
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            # For chat models
            if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
                return ChatOpenAI(model_name=model_name, temperature=temperature)
            # For completion models
            else:
                return OpenAI(model_name=model_name, temperature=temperature)
        except ImportError:
            raise ImportError("langchain package not installed. Install with 'pip install langchain openai'")
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI LLM: {str(e)}")
    
    @staticmethod
    def get_huggingface_llm(model_name: str = "google/flan-t5-base", max_length: int = 512):
        """
        Get a Hugging Face language model
        
        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum length of generated text
            
        Returns:
            Hugging Face LLM instance
        """
        try:
            from langchain.llms import HuggingFacePipeline
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine model type
            if "t5" in model_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                pipe_type = "text2text-generation"
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                pipe_type = "text-generation"
                
            # Create pipeline
            pipe = pipeline(
                pipe_type,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length
            )
            
            # Create LangChain wrapper
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            raise ImportError("langchain and/or transformers packages not installed. Install with 'pip install langchain transformers torch'")
        except Exception as e:
            raise RuntimeError(f"Error initializing Hugging Face LLM: {str(e)}")

    