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