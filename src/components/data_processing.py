"""
Document processing module for the RAG system.

This module provides functions and classes for processing documents, including:
- Chunking documents with different strategies
- Extracting text from different file formats
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import nltk
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter
)
import streamlit as st # Or use print
import PyPDF2 # If get_pdf_text is here
# import docx   # If docx processing is here

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_text_from_file(file_path: str) -> str:
        """
        Extract text from various file formats
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return get_pdf_text(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                raise ImportError("python-docx package not installed. Install with 'pip install python-docx'")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

def create_document_from_text(text: str, title: str = None) -> Dict[str, str]:
    """
    Create a document dictionary from text
    
    Args:
        text: Document text
        title: Optional document title
        
    Returns:
        Document dictionary
    """
    return {
        "title": title or "Untitled Document",
        "text": text
    }

def get_text_chunks(text, strategy_name="recursive", chunk_size=1000, chunk_overlap=200):
    """
    Split text into chunks using the specified strategy.
    
    Args:
        text: The text to split
        strategy_name: The chunking strategy to use (recursive, token, sentence)
        chunk_size: The size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        List of text chunks
    """
    if strategy_name == "recursive":
        # Default recursive character splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
    elif strategy_name == "token":
        # Token-based splitting using SentenceTransformers
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size
        )
        
    elif strategy_name == "sentence":
        # Split by sentences with NLTKTextSplitter
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            from langchain.text_splitter import NLTKTextSplitter
            
            text_splitter = NLTKTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ImportError:
            print("NLTK not available, falling back to recursive splitter")
            return get_text_chunks(text, "recursive", chunk_size, chunk_overlap)
    
    else:
        # Fall back to recursive if unknown strategy
        print(f"Unknown chunking strategy: {strategy_name}. Defaulting to recursive.")
        return get_text_chunks(text, "recursive", chunk_size, chunk_overlap)
    
    chunks = text_splitter.split_text(text)
    return chunks

class DocumentChunker:
    """Class for chunking documents with different strategies"""

    def __init__(self, strategy: str, chunk_size: int, chunk_overlap: int, **kwargs):
        """
        Initializes the chunker by selecting and configuring the appropriate
        LangChain text splitter.
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = None # Will hold the LangChain splitter instance

        print(f"Initializing DocumentChunker - Strategy: {self.strategy}, Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")

        # --- Select splitter based on strategy (Same logic as provided before) ---
        if self.strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                length_function=len, add_start_index=True
            )
        elif self.strategy == "token":
            model_name = kwargs.get("embedding_model_name")
            if model_name:
                try:
                    self.splitter = SentenceTransformersTokenTextSplitter(
                        chunk_overlap=self.chunk_overlap, model_name=model_name,
                        tokens_per_chunk=self.chunk_size
                    )
                except Exception as e:
                    print(f"WARN: Failed SentenceTransformersTokenTextSplitter init, fallback Recursive: {e}")
                    self.strategy = "recursive"
            else:
                print("WARN: 'token' strategy needs 'embedding_model_name', fallback Recursive.")
                self.strategy = "recursive"
            if self.strategy == "recursive" and self.splitter is None:
                self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif self.strategy == "sentence":
            try:
                try: nltk.data.find('tokenizers/punkt')
                except: nltk.download('punkt', quiet=True)
                self.splitter = NLTKTextSplitter(chunk_size=self.chunk_size)
            except ImportError:
                print("WARN: NLTK not found, fallback Recursive for 'sentence'.")
                self.strategy = "recursive"
            except Exception as e:
                print(f"WARN: NLTK error, fallback Recursive: {e}")
                self.strategy = "recursive"
            if self.strategy == "recursive" and self.splitter is None:
                self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif self.strategy == "fixed":
            self.splitter = CharacterTextSplitter(
                separator = "\n", chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap, length_function = len
            )
        elif self.strategy == "paragraph":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True
            )
        elif self.strategy == "semantic":
            print("WARN: 'semantic' chunking fallback to Recursive.")
            self.strategy = "recursive"
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else: # Default fallback
            print(f"WARN: Unknown strategy '{self.strategy}', fallback Recursive.")
            self.strategy = "recursive"
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        if self.splitter is None:
            raise ValueError(f"Could not initialize text splitter for strategy: {self.strategy}")
    
    @staticmethod
    def chunk_by_fixed_size(documents: List[Dict[str, str]], 
                        chunk_size: int,
                        overlap: int = 0) -> List[Dict[str, str]]:
        """
        Chunk documents by fixed token size
        
        Args:
            documents: List of documents with 'text' and other metadata
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            # Simple tokenization by words (in production you'd use a better tokenizer)
            tokens = nltk.word_tokenize(text)
            
            # Create chunks with the specified size and overlap
            for i in range(0, len(tokens), chunk_size - overlap):
                # Get chunk tokens
                chunk_tokens = tokens[i:i + chunk_size]
                
                # Skip chunks that are too small (last partial chunks)
                if len(chunk_tokens) < chunk_size // 2:
                    continue
                    
                # Combine tokens back into text
                chunk_text = " ".join(chunk_tokens)
                
                # Create a new document with the chunk and original metadata
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_{i//(chunk_size - overlap)}"
                
                chunked_docs.append(chunk_doc)
                
        return chunked_docs
    
    @staticmethod
    def chunk_by_paragraph(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk documents by paragraph breaks
        
        Args:
            documents: List of documents with 'text' and other metadata
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            
            # Split by paragraph breaks (double newline)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # If no clear paragraphs found, try single newlines
            if len(paragraphs) <= 1:
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
                
            # If still no clear paragraphs, keep as single chunk
            if len(paragraphs) <= 1:
                chunked_docs.append(doc)
                continue
                
            # Create new documents for each paragraph
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.split()) < 10:  # Skip very short paragraphs
                    continue
                    
                chunk_doc = doc.copy()
                chunk_doc['text'] = paragraph
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_p{i}"
                
                chunked_docs.append(chunk_doc)
                
        return chunked_docs
    
    @staticmethod
    def chunk_by_semantic_units(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chunk documents by trying to preserve semantic units
        This is a simplified version that uses sentence boundaries and length limits
        
        Args:
            documents: List of documents with 'text' and other metadata
        
        Returns:
            List of chunked documents with original metadata preserved
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        chunked_docs = []
        target_chunk_size = 300  # Target word count per chunk
        
        for doc in documents:
            text = doc['text']
            
            # Split by sentences
            sentences = nltk.sent_tokenize(text)
            
            current_chunk = []
            current_size = 0
            chunk_num = 0
            
            for sentence in sentences:
                sentence_len = len(sentence.split())
                
                # If adding this sentence exceeds target size and we already have content,
                # finish the current chunk
                if current_size > 0 and current_size + sentence_len > target_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunk_doc = doc.copy()
                    chunk_doc['text'] = chunk_text
                    chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_s{chunk_num}"
                    chunked_docs.append(chunk_doc)
                    
                    # Start a new chunk
                    current_chunk = [sentence]
                    current_size = sentence_len
                    chunk_num += 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_size += sentence_len
                    
            # Don't forget to add the last chunk if it has content
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_s{chunk_num}"
                chunked_docs.append(chunk_doc)
                
        return chunked_docs


    def get_pdf_text(pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            import PyPDF2
            
            text = ""
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                    
                # Process each page
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                    
            return text
        except ImportError:
            raise ImportError("PyPDF2 package not installed. Install with 'pip install PyPDF2'")
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

    def chunk_document(self, document: Dict[str, Any]) -> List[str]:
        """
        Chunks a single document dictionary using the initialized splitter.

        Args:
            document (Dict[str, Any]): A dictionary representing the document, must contain 'text' key.

        Returns:
            List[str]: A list of chunked text strings.
                       Returns empty list if input text is empty or splitter failed.
        """
        text = document.get("text", "")
        if not text:
            return []
        if self.splitter is None: # Check if splitter was initialized in __init__
            print(f"ERROR: Splitter not initialized for document chunking (strategy: {self.strategy}). Skipping document.")
            # Optionally raise an error or use st.error if available/desired
            return []

        try:
            # Use the LangChain splitter instance stored in self.splitter
            return self.splitter.split_text(text)
        except Exception as e:
            print(f"ERROR: Failed to chunk document (Title: {document.get('title', 'N/A')}) using strategy '{self.strategy}': {e}")
            # Optionally log traceback here
            return [] # Return empty list on error for this document

    # Add this method (called by rag_app.py)
    def chunk_corpus(self, corpus: List[Dict[str, Any]]) -> List[str]:
        """
        Chunks all documents in a corpus using the chunk_document method.

        Args:
            corpus (List[Dict[str, Any]]): The list of document dictionaries.

        Returns:
            List[str]: A flat list of all chunked text strings from the corpus.
        """
        all_chunks = []
        if self.splitter is None:
             print("ERROR: Cannot chunk corpus because splitter is not initialized.")
             # Optionally use st.error if available/desired
             return []

        print(f"Chunking {len(corpus)} documents using instance method...")
        for i, doc in enumerate(corpus):
            doc_chunks = self.chunk_document(doc) # Call the instance method
            all_chunks.extend(doc_chunks)
            # Optional: Add progress update
            # if (i + 1) % 10 == 0: print(f"  Processed {i+1}/{len(corpus)} documents...")
        print(f"Finished chunking corpus. Total chunks: {len(all_chunks)}")
        return all_chunks
