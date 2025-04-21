"""
Document processing module for the RAG system.

This module provides functions and classes for processing documents, including:
- Chunking documents with different strategies
- Extracting text from different file formats
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentChunker:
    """Class for chunking documents with different strategies"""
    
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