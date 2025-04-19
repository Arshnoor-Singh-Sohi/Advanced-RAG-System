import nltk
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class AdvancedSemanticChunker:
    """Advanced semantic chunking implementation for RAG systems"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified embedding model"""
        self.embedding_model = SentenceTransformer(model_name)
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_by_semantic_segmentation(self, documents: List[Dict[str, str]], 
                                      target_chunk_size: int = 250,
                                      min_chunk_size: int = 100) -> List[Dict[str, str]]:
        """
        Chunk documents by detecting semantic boundaries using embedding similarity
        
        Args:
            documents: List of documents with 'text' field
            target_chunk_size: Target word count per chunk
            min_chunk_size: Minimum chunk size to maintain
            
        Returns:
            List of chunked documents with original metadata preserved
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            # Split by sentences
            sentences = nltk.sent_tokenize(text)
            
            # If very few sentences, treat each as a chunk
            if len(sentences) <= 3:
                chunk_doc = doc.copy()
                chunk_doc['text'] = text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_s0"
                chunked_docs.append(chunk_doc)
                continue
            
            # Calculate sentence embeddings
            sentence_embeddings = self.embedding_model.encode(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(
                    [sentence_embeddings[i]], 
                    [sentence_embeddings[i+1]]
                )[0][0]
                similarities.append(sim)
            
            # Find potential break points (local minima in similarity)
            break_points = []
            for i in range(1, len(similarities) - 1):
                if similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1]:
                    break_points.append(i+1)  # +1 because we want the index of the next sentence
            
            # If no natural break points, fall back to size-based chunking
            if not break_points:
                break_points = [i for i in range(target_chunk_size, len(sentences), target_chunk_size)]
            
            # Create chunks based on break points
            chunks = []
            start_idx = 0
            
            # Sort break points for sequential processing
            break_points.sort()
            
            for bp in break_points:
                # Skip if the resulting chunk would be too small
                if bp - start_idx < min_chunk_size // 10:  # Assuming average 10 words per sentence
                    continue
                    
                # Create chunk
                chunk_text = " ".join(sentences[start_idx:bp])
                chunks.append(chunk_text)
                start_idx = bp
            
            # Don't forget the last chunk
            if start_idx < len(sentences):
                last_chunk = " ".join(sentences[start_idx:])
                chunks.append(last_chunk)
            
            # Create document objects for each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_semantic{i}"
                chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def chunk_by_topic_modeling(self, documents: List[Dict[str, str]], 
                               num_topics: int = 5) -> List[Dict[str, str]]:
        """
        Chunk documents by detecting topic boundaries using TF-IDF
        
        Args:
            documents: List of documents with 'text' field
            num_topics: Number of topics to consider for segmentation
            
        Returns:
            List of chunked documents with original metadata preserved
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            # Split by paragraphs for initial segmentation
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # If very few paragraphs, return as is
            if len(paragraphs) <= 3:
                chunk_doc = doc.copy()
                chunk_doc['text'] = text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_t0"
                chunked_docs.append(chunk_doc)
                continue
            
            # Create TF-IDF vectors for paragraphs
            vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
            
            try:
                tfidf_matrix = vectorizer.fit_transform(paragraphs)
                
                # Calculate similarity between consecutive paragraphs
                similarities = []
                for i in range(len(paragraphs) - 1):
                    sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
                    similarities.append(sim)
                
                # Find topic boundaries (low similarity points)
                boundaries = []
                threshold = np.mean(similarities) - 0.5 * np.std(similarities)
                
                for i, sim in enumerate(similarities):
                    if sim < threshold:
                        boundaries.append(i+1)  # +1 to get the index of the next paragraph
                
                # Ensure we don't have too many boundaries
                if len(boundaries) > num_topics - 1:
                    # Keep only the lowest similarity points
                    boundary_scores = [(i, similarities[i-1]) for i in boundaries]
                    boundary_scores.sort(key=lambda x: x[1])  # Sort by similarity (ascending)
                    boundaries = [i for i, _ in boundary_scores[:num_topics-1]]
                    boundaries.sort()  # Sort boundaries by position
                
                # Create chunks based on boundaries
                chunks = []
                start_idx = 0
                
                for b in boundaries:
                    chunk_text = "\n\n".join(paragraphs[start_idx:b])
                    chunks.append(chunk_text)
                    start_idx = b
                
                # Add the last chunk
                chunk_text = "\n\n".join(paragraphs[start_idx:])
                chunks.append(chunk_text)
                
            except Exception as e:
                # Fallback to simple paragraph chunking if TF-IDF fails
                print(f"Topic modeling failed with error: {e}. Falling back to paragraph chunking.")
                chunks = paragraphs
            
            # Create document objects for each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_topic{i}"
                chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def chunk_by_sliding_window_with_coherence(self, documents: List[Dict[str, str]],
                                             window_size: int = 200,
                                             stride: int = 100,
                                             coherence_threshold: float = 0.7) -> List[Dict[str, str]]:
        """
        Create chunks using sliding window approach but ensure semantic coherence
        
        Args:
            documents: List of documents with 'text' field
            window_size: Size of sliding window in words
            stride: Step size for window in words
            coherence_threshold: Minimum coherence score to split a chunk
            
        Returns:
            List of chunked documents with original metadata preserved
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            # Tokenize by words
            words = text.split()
            
            # If document is smaller than window, return as single chunk
            if len(words) <= window_size:
                chunk_doc = doc.copy()
                chunk_doc['text'] = text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_sw0"
                chunked_docs.append(chunk_doc)
                continue
            
            # Create initial window chunks
            initial_chunks = []
            
            for i in range(0, len(words) - window_size + 1, stride):
                chunk_text = " ".join(words[i:i+window_size])
                initial_chunks.append(chunk_text)
            
            # Ensure the last chunk is included if it doesn't align with stride
            if (len(words) - window_size) % stride != 0:
                last_chunk = " ".join(words[-(window_size):])
                initial_chunks.append(last_chunk)
            
            # Check coherence between consecutive chunks and merge if needed
            merged_chunks = []
            current_chunk = initial_chunks[0]
            
            for i in range(1, len(initial_chunks)):
                # Get embeddings for current_chunk and next chunk
                embeddings = self.embedding_model.encode([current_chunk, initial_chunks[i]])
                
                # Check coherence using cosine similarity
                coherence = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                if coherence >= coherence_threshold:
                    # Merge chunks by taking the union of their text spans
                    start_of_current = len(merged_chunks) * stride
                    end_of_current = start_of_current + len(current_chunk.split())
                    end_of_next = (i + 1) * stride + window_size
                    
                    # Merged chunk is from start of current to end of next
                    merged_text = " ".join(words[start_of_current:min(end_of_next, len(words))])
                    current_chunk = merged_text
                else:
                    # Save current chunk and start a new one
                    merged_chunks.append(current_chunk)
                    current_chunk = initial_chunks[i]
            
            # Don't forget to add the last chunk
            merged_chunks.append(current_chunk)
            
            # Create document objects for each chunk
            for i, chunk_text in enumerate(merged_chunks):
                chunk_doc = doc.copy()
                chunk_doc['text'] = chunk_text
                chunk_doc['chunk_id'] = f"{doc.get('title', 'doc')}_coherent{i}"
                chunked_docs.append(chunk_doc)
        
        return chunked_docs

# Example usage:
# chunker = AdvancedSemanticChunker()
# semantic_chunks = chunker.chunk_by_semantic_segmentation(documents)
# topic_chunks = chunker.chunk_by_topic_modeling(documents)
# coherent_chunks = chunker.chunk_by_sliding_window_with_coherence(documents)