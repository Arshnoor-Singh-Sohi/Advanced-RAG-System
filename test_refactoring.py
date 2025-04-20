"""
Test script to verify the refactored RAG components are working correctly.
This script tests both backward compatibility and direct imports from the new modules.
"""

import os
import sys
import numpy as np


# Create a dummy vectorstore-like object for testing
class DummyVectorStore:
    def __init__(self, name):
        self.name = name
        self.vectors = np.random.rand(10, 10)
    
    def save_local(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_backward_compatibility():
    """Test that the original import paths still work."""
    print("Testing backward compatibility...")
    
    # Import from original module
    from src.components.rag_components import DocumentChunker, EmbeddingProvider, RetrievalMethods
    
    # Test document chunking
    sample_docs = [
        {"title": "Test Doc 1", "text": "This is a test document with some content for chunking."},
        {"title": "Test Doc 2", "text": "Another test document that will be processed by the chunker."}
    ]
    
    # Test fixed-size chunking
    fixed_chunks = DocumentChunker.chunk_by_fixed_size(sample_docs, chunk_size=5, overlap=0)
    print(f"Fixed-size chunking created {len(fixed_chunks)} chunks")
    
    # Test paragraph chunking
    para_chunks = DocumentChunker.chunk_by_paragraph(sample_docs)
    print(f"Paragraph chunking created {len(para_chunks)} chunks")
    
    print("Backward compatibility tests passed!")
    return True

def test_direct_imports():
    """Test importing directly from the new modules."""
    print("\nTesting direct imports from new modules...")
    
    # Import from new modules
    from src.components.data_processing import DocumentChunker
    from src.components.vectorstore_handler import EmbeddingProvider, RetrievalMethods
    from src.utils.utils import calculate_lexical_diversity
    
    # Test document chunking from data_processing
    sample_docs = [
        {"title": "Test Doc 1", "text": "This is a test document with some content for chunking."},
        {"title": "Test Doc 2", "text": "Another test document that will be processed by the chunker."}
    ]
    
    chunks = DocumentChunker.chunk_by_semantic_units(sample_docs)
    print(f"Semantic chunking created {len(chunks)} chunks")
    
    # Test utility function
    diversity = calculate_lexical_diversity("This is a test of lexical diversity with repeated words like test")
    print(f"Lexical diversity: {diversity:.2f}")
    
    print("Direct import tests passed!")
    return True



def test_save_load_vectorstore():
    """Test save and load functionality for vector stores."""
    print("\nTesting vector store save/load functionality...")
    
    from src.components.vectorstore_handler import save_vectorstore, load_vectorstore
    

    
    # Create test directory
    os.makedirs("test_data", exist_ok=True)
    test_path = "test_data/test_vectorstore.pkl"
    
    # Create and save dummy vectorstore
    dummy_store = DummyVectorStore("test_store")
    save_vectorstore(dummy_store, test_path)
    print(f"Saved vector store to {test_path}")
    
    # Try to load it back
    try:
        loaded_store = load_vectorstore(test_path)
        print(f"Successfully loaded vector store: {loaded_store.name}")
        
        # Clean up
        os.remove(test_path)
        print("Test passed!")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting refactoring tests...\n")
    
    # Run backward compatibility tests
    backward_compat = test_backward_compatibility()
    
    # Run direct import tests
    direct_imports = test_direct_imports()
    
    # Run vectorstore tests
    vectorstore_test = test_save_load_vectorstore()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Backward Compatibility: {'✓' if backward_compat else '✗'}")
    print(f"Direct Imports: {'✓' if direct_imports else '✗'}")
    print(f"Vector Store Save/Load: {'✓' if vectorstore_test else '✗'}")
    
    if backward_compat and direct_imports and vectorstore_test:
        print("\nAll tests passed! The refactoring appears to be successful.")
    else:
        print("\nSome tests failed. Please check the output for details.")