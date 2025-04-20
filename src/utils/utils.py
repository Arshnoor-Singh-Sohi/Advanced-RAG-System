"""
General utility functions for the RAG system.
"""

import os
import json
from typing import Dict, Any, List, Optional

def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to ensure exists
    """
    os.makedirs(directory_path, exist_ok=True)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Ensure the directory exists
    ensure_directory(os.path.dirname(file_path))
    
    # Save the data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_lexical_diversity(text: str) -> float:
    """
    Calculate lexical diversity (ratio of unique words to total words).
    
    Args:
        text: Text to analyze
        
    Returns:
        Lexical diversity score between 0 and 1
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)