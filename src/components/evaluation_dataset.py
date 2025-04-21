"""
Evaluation Dataset Module

This module provides utilities for creating, loading, and using evaluation datasets
for RAG systems. It includes:
- Dataset format definitions
- Example dataset creation
- Dataset loading and processing
- Evaluation metrics calculation
"""

import os
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Define the default dataset directory
DEFAULT_DATASET_DIR = os.path.join("data", "evaluation")

class EvaluationDataset:
    """Class for managing RAG evaluation datasets"""
    
    def __init__(self, name: str = "default"):
        """
        Initialize evaluation dataset
        
        Args:
            name: Name of the dataset
        """
        self.name = name
        self.questions = []
        self.dataset_path = None
        
        # Create evaluation data directory if it doesn't exist
        os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)
    
    def add_question(self, 
                    question: str, 
                    expected_answer: str, 
                    relevant_docs: Optional[List[str]] = None,
                    category: str = "general",
                    difficulty: str = "medium"):
        """
        Add a question to the evaluation dataset
        
        Args:
            question: Question text
            expected_answer: Expected answer
            relevant_docs: Optional list of relevant document IDs/titles
            category: Question category (e.g., 'general', 'technical')
            difficulty: Question difficulty ('easy', 'medium', 'hard')
        """
        # Create question entry
        question_entry = {
            "question": question,
            "expected_answer": expected_answer,
            "relevant_docs": relevant_docs or [],
            "category": category,
            "difficulty": difficulty
        }
        
        # Add to questions list
        self.questions.append(question_entry)
        
        print(f"Added question: '{question[:50]}{'...' if len(question) > 50 else ''}'")
    
    def save(self, filepath: Optional[str] = None):
        """
        Save the evaluation dataset to disk
        
        Args:
            filepath: Optional custom filepath, otherwise uses default location
        """
        if not filepath:
            # Use default location: data/evaluation/{name}_{timestamp}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
            filepath = os.path.join(DEFAULT_DATASET_DIR, filename)
        
        # Create dataset object
        dataset = {
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "questions": self.questions
        }
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        self.dataset_path = filepath
        print(f"Saved evaluation dataset to {filepath} with {len(self.questions)} questions")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load evaluation dataset from disk
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            EvaluationDataset instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        # Load from file
        with open(filepath, 'r') as f:
            dataset = json.load(f)
            
        # Create dataset instance
        instance = cls(name=dataset.get("name", "loaded_dataset"))
        instance.questions = dataset.get("questions", [])
        instance.dataset_path = filepath
        
        print(f"Loaded evaluation dataset from {filepath} with {len(instance.questions)} questions")
        return instance
    
    @classmethod
    def from_csv(cls, filepath: str, name: Optional[str] = None):
        """
        Create evaluation dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            name: Optional name for the dataset
            
        Returns:
            EvaluationDataset instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
            
        # Create dataset instance
        dataset_name = name or os.path.splitext(os.path.basename(filepath))[0]
        instance = cls(name=dataset_name)
        
        # Load questions from CSV
        try:
            df = pd.read_csv(filepath)
            required_columns = ["question", "expected_answer"]
            
            # Check required columns
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV file must have '{col}' column")
                    
            # Optional columns
            has_relevant_docs = "relevant_docs" in df.columns
            has_category = "category" in df.columns
            has_difficulty = "difficulty" in df.columns
            
            # Add questions
            for _, row in df.iterrows():
                relevant_docs = row["relevant_docs"].split("|") if has_relevant_docs and pd.notna(row["relevant_docs"]) else []
                category = row["category"] if has_category and pd.notna(row["category"]) else "general"
                difficulty = row["difficulty"] if has_difficulty and pd.notna(row["difficulty"]) else "medium"
                
                instance.add_question(
                    question=row["question"],
                    expected_answer=row["expected_answer"],
                    relevant_docs=relevant_docs,
                    category=category,
                    difficulty=difficulty
                )
                
            print(f"Created evaluation dataset from CSV with {len(instance.questions)} questions")
            return instance
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    def create_template_csv(self, filepath: str):
        """
        Create a template CSV file for evaluation dataset
        
        Args:
            filepath: Path to save the template
        """
        # Define columns
        columns = ["question", "expected_answer", "relevant_docs", "category", "difficulty"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create CSV file
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            
            # Add example row
            writer.writerow([
                "What is RAG?",
                "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval systems with generative models.",
                "intro_to_rag.pdf|page_2",
                "technical",
                "easy"
            ])
            
        print(f"Created template CSV file at {filepath}")
    
    def filter_by_category(self, category: str):
        """
        Filter questions by category
        
        Args:
            category: Category to filter by
            
        Returns:
            New EvaluationDataset instance with filtered questions
        """
        filtered = EvaluationDataset(name=f"{self.name}_{category}")
        
        for question in self.questions:
            if question["category"] == category:
                filtered.questions.append(question.copy())
                
        print(f"Filtered {len(self.questions)} questions to {len(filtered.questions)} in category '{category}'")
        return filtered
    
    def filter_by_difficulty(self, difficulty: str):
        """
        Filter questions by difficulty
        
        Args:
            difficulty: Difficulty level to filter by
            
        Returns:
            New EvaluationDataset instance with filtered questions
        """
        filtered = EvaluationDataset(name=f"{self.name}_{difficulty}")
        
        for question in self.questions:
            if question["difficulty"] == difficulty:
                filtered.questions.append(question.copy())
                
        print(f"Filtered {len(self.questions)} questions to {len(filtered.questions)} with difficulty '{difficulty}'")
        return filtered
    
    def sample(self, n: int = 5, seed: Optional[int] = None):
        """
        Get a random sample of questions
        
        Args:
            n: Number of questions to sample
            seed: Random seed for reproducibility
            
        Returns:
            New EvaluationDataset instance with sampled questions
        """
        if n >= len(self.questions):
            return self
            
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Sample questions
        sampled_indices = np.random.choice(len(self.questions), n, replace=False)
        
        # Create new dataset
        sampled = EvaluationDataset(name=f"{self.name}_sample_{n}")
        
        for idx in sampled_indices:
            sampled.questions.append(self.questions[idx].copy())
            
        print(f"Sampled {n} questions from {len(self.questions)} total questions")
        return sampled


def create_example_dataset():
    """
    Create an example evaluation dataset
    
    Returns:
        Example EvaluationDataset
    """
    dataset = EvaluationDataset(name="rag_example")
    
    # Add example questions for RAG evaluation
    dataset.add_question(
        question="What is Retrieval-Augmented Generation (RAG)?",
        expected_answer="Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with information retrieval systems. It enhances the quality of generated text by retrieving relevant information from a knowledge base.",
        category="definition",
        difficulty="easy"
    )
    
    dataset.add_question(
        question="What are the main components of a RAG system?",
        expected_answer="A RAG system typically consists of three main components: 1) A Retriever that finds relevant information, 2) A Generator that creates responses, and 3) A Knowledge Base of documents or facts.",
        category="components",
        difficulty="easy"
    )
    
    dataset.add_question(
        question="How does vector search work in RAG systems?",
        expected_answer="Vector search in RAG systems works by converting text into numerical vectors and finding documents with similar vector representations. This allows for semantic searching beyond simple keyword matching.",
        category="technique",
        difficulty="medium"
    )
    
    dataset.add_question(
        question="What are embedding models and why are they important for RAG?",
        expected_answer="Embedding models transform text into numerical vectors that capture semantic meaning. They are crucial for RAG systems because they enable semantic similarity search, allowing the system to find relevant documents even when the wording differs from the query.",
        category="technique",
        difficulty="medium"
    )
    
    dataset.add_question(
        question="What is document chunking and why is it important in RAG systems?",
        expected_answer="Document chunking is the process of splitting documents into smaller pieces to enable more precise retrieval. It's important for RAG systems because it allows the retriever to find specific relevant sections rather than entire documents, improving the precision of the generated answers.",
        category="technique",
        difficulty="medium"
    )
    
    dataset.add_question(
        question="What is the difference between BM25 and vector search in RAG systems?",
        expected_answer="BM25 is a keyword-based search algorithm that ranks documents based on term frequency and document length, while vector search uses semantic embeddings to find documents with similar meaning regardless of specific keywords. BM25 is better at exact matches, while vector search excels at understanding concepts and semantics.",
        category="comparison",
        difficulty="hard"
    )
    
    dataset.add_question(
        question="What is reranking in the context of RAG, and why is it useful?",
        expected_answer="Reranking in RAG is the process of taking initially retrieved documents and reordering them using more sophisticated (often computationally expensive) models or criteria. It's useful because it improves precision by applying more advanced relevance judgments after the initial retrieval step, which helps provide the generator with higher quality context.",
        category="technique",
        difficulty="hard"
    )
    
    dataset.add_question(
        question="How does the chunking strategy affect RAG performance?",
        expected_answer="The chunking strategy affects RAG performance by determining how documents are split into retrievable pieces. Different strategies (fixed-size, sentence-based, paragraph-based, semantic) influence whether related information stays together or gets split apart. This impacts retrieval precision, as overly large chunks might contain irrelevant information, while too small chunks might lose context.",
        category="performance",
        difficulty="hard"
    )
    
    dataset.add_question(
        question="What is Maximal Marginal Relevance (MMR) in RAG systems?",
        expected_answer="Maximal Marginal Relevance (MMR) is a technique used in RAG systems to balance relevance and diversity in retrieved documents. It works by selecting documents that are both relevant to the query and different from already selected documents, which helps reduce redundancy in the context provided to the generator.",
        category="technique",
        difficulty="hard"
    )
    
    dataset.add_question(
        question="What is HyDE (Hypothetical Document Embeddings) in RAG?",
        expected_answer="HyDE (Hypothetical Document Embeddings) is a technique where the system first generates a hypothetical document that might contain the answer to a query, then uses the embedding of this document for retrieval instead of the query embedding. This bridges the gap between questions and answers in vector space, improving retrieval quality especially for complex queries.",
        category="technique",
        difficulty="hard"
    )
    
    # Save the dataset
    dataset_path = os.path.join(DEFAULT_DATASET_DIR, "rag_example.json")
    dataset.save(dataset_path)
    
    return dataset


def list_available_datasets():
    """
    List all available evaluation datasets in the default directory
    
    Returns:
        List of dataset filepaths
    """
    if not os.path.exists(DEFAULT_DATASET_DIR):
        return []
        
    # Get JSON files
    dataset_files = [
        os.path.join(DEFAULT_DATASET_DIR, f)
        for f in os.listdir(DEFAULT_DATASET_DIR)
        if f.endswith('.json')
    ]
    
    return dataset_files


# If this module is run directly, create an example dataset
if __name__ == "__main__":
    print("Creating example evaluation dataset...")
    dataset = create_example_dataset()
    print(f"Created example dataset with {len(dataset.questions)} questions")
    
    # Also create a template CSV
    template_path = os.path.join(DEFAULT_DATASET_DIR, "template.csv")
    dataset.create_template_csv(template_path)