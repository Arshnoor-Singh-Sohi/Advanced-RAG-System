"""
Evaluation Runner Module

This module provides an evaluation runner for RAG systems:
- Running evaluations on datasets
- Calculating metrics for different configurations
- Comparing different RAG configurations
- Generating evaluation reports
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import traceback

# Import from other modules
from src.components.evaluation import RAGEvaluator
from src.components.evaluation_dataset import EvaluationDataset, DEFAULT_DATASET_DIR

# Default results directory
DEFAULT_RESULTS_DIR = os.path.join("results", "evaluation")

class RAGEvaluationRunner:
    """Runner for comprehensive RAG system evaluations"""
    
    def __init__(self, rag_app, dataset: Optional[EvaluationDataset] = None):
        """
        Initialize evaluation runner
        
        Args:
            rag_app: RAG application instance
            dataset: Optional evaluation dataset
        """
        self.rag_app = rag_app
        self.dataset = dataset
        self.results = []
        self.evaluator = RAGEvaluator()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    
    def load_dataset(self, dataset_path: str):
        """
        Load evaluation dataset
        
        Args:
            dataset_path: Path to dataset file
        """
        self.dataset = EvaluationDataset.load(dataset_path)
        print(f"Loaded evaluation dataset with {len(self.dataset.questions)} questions")
    
    def run_evaluation(self, 
                      config_name: str = "default",
                      max_questions: Optional[int] = None,
                      show_progress: bool = True):
        """
        Run evaluation on the loaded dataset
        
        Args:
            config_name: Name for this configuration
            max_questions: Maximum number of questions to evaluate
            show_progress: Whether to show a progress bar
        """
        if not self.dataset or not self.dataset.questions:
            raise ValueError("No evaluation dataset loaded")
            
        if not self.rag_app:
            raise ValueError("No RAG application provided")
            
        questions = self.dataset.questions
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
            
        print(f"Running evaluation on {len(questions)} questions with config '{config_name}'")
        
        # Store current configuration
        current_config = self.rag_app.config.copy() if hasattr(self.rag_app, 'config') else {}
        
        # Initialize results for this run
        run_results = {
            "config_name": config_name,
            "config": current_config,
            "timestamp": datetime.now().isoformat(),
            "dataset_name": self.dataset.name if self.dataset else "unknown",
            "question_results": []
        }
        
        # Track overall metrics
        all_retrieval_times = []
        all_answer_times = []
        all_precisions = []
        all_recalls = []
        all_mrrs = []
        all_rouge_scores = []
        all_faithfulness_scores = []
        
        # Create progress bar if requested
        questions_iter = tqdm(questions, desc="Evaluating") if show_progress else questions
        
        # Process each question
        for i, question_item in enumerate(questions_iter):
            question = question_item["question"]
            expected_answer = question_item["expected_answer"]
            relevant_docs = question_item.get("relevant_docs", [])
            category = question_item.get("category", "general")
            difficulty = question_item.get("difficulty", "medium")
            
            # Track metrics for this question
            retrieval_time = 0
            answer_time = 0
            retrieved_docs = []
            
            try:
                # Process query and time it
                start_time = time.time()
                
                # Get answer and retrieved documents
                answer, contexts = self.rag_app.process_query(question)
                
                # Calculate times
                total_time = time.time() - start_time
                # Assume retrieval is 70% of total time (approximation)
                retrieval_time = total_time * 0.7
                answer_time = total_time * 0.3
                
                # Get document IDs (using title as ID)
                retrieved_docs = [context.get("title", f"doc_{i}") for i, context in enumerate(contexts)]
                retrieved_texts = [context.get("text", "") for context in contexts]
                
                # Calculate retrieval metrics
                precision = self.evaluator.precision_at_k(relevant_docs, retrieved_docs, k=len(retrieved_docs))
                recall = self.evaluator.recall_at_k(relevant_docs, retrieved_docs, k=len(retrieved_docs))
                mrr = self.evaluator.mean_reciprocal_rank(relevant_docs, retrieved_docs)
                
                # Calculate answer quality metrics
                rouge_scores = self.evaluator.rouge_scores(answer, expected_answer)
                rouge1 = rouge_scores.get("rouge1", 0)
                
                # Faithfulness (how well the answer uses the retrieved information)
                faithfulness = self.evaluator.faithfulness_score(answer, retrieved_texts)
                
                # Record metrics
                all_retrieval_times.append(retrieval_time)
                all_answer_times.append(answer_time)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_mrrs.append(mrr)
                all_rouge_scores.append(rouge1)
                all_faithfulness_scores.append(faithfulness)
                
                # Store result for this question
                question_result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": answer,
                    "retrieved_docs": retrieved_docs,
                    "relevant_docs": relevant_docs,
                    "retrieval_time": retrieval_time,
                    "answer_time": answer_time,
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "rouge1": rouge1,
                    "faithfulness": faithfulness,
                    "category": category,
                    "difficulty": difficulty
                }
                
                run_results["question_results"].append(question_result)
                
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
                traceback.print_exc()
                
                # Record failure
                question_result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": f"ERROR: {str(e)}",
                    "retrieved_docs": [],
                    "relevant_docs": relevant_docs,
                    "retrieval_time": 0,
                    "answer_time": 0,
                    "precision": 0,
                    "recall": 0,
                    "mrr": 0,
                    "rouge1": 0,
                    "faithfulness": 0,
                    "category": category,
                    "difficulty": difficulty,
                    "error": str(e)
                }
                
                run_results["question_results"].append(question_result)
        
        # Calculate aggregate metrics
        run_results["aggregate_metrics"] = {
            "avg_retrieval_time": np.mean(all_retrieval_times) if all_retrieval_times else 0,
            "avg_answer_time": np.mean(all_answer_times) if all_answer_times else 0,
            "avg_precision": np.mean(all_precisions) if all_precisions else 0,
            "avg_recall": np.mean(all_recalls) if all_recalls else 0,
            "avg_mrr": np.mean(all_mrrs) if all_mrrs else 0,
            "avg_rouge1": np.mean(all_rouge_scores) if all_rouge_scores else 0,
            "avg_faithfulness": np.mean(all_faithfulness_scores) if all_faithfulness_scores else 0,
            "question_count": len(questions)
        }
        
        # Print summary metrics
        print(f"\nEvaluation Summary for '{config_name}':")
        for metric, value in run_results["aggregate_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
                
        # Add to results
        self.results.append(run_results)
        
        # Save results
        self._save_results()
        
        return run_results
    
    def _save_results(self):
        """Save evaluation results to disk"""
        if not self.results:
            return
            
        # Create results file path
        results_file = os.path.join(DEFAULT_RESULTS_DIR, f"evaluation_{self.run_id}.json")
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Saved evaluation results to {results_file}")
    
    def generate_report(self, save_path: Optional[str] = None, show_plots: bool = False):
        """
        Generate evaluation report with visualizations
        
        Args:
            save_path: Optional path to save the report
            show_plots: Whether to display plots (for notebooks)
            
        Returns:
            Path to the report directory
        """
        if not self.results:
            print("No evaluation results to report")
            return None
            
        # Create report directory
        if save_path:
            report_dir = save_path
        else:
            report_dir = os.path.join(DEFAULT_RESULTS_DIR, f"report_{self.run_id}")
            
        os.makedirs(report_dir, exist_ok=True)
        
        # Create summary dataframe for each configuration
        config_summaries = []
        
        for result in self.results:
            config_name = result["config_name"]
            metrics = result["aggregate_metrics"]
            
            summary = {
                "config_name": config_name,
                **metrics
            }
            
            config_summaries.append(summary)
            
        # Create summary dataframe
        summary_df = pd.DataFrame(config_summaries)
        
        # Save summary to CSV
        summary_csv = os.path.join(report_dir, "summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        # Create detailed dataframe for all questions across configurations
        all_questions = []
        
        for result in self.results:
            config_name = result["config_name"]
            
            for q_result in result["question_results"]:
                question_data = {
                    "config_name": config_name,
                    **{k: v for k, v in q_result.items() if k not in ["retrieved_docs", "relevant_docs"]}
                }
                
                all_questions.append(question_data)
                
        # Create question results dataframe
        questions_df = pd.DataFrame(all_questions)
        
        # Save detailed results to CSV
        details_csv = os.path.join(report_dir, "question_details.csv")
        questions_df.to_csv(details_csv, index=False)
        
        # Generate visualizations
        self._generate_summary_plots(summary_df, questions_df, report_dir, show_plots)
        
        print(f"Generated evaluation report in {report_dir}")
        return report_dir
    
    def _generate_summary_plots(self, summary_df, questions_df, report_dir, show_plots):
        """
        Generate summary plots for the report
        
        Args:
            summary_df: Summary dataframe
            questions_df: Questions dataframe
            report_dir: Directory to save plots
            show_plots: Whether to display plots
        """
        # Create plots directory
        plots_dir = os.path.join(report_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Retrieval metrics comparison
        plt.figure(figsize=(12, 6))
        metrics = ['avg_precision', 'avg_recall', 'avg_mrr']
        
        # Reshape data for plotting
        plot_data = []
        for _, row in summary_df.iterrows():
            for metric in metrics:
                plot_data.append({
                    'Configuration': row['config_name'],
                    'Metric': metric.replace('avg_', ''),
                    'Value': row[metric]
                })
                
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        sns.barplot(x='Configuration', y='Value', hue='Metric', data=plot_df)
        plt.title('Retrieval Metrics by Configuration')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, 'retrieval_metrics.png'))
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot 2: Answer quality metrics
        plt.figure(figsize=(12, 6))
        metrics = ['avg_rouge1', 'avg_faithfulness']
        
        # Reshape data for plotting
        plot_data = []
        for _, row in summary_df.iterrows():
            for metric in metrics:
                plot_data.append({
                    'Configuration': row['config_name'],
                    'Metric': metric.replace('avg_', ''),
                    'Value': row[metric]
                })
                
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        sns.barplot(x='Configuration', y='Value', hue='Metric', data=plot_df)
        plt.title('Answer Quality Metrics by Configuration')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, 'answer_quality.png'))
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot 3: Performance metrics
        plt.figure(figsize=(12, 6))
        metrics = ['avg_retrieval_time', 'avg_answer_time']
        
        # Reshape data for plotting
        plot_data = []
        for _, row in summary_df.iterrows():
            for metric in metrics:
                plot_data.append({
                    'Configuration': row['config_name'],
                    'Metric': metric.replace('avg_', ''),
                    'Value': row[metric]
                })
                
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        sns.barplot(x='Configuration', y='Value', hue='Metric', data=plot_df)
        plt.title('Performance Metrics by Configuration (seconds)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, 'performance.png'))
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot 4: Difficulty analysis
        if 'difficulty' in questions_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Calculate average precision by difficulty and configuration
            difficulty_df = questions_df.groupby(['config_name', 'difficulty'])['precision'].mean().reset_index()
            
            sns.barplot(x='difficulty', y='precision', hue='config_name', data=difficulty_df)
            plt.title('Precision by Difficulty Level')
            plt.ylim(0, 1)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, 'difficulty_analysis.png'))
            if show_plots:
                plt.show()
            plt.close()
            
        # Plot 5: Category analysis
        if 'category' in questions_df.columns and len(questions_df['category'].unique()) > 1:
            plt.figure(figsize=(14, 6))
            
            # Calculate average precision by category and configuration
            category_df = questions_df.groupby(['config_name', 'category'])['precision'].mean().reset_index()
            
            sns.barplot(x='category', y='precision', hue='config_name', data=category_df)
            plt.title('Precision by Question Category')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, 'category_analysis.png'))
            if show_plots:
                plt.show()
            plt.close()
    
    def compare_configurations(self, config_names: Optional[List[str]] = None):
        """
        Compare different configurations
        
        Args:
            config_names: Optional list of configuration names to compare
            
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            print("No evaluation results to compare")
            return None
            
        # Filter results if config_names provided
        if config_names:
            filtered_results = [r for r in self.results if r["config_name"] in config_names]
        else:
            filtered_results = self.results
            
        if not filtered_results:
            print("No matching configurations found")
            return None
            
        # Create comparison dataframe
        comparison = []
        
        for result in filtered_results:
            config_name = result["config_name"]
            metrics = result["aggregate_metrics"]
            
            row = {
                "Configuration": config_name,
                "Precision": metrics["avg_precision"],
                "Recall": metrics["avg_recall"],
                "MRR": metrics["avg_mrr"],
                "ROUGE-1": metrics["avg_rouge1"],
                "Faithfulness": metrics["avg_faithfulness"],
                "Retrieval Time (s)": metrics["avg_retrieval_time"],
                "Answer Time (s)": metrics["avg_answer_time"],
                "Questions": metrics["question_count"]
            }
            
            comparison.append(row)
            
        df = pd.DataFrame(comparison)
        
        return df
    
    def find_best_configuration(self, metric: str = "avg_mrr"):
        """
        Find the best configuration based on a specific metric
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Best configuration name and score
        """
        if not self.results:
            print("No evaluation results to analyze")
            return None, None
            
        best_score = -float('inf')
        best_config = None
        
        for result in self.results:
            config_name = result["config_name"]
            metrics = result["aggregate_metrics"]
            
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_config = config_name
                
        if best_config:
            print(f"Best configuration for {metric}: {best_config} with score {best_score:.4f}")
            
        return best_config, best_score


class EvaluationManager:
    """Utility class for managing evaluation datasets and results"""
    
    @staticmethod
    def create_example_dataset():
        """Create an example evaluation dataset"""
        from src.components.evaluation_dataset import create_example_dataset
        return create_example_dataset()
    
    @staticmethod
    def list_datasets():
        """List available evaluation datasets"""
        from src.components.evaluation_dataset import list_available_datasets
        datasets = list_available_datasets()
        
        if not datasets:
            print("No evaluation datasets found")
            return []
            
        print(f"Found {len(datasets)} evaluation datasets:")
        for i, dataset in enumerate(datasets):
            name = os.path.basename(dataset)
            print(f"{i+1}. {name}")
            
        return datasets
    
    @staticmethod
    def list_results():
        """List available evaluation results"""
        if not os.path.exists(DEFAULT_RESULTS_DIR):
            print("No evaluation results found")
            return []
            
        # Get results files
        results_files = [
            os.path.join(DEFAULT_RESULTS_DIR, f)
            for f in os.listdir(DEFAULT_RESULTS_DIR)
            if f.endswith('.json') and f.startswith('evaluation_')
        ]
        
        if not results_files:
            print("No evaluation results found")
            return []
            
        print(f"Found {len(results_files)} evaluation result files:")
        for i, result_file in enumerate(results_files):
            name = os.path.basename(result_file)
            print(f"{i+1}. {name}")
            
        return results_files
    
    @staticmethod
    def load_results(result_file: str) -> List[Dict[str, Any]]:
        """
        Load evaluation results from file
        
        Args:
            result_file: Path to results file
            
        Returns:
            List of result dictionaries
        """
        if not os.path.exists(result_file):
            print(f"Results file not found: {result_file}")
            return []
            
        with open(result_file, 'r') as f:
            return json.load(f)