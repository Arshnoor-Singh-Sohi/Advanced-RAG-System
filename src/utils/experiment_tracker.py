"""
Experiment Tracker Module

This module provides tools for tracking and analyzing RAG experiments:
- ExperimentTracker: For logging experiment configurations, results, and generating reports
"""

import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

class ExperimentTracker:
    """Tracks experimental results for RAG components"""
    
    def __init__(self, experiment_name: str):
        """
        Initialize a new experiment tracker
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.experiment_dir = f"results/{experiment_name}_{self.timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"Experiment tracker initialized: {experiment_name}")
        print(f"Results will be saved to: {self.experiment_dir}")
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """
        Log the experiment configuration
        
        Args:
            config: Dictionary with experiment configuration
        """
        # Save config to experiment directory
        with open(f"{self.experiment_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Experiment configuration logged with {len(config)} parameters")
    
    def log_iteration(self, iteration_data: Dict[str, Any]):
        """
        Log data from a single experimental iteration
        
        Args:
            iteration_data: Dictionary with iteration results
        """
        self.results.append(iteration_data)
        
        # Also save results after each iteration to prevent data loss
        self.save_results()
        
        # Print a summary of logged metrics
        metrics = {k: v for k, v in iteration_data.items() if k.startswith('metric_')}
        if metrics:
            print(f"Iteration logged with metrics: ", end="")
            metric_str = ", ".join([f"{k.replace('metric_', '')}={v:.4f}" for k, v in metrics.items()])
            print(metric_str)
        else:
            print(f"Iteration logged with {len(iteration_data)} parameters")
    
    def save_results(self):
        """Save all experiment results"""
        # Save as JSON
        with open(f"{self.experiment_dir}/results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easier analysis
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(f"{self.experiment_dir}/results.csv", index=False)
    
    def generate_report(self):
        """
        Generate visualization and summary report of experiment results
        
        Returns:
            Path to the report directory
        """
        if not self.results:
            print("No results to generate report from")
            return None
            
        df = pd.DataFrame(self.results)
        
        # Save report files to the experiment directory
        report_path = f"{self.experiment_dir}/report"
        os.makedirs(report_path, exist_ok=True)
        
        # Summary statistics
        summary = df.describe()
        summary.to_csv(f"{report_path}/summary_stats.csv")
        
        # Create visualizations based on experiment type
        if len(df) > 0:
            # Determine what to plot based on columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Plot metrics by configuration variants
            for metric in numeric_cols:
                if metric.startswith('metric_'):  # Convention for metrics
                    self._plot_metric(df, metric, report_path)
        
        print(f"Report generated in {report_path}")
        return report_path
    
    def _plot_metric(self, df, metric_name, save_path):
        """
        Plot a specific metric against different configurations
        
        Args:
            df: DataFrame with results
            metric_name: Name of the metric column to plot
            save_path: Directory to save the plot
        """
        # Identify potential grouping variables (non-metric columns)
        potential_groups = [col for col in df.columns 
                            if not col.startswith('metric_')
                            and df[col].nunique() > 1
                            and df[col].nunique() < 10]
        
        for group_var in potential_groups:
            plt.figure(figsize=(10, 6))
            
            # Bar plots for categorical variables
            if df[group_var].dtype == 'object' or df[group_var].nunique() < 10:
                sns.barplot(x=group_var, y=metric_name, data=df)
                plt.title(f"{metric_name} by {group_var}")
                plt.xticks(rotation=45)
            # Line plots for numerical variables
            else:
                sns.lineplot(x=group_var, y=metric_name, data=df)
                plt.title(f"{metric_name} vs {group_var}")
                
            plt.tight_layout()
            plt.savefig(f"{save_path}/{metric_name}_by_{group_var}.png")
            plt.close()
        
        # Create a heatmap if there are two categorical variables to group by
        if len(potential_groups) >= 2:
            # Find two grouping variables with manageable number of categories
            group_vars = [v for v in potential_groups if df[v].nunique() <= 6]
            
            if len(group_vars) >= 2:
                group1, group2 = group_vars[0], group_vars[1]
                
                # Create a pivot table
                try:
                    pivot_df = df.pivot_table(index=group1, columns=group2, values=metric_name)
                    
                    # Plot heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
                    plt.title(f"{metric_name} by {group1} and {group2}")
                    plt.tight_layout()
                    plt.savefig(f"{save_path}/{metric_name}_heatmap.png")
                    plt.close()
                except:
                    # Skip if pivot fails
                    pass
    
    def find_best_configuration(self, metric_name: str = "metric_mrr") -> Dict[str, Any]:
        """
        Find the best configuration based on a specific metric
        
        Args:
            metric_name: Name of the metric to optimize (default: "metric_mrr")
            
        Returns:
            Dictionary with the best configuration
        """
        if not self.results or metric_name not in self.results[0]:
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Find row with highest metric value
        best_idx = df[metric_name].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        # Remove metrics from the configuration
        config = {k: v for k, v in best_config.items() if not k.startswith('metric_')}
        
        return config
    
    def add_comparative_analysis(self, other_experiment_paths: List[str]):
        """
        Add comparative analysis with other experiments
        
        Args:
            other_experiment_paths: List of paths to other experiment results
        """
        # Create a directory for comparative analysis
        comparison_path = f"{self.experiment_dir}/comparisons"
        os.makedirs(comparison_path, exist_ok=True)
        
        # Load current experiment results
        current_df = pd.DataFrame(self.results)
        
        # Add experiment name to identify in the combined dataset
        current_df['experiment'] = self.experiment_name
        
        # Dictionary to store DataFrames from each experiment
        all_dfs = {self.experiment_name: current_df}
        
        # Load other experiment results
        for path in other_experiment_paths:
            if not os.path.exists(f"{path}/results.csv"):
                print(f"Warning: Results file not found in {path}")
                continue
                
            # Load results
            exp_df = pd.read_csv(f"{path}/results.csv")
            
            # Extract experiment name from path
            exp_name = os.path.basename(path).split("_")[0]
            
            # Add experiment name
            exp_df['experiment'] = exp_name
            
            # Add to dictionary
            all_dfs[exp_name] = exp_df
        
        # Combine all DataFrames
        if len(all_dfs) > 1:
            combined_df = pd.concat(all_dfs.values(), ignore_index=True)
            
            # Save combined results
            combined_df.to_csv(f"{comparison_path}/combined_results.csv", index=False)
            
            # Generate comparative visualizations
            self._generate_comparative_plots(combined_df, comparison_path)
            
            print(f"Comparative analysis saved to {comparison_path}")
    
    def _generate_comparative_plots(self, df, save_path):
        """
        Generate comparative plots for experiments
        
        Args:
            df: Combined DataFrame with experiment results
            save_path: Directory to save plots
        """
        # Plot metrics across experiments
        metric_cols = [col for col in df.columns if col.startswith('metric_')]
        
        for metric in metric_cols:
            # Clean metric name for display
            display_metric = metric.replace('metric_', '')
            
            # Bar plot of average metric by experiment
            plt.figure(figsize=(10, 6))
            sns.barplot(x='experiment', y=metric, data=df)
            plt.title(f"Average {display_metric} by Experiment")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_path}/{display_metric}_by_experiment.png")
            plt.close()
            
            # Box plot to show distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='experiment', y=metric, data=df)
            plt.title(f"Distribution of {display_metric} by Experiment")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_path}/{display_metric}_boxplot.png")
            plt.close()