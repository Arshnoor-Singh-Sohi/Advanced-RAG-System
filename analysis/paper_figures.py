"""
Paper Figures Generator

This script generates high-quality figures and tables for the research paper
based on experiment results. It creates visualization in IEEE conference format.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np
import json
import matplotlib as mpl

def generate_paper_figures(results_dir: str = "results", output_dir: str = "paper_figures"):
    """
    Generate high-quality figures and tables for the IEEE paper
    
    Args:
        results_dir: Directory with experiment results
        output_dir: Directory to save output figures and analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result CSV files
    result_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith("results.csv"):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("No result files found")
        return
    
    # Read all results
    all_results = []
    for file in result_files:
        # Determine experiment type from folder name
        experiment_name = os.path.basename(os.path.dirname(file)).split("_")[0]
        
        # Read data
        df = pd.read_csv(file)
        df["experiment"] = experiment_name
        all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Generate figures for each experiment type
    if "chunking" in combined_df["experiment"].values:
        generate_chunking_figures(combined_df, output_dir)
    
    if "embedding" in combined_df["experiment"].values:
        generate_embedding_figures(combined_df, output_dir)
    
    if "retrieval" in combined_df["experiment"].values:
        generate_retrieval_figures(combined_df, output_dir)
    
    if "query_processing" in combined_df["experiment"].values:
        generate_query_processing_figures(combined_df, output_dir)
    
    if "reranking" in combined_df["experiment"].values:
        generate_reranking_figures(combined_df, output_dir)
    
    if "generation" in combined_df["experiment"].values:
        generate_generation_figures(combined_df, output_dir)
    
    # Generate overall performance comparison figure
    generate_overall_comparison(combined_df, output_dir)
    
    print(f"Paper figures generated in {output_dir}")

def set_ieee_style():
    """Set matplotlib style for IEEE paper figures"""
    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'figure.figsize': (6.5, 4.5),  # IEEE column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

def generate_chunking_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for chunking experiments"""
    set_ieee_style()
    
    # Filter to chunking experiment results
    chunking_df = df[df["experiment"] == "chunking"].copy()
    
    # Create directory for chunking figures
    chunking_dir = os.path.join(output_dir, "chunking")
    os.makedirs(chunking_dir, exist_ok=True)
    
    # Figure 1: Chunking Strategy Comparison
    plt.figure()
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Create a new aggregated dataframe for plotting
    plot_data = pd.melt(
        chunking_df, 
        id_vars=["chunking_strategy", "retrieval_method"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    # Clean up metric names
    plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
    
    # Create facet grid
    g = sns.catplot(
        data=plot_data,
        x="chunking_strategy",
        y="Score",
        hue="retrieval_method",
        col="Metric",
        kind="bar",
        height=3,
        aspect=1.2,
        sharey=False
    )
    
    g.set_axis_labels("Chunking Strategy", "Score")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)
    g.tight_layout()
    
    plt.savefig(os.path.join(chunking_dir, "fig1_chunking_strategy_comparison.png"))
    plt.close()
    
    # Figure 2: Chunk Size Impact (for fixed strategy)
    fixed_df = chunking_df[chunking_df["chunking_strategy"] == "fixed"].copy()
    if not fixed_df.empty:
        plt.figure()
        g = sns.lineplot(
            data=fixed_df,
            x="chunk_size",
            y="metric_mrr",
            hue="retrieval_method",
            style="chunk_overlap",
            markers=True
        )
        
        plt.title("Impact of Chunk Size on MRR")
        plt.xlabel("Chunk Size (tokens)")
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(chunking_dir, "fig2_chunk_size_impact.png"))
        plt.close()
    
    # Generate summary table for paper (LaTeX format)
    summary = chunking_df.groupby(["chunking_strategy", "retrieval_method"])[
        ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    ].mean().reset_index()
    
    # Format the table for IEEE paper
    summary.columns = ["Chunking Strategy", "Retrieval Method", "Precision@3", "Recall@3", "MRR"]
    
    # Save as CSV (which can be imported into LaTeX)
    summary.to_csv(os.path.join(chunking_dir, "table1_chunking_performance.csv"), index=False,
               float_format="%.4f")
    
    # Also save as LaTeX table
    with open(os.path.join(chunking_dir, "table1_chunking_performance.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison of Different Chunking Strategies}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{|l|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Chunking Strategy} & \\textbf{Retrieval Method} & \\textbf{Precision@3} & \\textbf{Recall@3} & \\textbf{MRR} \\\\ \n")
        f.write("\\hline\n")
        
        for _, row in summary.iterrows():
            f.write(f"{row['Chunking Strategy']} & {row['Retrieval Method']} & {row['Precision@3']:.4f} & {row['Recall@3']:.4f} & {row['MRR']:.4f} \\\\ \n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:chunking}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

def generate_embedding_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for embedding experiments"""
    set_ieee_style()
    
    # Filter to embedding experiment results
    embedding_df = df[df["experiment"] == "embedding"].copy()
    
    # Create directory for embedding figures
    embedding_dir = os.path.join(output_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Figure 3: Embedding Model Performance Comparison
    plt.figure()
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    plot_data = pd.melt(
        embedding_df, 
        id_vars=["embedding_model", "embedding_dim"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
    
    g = sns.catplot(
        data=plot_data,
        x="embedding_model",
        y="Score",
        hue="Metric",
        kind="bar",
        height=4,
        aspect=1.5
    )
    
    g.set_axis_labels("Embedding Model", "Score")
    g.set_xticklabels(rotation=45)
    plt.title("Embedding Model Performance Comparison")
    plt.tight_layout()
    
    plt.savefig(os.path.join(embedding_dir, "fig3_embedding_model_performance.png"))
    plt.close()
    
    # Figure 4: Embedding Efficiency (Time vs Performance)
    plt.figure()
    
    # Create scatter plot
    if "embedding_time_sec" in embedding_df.columns and "metric_mrr" in embedding_df.columns:
        plt.scatter(
            embedding_df["embedding_time_sec"] / len(embedding_df["embedding_time_sec"].unique()),
            embedding_df["metric_mrr"],
            s=100,
            alpha=0.7
        )
        
        # Add text labels for each point
        for _, row in embedding_df.iterrows():
            plt.annotate(
                row["embedding_model"],
                (row["embedding_time_sec"] / len(embedding_df["embedding_time_sec"].unique()),
                 row["metric_mrr"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title("Embedding Model Efficiency: Time vs. Performance")
        plt.xlabel("Embedding Time per Document (seconds)")
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(embedding_dir, "fig4_embedding_efficiency.png"))
        plt.close()
    
    # Generate summary table for paper
    summary = embedding_df.groupby(["embedding_model", "embedding_dim"])[
        ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr",
         "embedding_time_sec", "docs_per_second"]
    ].mean().reset_index()
    
    summary.columns = ["Model", "Dimension", "Precision@3", "Recall@3", "MRR",
                     "Embedding Time (s)", "Docs/Second"]
    
    # Save as CSV
    summary.to_csv(os.path.join(embedding_dir, "table2_embedding_performance.csv"),
               index=False, float_format="%.4f")
    
    # Also save as LaTeX table
    with open(os.path.join(embedding_dir, "table2_embedding_performance.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison of Different Embedding Models}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Dim} & \\textbf{Precision@3} & \\textbf{Recall@3} & \\textbf{MRR} & \\textbf{Docs/Sec} \\\\ \n")
        f.write("\\hline\n")
        
        for _, row in summary.iterrows():
            f.write(f"{row['Model']} & {int(row['Dimension'])} & {row['Precision@3']:.4f} & {row['Recall@3']:.4f} & {row['MRR']:.4f} & {row['Docs/Second']:.2f} \\\\ \n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:embedding}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

def generate_retrieval_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for retrieval method experiments"""
    set_ieee_style()
    
    # Filter to retrieval experiment results
    retrieval_df = df[df["experiment"] == "retrieval"].copy()
    
    # Create directory for retrieval figures
    retrieval_dir = os.path.join(output_dir, "retrieval")
    os.makedirs(retrieval_dir, exist_ok=True)
    
    # Figure 5: Retrieval Method Comparison
    plt.figure()
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    plot_data = pd.melt(
        retrieval_df, 
        id_vars=["retrieval_method", "retrieval_type"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
    
    g = sns.catplot(
        data=plot_data,
        x="retrieval_method",
        y="Score",
        hue="Metric",
        kind="bar",
        height=4,
        aspect=1.5
    )
    
    g.set_axis_labels("Retrieval Method", "Score")
    g.set_xticklabels(rotation=45)
    plt.title("Retrieval Method Performance Comparison")
    plt.tight_layout()
    
    plt.savefig(os.path.join(retrieval_dir, "fig5_retrieval_method_comparison.png"))
    plt.close()
    
    # Figure 6: Impact of Query Expansion
    if "query_expansion" in retrieval_df.columns:
        # Filter to include only rows with query expansion data
        expansion_df = retrieval_df[retrieval_df["query_expansion"].notna()].copy()
        
        if not expansion_df.empty:
            plt.figure()
            
            # Convert to categorical type to help with plotting
            expansion_df["query_expansion"] = expansion_df["query_expansion"].astype(str)
            
            g = sns.catplot(
                data=expansion_df,
                x="retrieval_method",
                y="metric_mrr",
                hue="query_expansion",
                kind="bar",
                height=4,
                aspect=1.5
            )
            
            g.set_axis_labels("Retrieval Method", "Mean Reciprocal Rank (MRR)")
            g.set_xticklabels(rotation=45)
            plt.title("Impact of Query Expansion on Retrieval Performance")
            plt.tight_layout()
            
            plt.savefig(os.path.join(retrieval_dir, "fig6_query_expansion_impact.png"))
            plt.close()
    
    # Figure 7: Hybrid Search Alpha Parameter Impact
    # Filter to hybrid methods with alpha parameter
    hybrid_df = retrieval_df[
        (retrieval_df["retrieval_type"] == "hybrid") & 
        ("alpha" in retrieval_df.columns)
    ].copy()
    
    if not hybrid_df.empty:
        plt.figure()
        
        # Group by alpha and get mean MRR
        alpha_groups = hybrid_df.groupby("alpha")["metric_mrr"].mean().reset_index()
        
        plt.plot(alpha_groups["alpha"], alpha_groups["metric_mrr"], 
               marker='o', linewidth=2, markersize=8)
        
        plt.title("Impact of Alpha Parameter on Hybrid Retrieval")
        plt.xlabel("Alpha (Vector Weight)")
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(alpha_groups["alpha"])
        plt.tight_layout()
        
        plt.savefig(os.path.join(retrieval_dir, "fig7_hybrid_alpha_impact.png"))
        plt.close()
    
    # Generate summary table for paper
    summary = retrieval_df.groupby(["retrieval_method", "retrieval_type", "query_expansion"])[
        ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    ].mean().reset_index()
    
    summary.columns = ["Method", "Type", "Query Expansion", "Precision@3", "Recall@3", "MRR"]
    
    # Save as CSV
    summary.to_csv(os.path.join(retrieval_dir, "table3_retrieval_performance.csv"),
               index=False, float_format="%.4f")
    
    # Also save as LaTeX table
    with open(os.path.join(retrieval_dir, "table3_retrieval_performance.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison of Different Retrieval Methods}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{|l|l|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Type} & \\textbf{Query Expansion} & \\textbf{Precision@3} & \\textbf{Recall@3} & \\textbf{MRR} \\\\ \n")
        f.write("\\hline\n")
        
        for _, row in summary.iterrows():
            f.write(f"{row['Method']} & {row['Type']} & {row['Query Expansion']} & {row['Precision@3']:.4f} & {row['Recall@3']:.4f} & {row['MRR']:.4f} \\\\ \n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:retrieval}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

def generate_query_processing_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for query processing experiments"""
    set_ieee_style()
    
    # Filter to query processing experiment results
    query_df = df[df["experiment"] == "query_processing"].copy()
    
    # Create directory for query processing figures
    query_dir = os.path.join(output_dir, "query_processing")
    os.makedirs(query_dir, exist_ok=True)
    
    # Figure 8: Query Processing Technique Comparison
    plt.figure(figsize=(7, 5))
    
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    plot_data = pd.melt(
        query_df, 
        id_vars=["technique", "description"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
    
    g = sns.catplot(
        data=plot_data,
        x="technique",
        y="Score",
        hue="Metric",
        kind="bar",
        height=4,
        aspect=1.5
    )
    
    g.set_axis_labels("Query Processing Technique", "Score")
    g.set_xticklabels(rotation=45)
    plt.title("Query Processing Technique Performance Comparison")
    plt.tight_layout()
    
    plt.savefig(os.path.join(query_dir, "fig8_query_technique_comparison.png"))
    plt.close()
    
    # Figure 9: Efficiency vs. Performance
    if "processing_time_sec" in query_df.columns and "metric_mrr" in query_df.columns:
        plt.figure()
        
        # Calculate queries per second
        query_df["queries_per_second"] = 1.0 / query_df["processing_time_sec"]
        
        # Create scatter plot
        scatter = plt.scatter(
            query_df["processing_time_sec"],
            query_df["metric_mrr"],
            s=100,
            alpha=0.7
        )
        
        # Add text labels for each point
        for _, row in query_df.iterrows():
            plt.annotate(
                row["technique"],
                (row["processing_time_sec"], row["metric_mrr"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title("Query Processing Efficiency vs. Performance")
        plt.xlabel("Processing Time (seconds)")
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(query_dir, "fig9_query_efficiency_vs_performance.png"))
        plt.close()
    
    # Generate summary table for paper
    summary = query_df.groupby(["technique", "description"])[
        ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    ].mean().reset_index()
    
    # Format the table for paper
    summary.columns = ["Technique", "Description", "Precision@3", "Recall@3", "MRR"]
    
    # Save as CSV
    summary.to_csv(os.path.join(query_dir, "table4_query_processing_performance.csv"),
               index=False, float_format="%.4f")
    
    # Also save as LaTeX table
    with open(os.path.join(query_dir, "table4_query_processing_performance.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison of Different Query Processing Techniques}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{|l|p{5cm}|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Technique} & \\textbf{Description} & \\textbf{Precision@3} & \\textbf{Recall@3} & \\textbf{MRR} \\\\ \n")
        f.write("\\hline\n")
        
        for _, row in summary.iterrows():
            f.write(f"{row['Technique']} & {row['Description']} & {row['Precision@3']:.4f} & {row['Recall@3']:.4f} & {row['MRR']:.4f} \\\\ \n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:query_processing}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

def generate_reranking_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for reranking experiments"""
    set_ieee_style()
    
    # Filter to reranking experiment results
    reranking_df = df[df["experiment"] == "reranking"].copy()
    
    # Create directory for reranking figures
    reranking_dir = os.path.join(output_dir, "reranking")
    os.makedirs(reranking_dir, exist_ok=True)
    
    # Figure 10: Reranking Method Comparison
    plt.figure()
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    plot_data = pd.melt(
        reranking_df, 
        id_vars=["reranking_method", "description"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
    
    g = sns.catplot(
        data=plot_data,
        x="reranking_method",
        y="Score",
        hue="Metric",
        kind="bar",
        height=4,
        aspect=1.5
    )
    
    g.set_axis_labels("Reranking Method", "Score")
    g.set_xticklabels(rotation=45)
    plt.title("Reranking Method Performance Comparison")
    plt.tight_layout()
    
    plt.savefig(os.path.join(reranking_dir, "fig10_reranking_method_comparison.png"))
    plt.close()
    
    # Figure 11: Reranking Improvement over Baseline
    # Get baseline performance (no_reranking)
    if "no_reranking" in reranking_df["reranking_method"].values and "metric_mrr" in reranking_df.columns:
        baseline_mrr = reranking_df[reranking_df["reranking_method"] == "no_reranking"]["metric_mrr"].mean()
        
        # Calculate improvement over baseline
        improvement_df = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
        improvement_df["improvement"] = improvement_df["metric_mrr"] - baseline_mrr
        improvement_df["relative_improvement"] = improvement_df["improvement"] / baseline_mrr * 100
        
        plt.figure()
        
        # Create bar plot of relative improvement
        sns.barplot(
            data=improvement_df,
            x="reranking_method",
            y="relative_improvement"
        )
        
        plt.title("Reranking Performance Improvement over Baseline")
        plt.xlabel("Reranking Method")
        plt.ylabel("Relative Improvement (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(reranking_dir, "fig11_reranking_improvement.png"))
        plt.close()
    
    # Figure 12: Reranking Time vs. Performance Improvement
    if ("reranking_time_sec" in reranking_df.columns and 
        "no_reranking" in reranking_df["reranking_method"].values and 
        "metric_mrr" in reranking_df.columns):
        
        baseline_mrr = reranking_df[reranking_df["reranking_method"] == "no_reranking"]["metric_mrr"].mean()
        
        # Calculate improvement and efficiency
        improvement_df = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
        improvement_df["improvement"] = improvement_df["metric_mrr"] - baseline_mrr
        improvement_df["relative_improvement"] = improvement_df["improvement"] / baseline_mrr * 100
        
        plt.figure()
        
        # Create scatter plot
        scatter = plt.scatter(
            improvement_df["reranking_time_sec"],
            improvement_df["relative_improvement"],
            s=100,
            alpha=0.7
        )
        
        # Add text labels for each point
        for _, row in improvement_df.iterrows():
            plt.annotate(
                row["reranking_method"],
                (row["reranking_time_sec"], row["relative_improvement"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title("Reranking Efficiency vs. Performance Improvement")
        plt.xlabel("Reranking Time (seconds)")
        plt.ylabel("Relative Improvement (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(reranking_dir, "fig12_reranking_efficiency_vs_improvement.png"))
        plt.close()
    
    # Generate summary table for paper
    summary = reranking_df.groupby(["reranking_method", "description"])[
        ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    ].mean().reset_index()
    
    # Format the table for paper
    summary.columns = ["Method", "Description", "Precision@3", "Recall@3", "MRR"]
    
    # Save as CSV
    summary.to_csv(os.path.join(reranking_dir, "table5_reranking_performance.csv"),
               index=False, float_format="%.4f")
    
    # Also save as LaTeX table
    with open(os.path.join(reranking_dir, "table5_reranking_performance.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison of Different Reranking Methods}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{tabular}{|l|p{5cm}|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Description} & \\textbf{Precision@3} & \\textbf{Recall@3} & \\textbf{MRR} \\\\ \n")
        f.write("\\hline\n")
        
        for _, row in summary.iterrows():
            f.write(f"{row['Method']} & {row['Description']} & {row['Precision@3']:.4f} & {row['Recall@3']:.4f} & {row['MRR']:.4f} \\\\ \n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:reranking}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

def generate_generation_figures(df: pd.DataFrame, output_dir: str):
    """Generate figures for generation experiments"""
    set_ieee_style()
    
    # Filter to generation experiment results
    generation_df = df[df["experiment"] == "generation"].copy()
    
    # Create directory for generation figures
    generation_dir = os.path.join(output_dir, "generation")
    os.makedirs(generation_dir, exist_ok=True)
    
    # Figure 13: Prompt Template Comparison
    if "prompt_template" in generation_df.columns:
        plt.figure()
        metrics = ["metric_answer_precision", "metric_faithfulness", "metric_citation_rate", "metric_answer_correctness"]
        
        # Check which metrics are available
        available_metrics = [m for m in metrics if m in generation_df.columns]
        
        if available_metrics:
            plot_data = pd.melt(
                generation_df, 
                id_vars=["prompt_template", "prompt_description"],
                value_vars=available_metrics,
                var_name="Metric",
                value_name="Score"
            )
            
            plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
            
            g = sns.catplot(
                data=plot_data,
                x="prompt_template",
                y="Score",
                hue="Metric",
                kind="bar",
                height=4,
                aspect=1.5
            )
            
            g.set_axis_labels("Prompt Template", "Score")
            g.set_xticklabels(rotation=45)
            plt.title("Prompt Template Performance Comparison")
            plt.tight_layout()
            
            plt.savefig(os.path.join(generation_dir, "fig13_prompt_template_comparison.png"))
            plt.close()
    
    # Figure 14: Retrieval Configuration Impact on Generation
    if "retrieval_config" in generation_df.columns and "metric_answer_correctness" in generation_df.columns:
        plt.figure()
        
        # Group by retrieval config and get mean correctness
        retrieval_impact = generation_df.groupby("retrieval_config")["metric_answer_correctness"].mean().reset_index()
        
        g = sns.barplot(
            data=retrieval_impact,
            x="retrieval_config",
            y="metric_answer_correctness"
        )
        
        plt.title("Impact of Retrieval Configuration on Answer Correctness")
        plt.xlabel("Retrieval Configuration")
        plt.ylabel("Answer Correctness")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(generation_dir, "fig14_retrieval_impact_on_generation.png"))
        plt.close()
    
    # Figure 15: Heatmap of Prompt vs Retrieval for Answer Precision
    if ("prompt_template" in generation_df.columns and
        "retrieval_config" in generation_df.columns and
        "metric_answer_precision" in generation_df.columns):
        
        # Create pivot table
        pivot_df = generation_df.pivot_table(
            index="prompt_template",
            columns="retrieval_config",
            values="metric_answer_precision",
            aggfunc="mean"
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title("Answer Precision by Prompt Template and Retrieval Configuration")
        plt.tight_layout()
        
        plt.savefig(os.path.join(generation_dir, "fig15_prompt_retrieval_heatmap.png"))
        plt.close()
    
    # Generate summary table for paper
    group_cols = []
    if "prompt_template" in generation_df.columns:
        group_cols.append("prompt_template")
    if "prompt_description" in generation_df.columns:
        group_cols.append("prompt_description")
    if "retrieval_config" in generation_df.columns:
        group_cols.append("retrieval_config")
    
    if group_cols:
        metrics_cols = [col for col in generation_df.columns if col.startswith("metric_")]
        
        summary = generation_df.groupby(group_cols)[metrics_cols].mean().reset_index()
        
        # Format column names for paper
        new_cols = []
        for col in summary.columns:
            if col.startswith("metric_"):
                new_cols.append(col.replace("metric_", "").replace("_", " ").title())
            else:
                new_cols.append(col.replace("_", " ").title())
        
        summary.columns = new_cols
        
        # Save as CSV
        summary.to_csv(os.path.join(generation_dir, "table6_generation_performance.csv"),
                   index=False, float_format="%.4f")
        
        # Also save as LaTeX table (simplified for flexibility)
        with open(os.path.join(generation_dir, "table6_generation_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Generation Configurations}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|" + "l|" * len(group_cols) + "c|" * (len(summary.columns) - len(group_cols)) + "}\n")
            f.write("\\hline\n")
            
            # Header row
            header = ""
            for col in summary.columns:
                header += f"\\textbf{{{col}}} & "
            header = header[:-3] + " \\\\ \n"
            f.write(header)
            
            f.write("\\hline\n")
            
            # Data rows
            for _, row in summary.iterrows():
                row_str = ""
                for i, col in enumerate(summary.columns):
                    if i < len(group_cols):
                        row_str += f"{row[col]} & "
                    else:
                        row_str += f"{row[col]:.4f} & "
                row_str = row_str[:-3] + " \\\\ \n"
                f.write(row_str)
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:generation}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

def generate_overall_comparison(df: pd.DataFrame, output_dir: str):
    """Generate overall comparison figures across all experiments"""
    set_ieee_style()
    
    # Create directory for overall comparison figures
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Find metrics common across experiments
    metric_columns = [col for col in df.columns if col.startswith("metric_")]
    experiments = df["experiment"].unique()
    
    # Find metrics present in all experiments
    common_metrics = []
    for metric in metric_columns:
        if all(metric in df[df["experiment"] == exp].columns for exp in experiments):
            common_metrics.append(metric)
    
    # Figure 16: Best performance by component
    if common_metrics:
        # Use MRR if available, otherwise first common metric
        if "metric_mrr" in common_metrics:
            comparison_metric = "metric_mrr"
        else:
            comparison_metric = common_metrics[0]
        
        # Get best score for each experiment
        best_scores = []
        for experiment in experiments:
            exp_df = df[df["experiment"] == experiment]
            if comparison_metric in exp_df.columns:
                max_score = exp_df[comparison_metric].max()
                best_scores.append({"Component": experiment.capitalize(), "Best Score": max_score})
        
        if best_scores:
            plt.figure(figsize=(8, 5))
            scores_df = pd.DataFrame(best_scores)
            
            g = sns.barplot(
                data=scores_df,
                x="Component",
                y="Best Score"
            )
            
            plt.title(f"Best {comparison_metric.replace('metric_', '').upper()} by RAG Component")
            plt.xlabel("RAG Component")
            plt.ylabel(comparison_metric.replace("metric_", "").upper())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(comparison_dir, "fig16_best_performance_by_component.png"))
            plt.close()
    
    # Figure 17: Performance impact of each component
    # Calculate how much each component improves over baseline
    impact_data = []
    
    # Chunking impact
    chunking_df = df[df["experiment"] == "chunking"]
    if not chunking_df.empty and "metric_mrr" in chunking_df.columns:
        # Get baseline (fixed chunking with default settings)
        baseline_df = chunking_df[(chunking_df["chunking_strategy"] == "fixed") & 
                                 (chunking_df["chunk_size"] == 128) & 
                                 (chunking_df["retrieval_method"] == "vector")]
        
        if not baseline_df.empty:
            baseline_score = baseline_df["metric_mrr"].mean()
            
            # Get best score
            best_score = chunking_df["metric_mrr"].max()
            
            # Calculate improvement
            abs_improvement = best_score - baseline_score
            rel_improvement = abs_improvement / baseline_score * 100
            
            impact_data.append({
                "Component": "Chunking",
                "Relative Improvement (%)": rel_improvement
            })
    
    # Embedding impact
    embedding_df = df[df["experiment"] == "embedding"]
    if not embedding_df.empty and "metric_mrr" in embedding_df.columns:
        # Get baseline (default embedding model)
        baseline_df = embedding_df[embedding_df["embedding_model"] == "all-MiniLM-L6-v2"]
        
        if not baseline_df.empty:
            baseline_score = baseline_df["metric_mrr"].mean()
            
            # Get best score
            best_score = embedding_df["metric_mrr"].max()
            
            # Calculate improvement
            abs_improvement = best_score - baseline_score
            rel_improvement = abs_improvement / baseline_score * 100
            
            impact_data.append({
                "Component": "Embedding",
                "Relative Improvement (%)": rel_improvement
            })
    
    # Retrieval impact
    retrieval_df = df[df["experiment"] == "retrieval"]
    if not retrieval_df.empty and "metric_mrr" in retrieval_df.columns:
        # Get baseline (vector search without expansion)
        baseline_df = retrieval_df[(retrieval_df["retrieval_type"] == "vector") & 
                                  (retrieval_df["query_expansion"] == "none")]
        
        if not baseline_df.empty:
            baseline_score = baseline_df["metric_mrr"].mean()
            
            # Get best score
            best_score = retrieval_df["metric_mrr"].max()
            
            # Calculate improvement
            abs_improvement = best_score - baseline_score
            rel_improvement = abs_improvement / baseline_score * 100
            
            impact_data.append({
                "Component": "Retrieval",
                "Relative Improvement (%)": rel_improvement
            })
    
    # Query processing impact
    query_df = df[df["experiment"] == "query_processing"]
    if not query_df.empty and "metric_mrr" in query_df.columns:
        # Get baseline (original queries)
        baseline_df = query_df[query_df["technique"] == "original"]
        
        if not baseline_df.empty:
            baseline_score = baseline_df["metric_mrr"].mean()
            
            # Get best score
            best_score = query_df["metric_mrr"].max()
            
            # Calculate improvement
            abs_improvement = best_score - baseline_score
            rel_improvement = abs_improvement / baseline_score * 100
            
            impact_data.append({
                "Component": "Query Processing",
                "Relative Improvement (%)": rel_improvement
            })
    
    # Reranking impact
    reranking_df = df[df["experiment"] == "reranking"]
    if not reranking_df.empty and "metric_mrr" in reranking_df.columns:
        # Get baseline (no reranking)
        baseline_df = reranking_df[reranking_df["reranking_method"] == "no_reranking"]
        
        if not baseline_df.empty:
            baseline_score = baseline_df["metric_mrr"].mean()
            
            # Get best score
            best_score = reranking_df["metric_mrr"].max()
            
            # Calculate improvement
            abs_improvement = best_score - baseline_score
            rel_improvement = abs_improvement / baseline_score * 100
            
            impact_data.append({
                "Component": "Reranking",
                "Relative Improvement (%)": rel_improvement
            })
    
    # Plot impact comparison
    if impact_data:
        plt.figure(figsize=(8, 5))
        impact_df = pd.DataFrame(impact_data)
        
        g = sns.barplot(
            data=impact_df,
            x="Component",
            y="Relative Improvement (%)"
        )
        
        plt.title("Relative Performance Improvement by RAG Component")
        plt.xlabel("RAG Component")
        plt.ylabel("Relative Improvement (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(comparison_dir, "fig17_component_impact_comparison.png"))
        plt.close()
    
    # Figure 18: End-to-end RAG Pipeline Optimization
    # Create a stacked bar chart showing cumulative improvements
    if impact_data:
        plt.figure(figsize=(10, 6))
        
        # Sort components by impact
        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values("Relative Improvement (%)")
        
        # Create cumulative improvements
        baseline = 100  # Start at 100%
        cumulative = [baseline]
        components = ["Baseline"]
        
        for _, row in impact_df.iterrows():
            component = row["Component"]
            improvement = row["Relative Improvement (%)"]
            
            # Add to cumulative
            new_value = cumulative[-1] * (1 + improvement/100)
            cumulative.append(new_value)
            components.append(component)
        
        # Plot
        plt.plot(components, cumulative, marker='o', linewidth=2, markersize=10)
        
        for i, (x, y) in enumerate(zip(components, cumulative)):
            if i > 0:
                plt.annotate(f"+{impact_df.iloc[i-1]['Relative Improvement (%)']:.1f}%",
                           (x, y),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
        
        plt.title("Cumulative Performance Improvement in RAG Pipeline")
        plt.ylabel("Relative Performance (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(comparison_dir, "fig18_cumulative_pipeline_improvement.png"))
        plt.close()
    
    # Create summary table for paper
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values("Relative Improvement (%)", ascending=False)
        
        # Save as CSV
        impact_df.to_csv(os.path.join(comparison_dir, "table7_component_impact.csv"),
                      index=False, float_format="%.2f")
        
        # Also save as LaTeX table
        with open(os.path.join(comparison_dir, "table7_component_impact.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Impact of RAG Components}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|l|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Component} & \\textbf{Relative Improvement (\\%)} \\\\ \n")
            f.write("\\hline\n")
            
            for _, row in impact_df.iterrows():
                f.write(f"{row['Component']} & {row['Relative Improvement (%)']:.2f}\\% \\\\ \n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:component_impact}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                      help="Directory with experiment results")
    parser.add_argument("--output-dir", type=str, default="paper_figures",
                      help="Directory to save output figures")
    
    args = parser.parse_args()
    
    generate_paper_figures(results_dir=args.results_dir, output_dir=args.output_dir)