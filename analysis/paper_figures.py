"""
Enhanced Paper Figures Generator

This script generates high-quality figures and tables for the research paper
based on experiment results. It creates visualization in IEEE conference format.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

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
        try:
            # Determine experiment type from folder name
            experiment_name = os.path.basename(os.path.dirname(file)).split("_")[0]
            
            # Read data
            df = pd.read_csv(file)
            df["experiment"] = experiment_name
            all_results.append(df)
            print(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_results:
        print("No valid result data found")
        return
        
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} rows")
    
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
    """Set enhanced matplotlib style for IEEE paper figures with better readability"""
    # Use a modern style as base
    plt.style.use('seaborn-v0_8-whitegrid') 
    
    # Define a custom color palette - colorblind friendly
    sns.set_palette("colorblind")
    
    # Update parameters for better readability and visual appeal
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,         # Increased base font size
        'axes.labelsize': 11,    # Increased label font size
        'axes.titlesize': 13,    # Increased title font size
        'xtick.labelsize': 9,    # Increased tick font size
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,  # Increased figure title size
        'figure.figsize': (7.0, 5.0),  # Slightly larger than IEEE column width for better readability
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,  # Increased padding
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,   # Remove top spine for cleaner look
        'axes.spines.right': False, # Remove right spine
        'axes.axisbelow': True,     # Grid lines below the plot elements
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': True,     # Add frame to legend
        'legend.framealpha': 0.8,   # Slightly transparent legend
        'legend.edgecolor': 'gray',
        'legend.borderpad': 0.5,
        'image.cmap': 'viridis'     # Modern colormap
    })

def get_color_palette(n_colors: int) -> List[str]:
    """Generate a colorblind-friendly palette with the required number of colors"""
    if n_colors <= 10:
        # Use colorblind palette for small number of categories
        return sns.color_palette("colorblind", n_colors)
    else:
        # For larger sets, use a perceptually uniform colormap
        return sns.color_palette("viridis", n_colors)

def set_figure_size(complexity: str = "medium") -> Tuple[float, float]:
    """Return appropriate figure size based on plot complexity"""
    if complexity == "simple":
        return (6.5, 4.5)  # Standard IEEE column width
    elif complexity == "medium":
        return (7.0, 5.0)  # Slightly larger
    elif complexity == "complex":
        return (8.0, 6.0)  # Much larger for complex plots
    elif complexity == "wide":
        return (9.0, 4.5)  # Wide format for bar charts with many categories
    elif complexity == "tall":
        return (6.5, 7.0)  # Tall format for plots with many y-axis categories
    else:
        return (7.0, 5.0)  # Default to medium

def check_empty_data(df: pd.DataFrame, metric_cols: List[str], group_cols: List[str] = None) -> bool:
    """Check if dataframe has sufficient data for plotting based on key columns"""
    if df.empty:
        return True
        
    # Check if metric columns exist and have data
    for col in metric_cols:
        if col not in df.columns or df[col].dropna().empty:
            return True
    
    # If grouping, check if we have multiple groups
    if group_cols:
        for col in group_cols:
            if col not in df.columns:
                return True
            if len(df[col].unique()) < 1:  # Need at least one group
                return True
    
    return False

def format_axis_labels(ax, x_rotation: int = 45, y_rotation: int = 0, 
                       max_x_chars: int = 15, max_y_chars: int = 15):
    """Format axis labels for better readability"""
    # Handle x-axis labels
    if x_rotation != 0:
        plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha='right' if x_rotation > 30 else 'center')
    
    # Handle y-axis labels
    if y_rotation != 0:
        plt.setp(ax.get_yticklabels(), rotation=y_rotation)
    
    # Truncate long x-tick labels
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    if max([len(str(label)) for label in x_labels]) > max_x_chars:
        new_labels = [str(label)[:max_x_chars] + '...' if len(str(label)) > max_x_chars else str(label) for label in x_labels]
        ax.set_xticklabels(new_labels)
    
    # Truncate long y-tick labels
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    if max([len(str(label)) for label in y_labels]) > max_y_chars:
        new_labels = [str(label)[:max_y_chars] + '...' if len(str(label)) > max_y_chars else str(label) for label in y_labels]
        ax.set_yticklabels(new_labels)

def add_value_labels(ax, spacing: int = 5, formatter=None, orientation: str = 'v'):
    """Add value labels on bars"""
    # Set default formatter
    if formatter is None:
        formatter = lambda x: f"{x:.2f}"
    
    # Vertical bars
    if orientation == 'v':
        for rect in ax.patches:
            height = rect.get_height()
            if height != 0:  # Only annotate non-zero bars
                ax.annotate(formatter(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, spacing),  # vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    # Horizontal bars
    else:
        for rect in ax.patches:
            width = rect.get_width()
            if width != 0:  # Only annotate non-zero bars
                ax.annotate(formatter(width),
                            xy=(width, rect.get_y() + rect.get_height() / 2),
                            xytext=(spacing, 0),  # horizontal offset
                            textcoords="offset points",
                            ha='left', va='center',
                            fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

def generate_chunking_figures(df: pd.DataFrame, output_dir: str):
    """Generate enhanced figures for chunking experiments"""
    set_ieee_style()
    
    # Filter to chunking experiment results
    chunking_df = df[df["experiment"] == "chunking"].copy()
    
    # Create directory for chunking figures
    chunking_dir = os.path.join(output_dir, "chunking")
    os.makedirs(chunking_dir, exist_ok=True)
    
    # Figure 1: Chunking Strategy Comparison
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Check if we have enough data for plotting
    if check_empty_data(chunking_df, metrics, ["chunking_strategy", "retrieval_method"]):
        print("Not enough data to generate chunking strategy comparison figure")
    else:
        # Create a new aggregated dataframe for plotting
        plot_data = pd.melt(
            chunking_df, 
            id_vars=["chunking_strategy", "retrieval_method"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean up metric names for better readability
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "").str.replace("_", " ").str.title()
        
        # Use categorical data type to control order
        plot_data["chunking_strategy"] = pd.Categorical(
            plot_data["chunking_strategy"],
            categories=["fixed", "recursive", "token", "sentence", "paragraph", "semantic"],
            ordered=True
        )
        
        # Set figure size based on number of strategies and metrics
        fig_size = set_figure_size("wide" if len(plot_data["chunking_strategy"].unique()) > 4 else "medium")
        
        # Create facet grid with larger figure for better spacing
        g = sns.catplot(
            data=plot_data,
            x="chunking_strategy",
            y="Score",
            hue="retrieval_method",
            col="Metric",
            kind="bar",
            height=fig_size[1]/2,  # Adjust height for better facet grid aspect
            aspect=1.0,
            sharey=False,
            palette=get_color_palette(len(plot_data["retrieval_method"].unique())),
            legend_out=False,
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Enhance the appearance
        g.set_axis_labels("Chunking Strategy", "Score")
        g.set_titles("{col_name}", fontsize=12)
        
        # Format x-axis labels to prevent overlap
        for ax in g.axes.flat:
            format_axis_labels(ax, x_rotation=45)
            
            # Add value labels to the bars
            for patch in ax.patches:
                height = patch.get_height()
                if not np.isnan(height) and height > 0:  # Only label non-zero bars
                    ax.text(
                        patch.get_x() + patch.get_width()/2,
                        height + 0.01,
                        f'{height:.2f}',
                        ha='center',
                        fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", ec="gray", fc="white", alpha=0.7)
                    )
        
        # Enhance legend
        g._legend.set_title("Retrieval Method")
        for t in g._legend.texts:
            t.set_text(t.get_text().title())  # Capitalize legend entries
            
        plt.tight_layout()
        g.fig.subplots_adjust(top=0.9)  # Make room for title
        g.fig.suptitle("Comparison of Chunking Strategies Across Different Metrics", fontsize=14)
        
        # Save with higher quality
        plt.savefig(os.path.join(chunking_dir, "fig1_chunking_strategy_comparison.png"), dpi=300)
        plt.close()
    
    # Figure 2: Chunk Size Impact (for fixed strategy)
    fixed_df = chunking_df[chunking_df["chunking_strategy"] == "fixed"].copy()
    if check_empty_data(fixed_df, ["chunk_size", "metric_mrr"], ["retrieval_method", "chunk_overlap"]):
        print("Not enough data to generate chunk size impact figure")
    else:
        plt.figure(figsize=set_figure_size("medium"))
        
        # Get unique overlap values and assign markers
        overlap_values = fixed_df["chunk_overlap"].unique()
        markers = ['o', 's', '^', 'D', '*', 'X', 'P'][:len(overlap_values)]
        
        # Create mapping of overlap values to markers
        overlap_markers = dict(zip(overlap_values, markers))
        
        # Sort data for connected lines
        fixed_df = fixed_df.sort_values(["retrieval_method", "chunk_size"])
        
        # Create line plot with enhanced styling
        g = sns.lineplot(
            data=fixed_df,
            x="chunk_size",
            y="metric_mrr",
            hue="retrieval_method",
            style="chunk_overlap",
            markers=True,
            dashes=False,
            palette=get_color_palette(len(fixed_df["retrieval_method"].unique())),
            linewidth=2.5,
            markersize=10,
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Add data points with white edgecolor for better visibility
        for method in fixed_df["retrieval_method"].unique():
            for overlap in fixed_df["chunk_overlap"].unique():
                subset = fixed_df[(fixed_df["retrieval_method"] == method) & 
                                 (fixed_df["chunk_overlap"] == overlap)]
                if not subset.empty:
                    marker = overlap_markers.get(overlap, 'o')
                    plt.plot(subset["chunk_size"], subset["metric_mrr"], 
                           marker=marker, linestyle='none', 
                           markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Enhance grid and axes
        plt.grid(True, linestyle='--', alpha=0.7)
        g.set(xlabel="Chunk Size (tokens)", ylabel="Mean Reciprocal Rank (MRR)")
        
        # Format x-axis as integer values
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        
        # Add more space to y-axis to avoid cutting off markers
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)
        
        # Improve legend - split into two legends for better clarity
        h1, l1 = g.get_legend_handles_labels()
        
        # Split the handles and labels
        retrieval_methods = fixed_df["retrieval_method"].unique()
        overlap_values = fixed_df["chunk_overlap"].unique()
        
        method_handles = h1[:len(retrieval_methods)]
        method_labels = l1[:len(retrieval_methods)]
        
        overlap_handles = h1[len(retrieval_methods):]
        overlap_labels = [f"Overlap: {ol}" for ol in l1[len(retrieval_methods):]]
        
        # Remove the original legend
        g.get_legend().remove()
        
        # Add the first legend for retrieval methods
        first_legend = plt.legend(method_handles, method_labels, 
                                title="Retrieval Method", 
                                loc='upper left', 
                                frameon=True,
                                framealpha=0.9,
                                edgecolor='gray')
        
        # Add the second legend for overlap values
        plt.legend(overlap_handles, overlap_labels, 
                  title="Chunk Overlap", 
                  loc='upper right', 
                  frameon=True,
                  framealpha=0.9,
                  edgecolor='gray')
        
        # Add the first legend back
        plt.gca().add_artist(first_legend)
        
        # Add title and adjust layout
        plt.title("Impact of Chunk Size on MRR", fontsize=14, pad=20)
        plt.tight_layout()
        
        # Annotate best configuration
        best_idx = fixed_df["metric_mrr"].idxmax()
        best_config = fixed_df.loc[best_idx]
        
        plt.annotate(f"Best: Size={best_config['chunk_size']}, Overlap={best_config['chunk_overlap']}",
                   xy=(best_config["chunk_size"], best_config["metric_mrr"]),
                   xytext=(0, 30),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
                   fontsize=10)
        
        plt.savefig(os.path.join(chunking_dir, "fig2_chunk_size_impact.png"), dpi=300)
        plt.close()
    
    # Generate improved summary table for paper (LaTeX format)
    if check_empty_data(chunking_df, metrics, ["chunking_strategy", "retrieval_method"]):
        print("Not enough data to generate chunking performance table")
    else:
        summary = chunking_df.groupby(["chunking_strategy", "retrieval_method"])[
            ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
        ].mean().reset_index()
        
        # Add standard deviation columns
        std_dev = chunking_df.groupby(["chunking_strategy", "retrieval_method"])[
            ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
        ].std().reset_index()
        
        # Rename standard deviation columns
        std_dev.columns = [
            "chunking_strategy", "retrieval_method", 
            "std_precision", "std_recall", "std_mrr"
        ]
        
        # Merge mean and std dev
        summary = pd.merge(summary, std_dev, on=["chunking_strategy", "retrieval_method"])
        
        # Format the table for IEEE paper
        summary.columns = [
            "Chunking Strategy", "Retrieval Method", 
            "Precision@3", "Recall@3", "MRR", 
            "Std Precision", "Std Recall", "Std MRR"
        ]
        
        # Highlight best values per metric
        best_precision_idx = summary["Precision@3"].idxmax()
        best_recall_idx = summary["Recall@3"].idxmax()
        best_mrr_idx = summary["MRR"].idxmax()
        
        # Save as CSV (which can be imported into LaTeX)
        summary.to_csv(os.path.join(chunking_dir, "table1_chunking_performance.csv"), index=False,
                   float_format="%.4f")
        
        # Also save as LaTeX table with better formatting
        with open(os.path.join(chunking_dir, "table1_chunking_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Chunking Strategies}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|l|l|c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Chunking} & \\textbf{Retrieval} & \\multicolumn{2}{c|}{\\textbf{Precision@3}} & \\multicolumn{2}{c|}{\\textbf{Recall@3}} & \\multicolumn{2}{c|}{\\textbf{MRR}} \\\\ \n")
            f.write("\\textbf{Strategy} & \\textbf{Method} & \\multicolumn{2}{c|}{} & \\multicolumn{2}{c|}{} & \\multicolumn{2}{c|}{} \\\\ \n")
            f.write("\\hline\n")
            
            for idx, row in summary.iterrows():
                # Prepare formatting for best values
                precision_format = "\\textbf{%.4f}" if idx == best_precision_idx else "%.4f"
                recall_format = "\\textbf{%.4f}" if idx == best_recall_idx else "%.4f"
                mrr_format = "\\textbf{%.4f}" if idx == best_mrr_idx else "%.4f"
                
                f.write(f"{row['Chunking Strategy']} & {row['Retrieval Method']} & ")
                f.write(f"{precision_format % row['Precision@3']} & {row['Std Precision']:.4f} & ")
                f.write(f"{recall_format % row['Recall@3']} & {row['Std Recall']:.4f} & ")
                f.write(f"{mrr_format % row['MRR']} & {row['Std MRR']:.4f} \\\\ \n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:chunking}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

def generate_embedding_figures(df: pd.DataFrame, output_dir: str):
    """Generate enhanced figures for embedding experiments"""
    set_ieee_style()
    
    # Filter to embedding experiment results
    embedding_df = df[df["experiment"] == "embedding"].copy()
    
    # Create directory for embedding figures
    embedding_dir = os.path.join(output_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Figure 3: Embedding Model Performance Comparison
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Check if we have enough data
    if check_empty_data(embedding_df, metrics, ["embedding_model"]):
        print("Not enough data to generate embedding model performance figure")
    else:
        plot_data = pd.melt(
            embedding_df, 
            id_vars=["embedding_model", "embedding_dim"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean up metric names for better readability
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "").str.replace("_", " ").str.title()
        
        # Determine figure size based on number of models
        fig_size = "wide" if len(plot_data["embedding_model"].unique()) > 4 else "medium"
        
        plt.figure(figsize=set_figure_size(fig_size))
        
        # Use custom color palette
        palette = get_color_palette(len(plot_data["Metric"].unique()))
        
        # Sort by embedding dimension for logical ordering
        plot_data = plot_data.sort_values(by=["embedding_dim", "embedding_model"])
        
        # Create a grouped bar chart
        g = sns.catplot(
            data=plot_data,
            x="embedding_model",
            y="Score",
            hue="Metric",
            kind="bar",
            height=5,
            aspect=1.5,
            palette=palette,
            legend=False,  # We'll add a custom legend
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Add value labels on bars
        for ax in g.axes.flat:
            add_value_labels(ax, formatter=lambda x: f"{x:.2f}")
        
        # Adjust x-axis labels to prevent overlap
        plt.xticks(rotation=45, ha="right")
        
        # Add embedding dimensions to x-axis labels
        ax = plt.gca()
        old_labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        
        for label in old_labels:
            dim = plot_data[plot_data["embedding_model"] == label]["embedding_dim"].iloc[0]
            new_labels.append(f"{label}\n({dim} dim)")
        
        ax.set_xticklabels(new_labels)
        
        # Add custom legend with better positioning
        plt.legend(title="Metric", loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
        
        # Add grid for easier comparison
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Label axes
        plt.xlabel("Embedding Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        
        # Add title with more information
        plt.title("Embedding Model Performance Comparison", fontsize=14, pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(embedding_dir, "fig3_embedding_model_performance.png"), dpi=300)
        plt.close()
    
    # Figure 4: Embedding Efficiency (Time vs Performance)
    if "embedding_time_sec" in embedding_df.columns and "metric_mrr" in embedding_df.columns:
        if embedding_df.empty or embedding_df["embedding_time_sec"].isnull().all():
            print("No embedding time data available for efficiency plot")
        else:
            plt.figure(figsize=set_figure_size("medium"))
            
            # Normalize embedding time to per-document time
            doc_counts = embedding_df.groupby("embedding_model").size()
            embedding_models = embedding_df["embedding_model"].unique()
            
            # Check if we can calculate per-document time
            has_doc_count = True
            for model in embedding_models:
                if model not in doc_counts or doc_counts[model] == 0:
                    has_doc_count = False
                    break
            
            if has_doc_count:
                # Calculate time per document
                embedding_df["time_per_doc"] = embedding_df.apply(
                    lambda row: row["embedding_time_sec"] / doc_counts[row["embedding_model"]], axis=1
                )
                time_col = "time_per_doc"
                time_label = "Embedding Time per Document (seconds)"
            else:
                # Use total time if per-document can't be calculated
                time_col = "embedding_time_sec"
                time_label = "Total Embedding Time (seconds)"
            
            # Create scatter plot with improved styling
            scatter = plt.scatter(
                embedding_df[time_col],
                embedding_df["metric_mrr"],
                s=150,  # Larger markers
                alpha=0.8,
                c=range(len(embedding_df)),  # Color points by index
                cmap="viridis",
                edgecolor="white",
                linewidth=1.5
            )
            
            # Add text labels for each point with better positioning and styling
            for i, row in embedding_df.iterrows():
                # Slightly different positions based on point position to avoid overlap
                offset_x = 0.1 * (1 - i / len(embedding_df))  # Varied horizontal offset
                offset_y = 0.01 * (1 + i % 2)  # Alternating vertical offset
                
                plt.annotate(
                    row["embedding_model"],
                    (row[time_col] + offset_x, row["metric_mrr"] + offset_y),
                    fontsize=9,
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add dimension info to the plot
            for i, row in embedding_df.iterrows():
                plt.annotate(
                    f"{int(row['embedding_dim'])}d",
                    (row[time_col], row["metric_mrr"]),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold'
                )
            
            # Add diagonal line showing efficiency frontier
            if not embedding_df.empty:
                # Calculate correlation between time and performance
                corr = embedding_df[time_col].corr(embedding_df["metric_mrr"])
                
                # Add correlation info to plot
                plt.annotate(
                    f"Correlation: {corr:.2f}",
                    (0.05, 0.05),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add efficiency frontier (points that offer best performance for given time)
            # Sort by time
            frontier_df = embedding_df.sort_values(by=time_col)
            max_mrr = 0
            frontier_points = []
            
            for i, row in frontier_df.iterrows():
                if row["metric_mrr"] > max_mrr:
                    max_mrr = row["metric_mrr"]
                    frontier_points.append((row[time_col], row["metric_mrr"]))
            
            if frontier_points:
                frontier_x, frontier_y = zip(*frontier_points)
                plt.plot(frontier_x, frontier_y, 'r--', linewidth=2, label="Efficiency Frontier")
            
            # Enhance axes and labels
            plt.title("Embedding Model Efficiency: Time vs. Performance", fontsize=14, pad=20)
            plt.xlabel(time_label, fontsize=12)
            plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend if we have the frontier
            if frontier_points:
                plt.legend(loc="lower right")
            
            # Adjust plot limits to accommodate labels
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            
            plt.xlim(x_min, x_max * 1.2)  # Add 20% space on the right
            plt.ylim(y_min * 0.95, y_max * 1.05)  # Add 5% space on top and bottom
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(embedding_dir, "fig4_embedding_efficiency.png"), dpi=300)
            plt.close()
    
    # Generate improved summary table for paper
    if check_empty_data(embedding_df, metrics + ["embedding_dim"]):
        print("Not enough data to generate embedding performance table")
    else:
        # Calculate summary statistics
        summary = embedding_df.groupby(["embedding_model", "embedding_dim"])[
            ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr",
             "embedding_time_sec", "docs_per_second"]
        ].agg(['mean', 'std']).reset_index()
        
        # Flatten the MultiIndex in columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Rename columns for readability
        summary = summary.rename(columns={
            "embedding_model_": "Model",
            "embedding_dim_": "Dimension",
            "metric_precision_at_3_mean": "Precision@3",
            "metric_precision_at_3_std": "Precision_std",
            "metric_recall_at_3_mean": "Recall@3",
            "metric_recall_at_3_std": "Recall_std",
            "metric_mrr_mean": "MRR",
            "metric_mrr_std": "MRR_std",
            "embedding_time_sec_mean": "Embedding Time (s)",
            "embedding_time_sec_std": "Time_std",
            "docs_per_second_mean": "Docs/Second",
            "docs_per_second_std": "Docs/Second_std"
        })
        
        # Identify best performance for each metric
        best_precision_idx = summary["Precision@3"].idxmax()
        best_recall_idx = summary["Recall@3"].idxmax()
        best_mrr_idx = summary["MRR"].idxmax()
        best_speed_idx = summary["Docs/Second"].idxmax()
        
        # Save as CSV with improved formatting
        summary.to_csv(
            os.path.join(embedding_dir, "table2_embedding_performance.csv"),
            index=False, 
            float_format="%.4f"
        )
        
        # Generate improved LaTeX table with standard deviations
        with open(os.path.join(embedding_dir, "table2_embedding_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Embedding Models}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|l|c|c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Model} & \\textbf{Dim} & \\multicolumn{2}{c|}{\\textbf{Precision@3}} & \\multicolumn{2}{c|}{\\textbf{Recall@3}} & \\multicolumn{2}{c|}{\\textbf{MRR}} & \\multicolumn{2}{c|}{\\textbf{Docs/Sec}} \\\\ \n")
            f.write("\\hline\n")
            
            for idx, row in summary.iterrows():
                # Formatting with bold for best values
                precision_format = "\\textbf{%.4f}" if idx == best_precision_idx else "%.4f"
                recall_format = "\\textbf{%.4f}" if idx == best_recall_idx else "%.4f"
                mrr_format = "\\textbf{%.4f}" if idx == best_mrr_idx else "%.4f"
                speed_format = "\\textbf{%.2f}" if idx == best_speed_idx else "%.2f"
                
                f.write(f"{row['Model']} & {int(row['Dimension'])} & ")
                f.write(f"{precision_format % row['Precision@3']} & {row['Precision_std']:.4f} & ")
                f.write(f"{recall_format % row['Recall@3']} & {row['Recall_std']:.4f} & ")
                f.write(f"{mrr_format % row['MRR']} & {row['MRR_std']:.4f} & ")
                f.write(f"{speed_format % row['Docs/Second']} & {row['Docs/Second_std']:.2f} \\\\ \n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:embedding}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

def generate_retrieval_figures(df: pd.DataFrame, output_dir: str):
    """Generate enhanced figures for retrieval method experiments"""
    set_ieee_style()
    
    # Filter to retrieval experiment results
    retrieval_df = df[df["experiment"] == "retrieval"].copy()
    
    # Create directory for retrieval figures
    retrieval_dir = os.path.join(output_dir, "retrieval")
    os.makedirs(retrieval_dir, exist_ok=True)
    
    # Figure 5: Retrieval Method Comparison
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Check if we have enough data
    if check_empty_data(retrieval_df, metrics, ["retrieval_method"]):
        print("Not enough data to generate retrieval method comparison figure")
    else:
        plot_data = pd.melt(
            retrieval_df, 
            id_vars=["retrieval_method", "retrieval_type"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean up metric names
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "").str.replace("_", " ").str.title()
        
        # Set figure size based on number of methods
        fig_size = "wide" if len(plot_data["retrieval_method"].unique()) > 4 else "medium"
        
        plt.figure(figsize=set_figure_size(fig_size))
        
        # Create enhanced grouped bar chart
        sns.set_style("whitegrid")
        
        # Create grouped bar plot with enhanced styling
        g = sns.catplot(
            data=plot_data,
            x="retrieval_method",
            y="Score",
            hue="Metric",
            kind="bar",
            height=5,
            aspect=1.5,
            palette=get_color_palette(len(plot_data["Metric"].unique())),
            errwidth=1.5,
            capsize=0.05,
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Add value labels to bars
        for ax in g.axes.flat:
            add_value_labels(ax, formatter=lambda x: f"{x:.2f}")
        
        # Enhance labels
        g.set_axis_labels("Retrieval Method", "Score")
        
        # Add retrieval type to x-axis labels
        retrieval_types = {}
        for method, type_val in zip(retrieval_df["retrieval_method"], retrieval_df["retrieval_type"]):
            retrieval_types[method] = type_val
        
        ax = plt.gca()
        old_labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        
        for label in old_labels:
            if label in retrieval_types:
                type_val = retrieval_types[label]
                new_labels.append(f"{label}\n({type_val})")
            else:
                new_labels.append(label)
        
        # Rotate x-tick labels for better fit
        plt.xticks(range(len(new_labels)), new_labels, rotation=45, ha="right")
        
        # Enhance legend
        plt.legend(title="Metric", loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
        
        # Add title
        plt.title("Retrieval Method Performance Comparison", fontsize=14, pad=20)
        
        # Improve grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(retrieval_dir, "fig5_retrieval_method_comparison.png"), dpi=300)
        plt.close()
    
    # Figure 6: Impact of Query Expansion
    if "query_expansion" in retrieval_df.columns:
        # Filter to include only rows with query expansion data
        expansion_df = retrieval_df[retrieval_df["query_expansion"].notna()].copy()
        
        if not expansion_df.empty:
            # Convert to categorical type to help with plotting
            expansion_df["query_expansion"] = expansion_df["query_expansion"].astype(str)
            
            # Use descriptive labels for expansion methods
            expansion_mapping = {
                "none": "No Expansion",
                "simple": "Simple Expansion",
                "llm": "LLM Expansion",
                "hybrid": "Hybrid Expansion",
                "True": "Expanded",
                "False": "Not Expanded"
            }
            
            # Apply mapping with fallback to original value
            expansion_df["expansion_label"] = expansion_df["query_expansion"].map(
                lambda x: expansion_mapping.get(x, x)
            )
            
            plt.figure(figsize=set_figure_size("medium"))
            
            # Create grouped bar chart with enhanced styling
            g = sns.catplot(
                data=expansion_df,
                x="retrieval_method",
                y="metric_mrr",
                hue="expansion_label",
                kind="bar",
                height=5,
                aspect=1.5,
                palette=get_color_palette(len(expansion_df["expansion_label"].unique())),
                legend_out=False,
                errorbar=("ci", 95)  # Add confidence intervals
            )
            
            # Add value labels
            for ax in g.axes.flat:
                add_value_labels(ax, formatter=lambda x: f"{x:.3f}")
            
            # Enhance appearance
            g.set_axis_labels("Retrieval Method", "Mean Reciprocal Rank (MRR)")
            plt.xticks(rotation=45, ha="right")
            
            # Add percentage improvement annotations
            if "none" in expansion_df["query_expansion"].values:
                for method in expansion_df["retrieval_method"].unique():
                    method_df = expansion_df[expansion_df["retrieval_method"] == method]
                    if "none" in method_df["query_expansion"].values:
                        baseline = method_df[method_df["query_expansion"] == "none"]["metric_mrr"].mean()
                        for exp_type in method_df["query_expansion"].unique():
                            if exp_type != "none":
                                exp_mrr = method_df[method_df["query_expansion"] == exp_type]["metric_mrr"].mean()
                                improvement = (exp_mrr - baseline) / baseline * 100
                                
                                # Find bar position for annotation
                                bar_idx = expansion_df[
                                    (expansion_df["retrieval_method"] == method) & 
                                    (expansion_df["query_expansion"] == exp_type)
                                ].index[0]
                                
                                # Add annotation about improvement
                                plt.annotate(
                                    f"+{improvement:.1f}%",
                                    xy=(0, 0),  # Will be updated by get_bar_positions
                                    xytext=(0, 15),
                                    textcoords="offset points",
                                    ha='center',
                                    va='bottom',
                                    fontsize=9,
                                    fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                                )
            
            # Add title
            plt.title("Impact of Query Expansion on Retrieval Performance", fontsize=14, pad=20)
            
            # Enhance legend
            g._legend.set_title("Query Expansion")
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(retrieval_dir, "fig6_query_expansion_impact.png"), dpi=300)
            plt.close()
    
    # Figure 7: Hybrid Search Alpha Parameter Impact
    # Filter to hybrid methods with alpha parameter
    if "alpha" in retrieval_df.columns:
        hybrid_df = retrieval_df[
            (retrieval_df["retrieval_type"] == "hybrid") & 
            (retrieval_df["alpha"].notna())
        ].copy()
        
        if not hybrid_df.empty:
            plt.figure(figsize=set_figure_size("medium"))
            
            # Group by alpha and get mean MRR and confidence intervals
            alpha_data = []
            
            for alpha_val in sorted(hybrid_df["alpha"].unique()):
                alpha_subset = hybrid_df[hybrid_df["alpha"] == alpha_val]
                mean_mrr = alpha_subset["metric_mrr"].mean()
                std_mrr = alpha_subset["metric_mrr"].std()
                count = len(alpha_subset)
                
                # Calculate 95% confidence interval
                if count > 1:
                    ci_95 = 1.96 * std_mrr / np.sqrt(count)
                else:
                    ci_95 = 0
                
                alpha_data.append({
                    "alpha": alpha_val,
                    "mrr": mean_mrr,
                    "ci_lower": mean_mrr - ci_95,
                    "ci_upper": mean_mrr + ci_95
                })
            
            alpha_groups = pd.DataFrame(alpha_data)
            
            # Create line plot with enhanced styling
            plt.plot(
                alpha_groups["alpha"], 
                alpha_groups["mrr"],
                marker='o',
                linewidth=2.5,
                markersize=10,
                color="#3182bd"
            )
            
            # Add confidence interval
            plt.fill_between(
                alpha_groups["alpha"],
                alpha_groups["ci_lower"],
                alpha_groups["ci_upper"],
                alpha=0.2,
                color="#3182bd",
                label="95% Confidence Interval"
            )
            
            # Add data points
            plt.scatter(
                alpha_groups["alpha"],
                alpha_groups["mrr"],
                s=100,
                c="#3182bd",
                edgecolor="white",
                linewidth=1.5,
                zorder=5
            )
            
            # Add value labels
            for i, row in alpha_groups.iterrows():
                plt.annotate(
                    f"{row['mrr']:.3f}",
                    (row["alpha"], row["mrr"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                )
            
            # Find optimal alpha
            best_idx = alpha_groups["mrr"].idxmax()
            best_alpha = alpha_groups.iloc[best_idx]["alpha"]
            best_mrr = alpha_groups.iloc[best_idx]["mrr"]
            
            # Highlight optimal point
            plt.scatter(
                [best_alpha],
                [best_mrr],
                s=200,
                c="gold",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
                label=f"Optimal α = {best_alpha:.1f}"
            )
            
            # Add visual indicators for vector vs BM25 weight
            cmap = plt.cm.coolwarm
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, orientation="horizontal", pad=0.2)
            cbar.set_label("Vector ← Weight Distribution → BM25")
            cbar.ax.set_xticks([0, 0.5, 1])
            cbar.ax.set_xticklabels(["BM25 Only\n(α=0)", "Equal Weight\n(α=0.5)", "Vector Only\n(α=1)"])
            
            # Add explanatory annotations
            plt.annotate(
                "Higher α → More weight to vector search",
                xy=(0.75, alpha_groups["mrr"].min()),
                xytext=(0.75, alpha_groups["mrr"].min() - 0.05),
                ha='center',
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black")
            )
            
            plt.annotate(
                "Lower α → More weight to BM25",
                xy=(0.25, alpha_groups["mrr"].min()),
                xytext=(0.25, alpha_groups["mrr"].min() - 0.05),
                ha='center',
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black")
            )
            
            # Enhance axes and labels
            plt.title("Impact of Alpha Parameter on Hybrid Retrieval", fontsize=14, pad=20)
            plt.xlabel("Alpha (Vector Weight)", fontsize=12)
            plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Set x-ticks to alpha values for clarity
            plt.xticks(alpha_groups["alpha"])
            
            # Add legend
            plt.legend(loc="upper left")
            
            # Adjust y-axis to make room for annotations
            y_min, y_max = plt.ylim()
            y_range = y_max - y_min
            plt.ylim(y_min - 0.15 * y_range, y_max + 0.05 * y_range)
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(retrieval_dir, "fig7_hybrid_alpha_impact.png"), dpi=300)
            plt.close()
    
    # Generate improved summary table for paper
    if check_empty_data(retrieval_df, metrics):
        print("Not enough data to generate retrieval performance table")
    else:
        # Get relevant grouping columns that exist in the data
        group_cols = ["retrieval_method", "retrieval_type"]
        if "query_expansion" in retrieval_df.columns:
            group_cols.append("query_expansion")
        
        # Calculate summary statistics
        summary = retrieval_df.groupby(group_cols)[metrics].agg(['mean', 'std']).reset_index()
        
        # Flatten the MultiIndex in columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Rename columns for better readability
        column_mapping = {
            "retrieval_method_": "Method",
            "retrieval_type_": "Type",
            "query_expansion_": "Query Expansion",
            "metric_precision_at_3_mean": "Precision@3",
            "metric_precision_at_3_std": "Precision_std",
            "metric_recall_at_3_mean": "Recall@3",
            "metric_recall_at_3_std": "Recall_std",
            "metric_mrr_mean": "MRR",
            "metric_mrr_std": "MRR_std"
        }
        
        # Apply column renaming (only for columns that exist)
        for old_col, new_col in column_mapping.items():
            if old_col in summary.columns:
                summary = summary.rename(columns={old_col: new_col})
        
        # Find best performance for each metric
        best_precision_idx = summary["Precision@3"].idxmax() if "Precision@3" in summary.columns else None
        best_recall_idx = summary["Recall@3"].idxmax() if "Recall@3" in summary.columns else None
        best_mrr_idx = summary["MRR"].idxmax() if "MRR" in summary.columns else None
        
        # Save as CSV
        summary.to_csv(
            os.path.join(retrieval_dir, "table3_retrieval_performance.csv"),
            index=False, 
            float_format="%.4f"
        )
        
        # Generate enhanced LaTeX table
        with open(os.path.join(retrieval_dir, "table3_retrieval_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Retrieval Methods}\n")
            f.write("\\begin{center}\n")
            
            # Construct table format based on available columns
            table_format = "|l|l|"
            if "Query Expansion" in summary.columns:
                table_format += "l|"
            
            # Add format for metric columns with standard deviation
            table_format += "c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|"
            
            f.write(f"\\begin{tabular}{{{table_format}}}\n")
            f.write("\\hline\n")
            
            # Construct header based on available columns
            header = "\\textbf{Method} & \\textbf{Type}"
            if "Query Expansion" in summary.columns:
                header += " & \\textbf{Query Expansion}"
            
            header += " & \\multicolumn{2}{c|}{\\textbf{Precision@3}} & \\multicolumn{2}{c|}{\\textbf{Recall@3}} & \\multicolumn{2}{c|}{\\textbf{MRR}} \\\\ \n"
            f.write(header)
            
            f.write("\\hline\n")
            
            # Write data rows with highlighting for best values
            for idx, row in summary.iterrows():
                # Prepare formatting for best values
                precision_format = "\\textbf{%.4f}" if idx == best_precision_idx else "%.4f"
                recall_format = "\\textbf{%.4f}" if idx == best_recall_idx else "%.4f"
                mrr_format = "\\textbf{%.4f}" if idx == best_mrr_idx else "%.4f"
                
                # Construct row based on available columns
                row_str = f"{row['Method']} & {row['Type']}"
                if "Query Expansion" in summary.columns:
                    expansion_val = row['Query Expansion']
                    # Clean up boolean strings for better display
                    if expansion_val == "True":
                        expansion_val = "Yes"
                    elif expansion_val == "False":
                        expansion_val = "No"
                    row_str += f" & {expansion_val}"
                
                # Add metric values with standard deviations
                row_str += f" & {precision_format % row['Precision@3']} & {row['Precision_std']:.4f}"
                row_str += f" & {recall_format % row['Recall@3']} & {row['Recall_std']:.4f}"
                row_str += f" & {mrr_format % row['MRR']} & {row['MRR_std']:.4f} \\\\ \n"
                
                f.write(row_str)
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:retrieval}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

def generate_query_processing_figures(df: pd.DataFrame, output_dir: str):
    """Generate enhanced figures for query processing experiments"""
    set_ieee_style()
    
    # Filter to query processing experiment results
    query_df = df[df["experiment"] == "query_processing"].copy()
    
    # Create directory for query processing figures
    query_dir = os.path.join(output_dir, "query_processing")
    os.makedirs(query_dir, exist_ok=True)
    
    # Figure 8: Query Processing Technique Comparison
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Check if we have enough data
    if check_empty_data(query_df, metrics, ["technique"]):
        print("Not enough data to generate query processing technique comparison figure")
    else:
        plot_data = pd.melt(
            query_df, 
            id_vars=["technique", "description"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean up metric names for better display
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "").str.replace("_", " ").str.title()
        
        # Determine figure size based on number of techniques
        fig_size = "wide" if len(plot_data["technique"].unique()) > 4 else "medium"
        
        # Create figure with enhanced styling
        plt.figure(figsize=set_figure_size(fig_size))
        
        # Create grouped bar chart
        g = sns.catplot(
            data=plot_data,
            x="technique",
            y="Score",
            hue="Metric",
            kind="bar",
            height=5,
            aspect=1.5,
            palette=get_color_palette(len(plot_data["Metric"].unique())),
            legend=False,  # We'll add a custom legend
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Add value labels to bars
        for ax in g.axes.flat:
            add_value_labels(ax, formatter=lambda x: f"{x:.2f}")
        
        # Get the technique descriptions for later use
        technique_descriptions = {}
        for tech, desc in zip(query_df["technique"], query_df["description"]):
            if tech not in technique_descriptions and not pd.isna(desc):
                technique_descriptions[tech] = desc
        
        # Adjust x-axis labels with descriptions
        ax = plt.gca()
        old_labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        
        # Truncate descriptions to reasonable length for x-axis
        for label in old_labels:
            if label in technique_descriptions:
                desc = technique_descriptions[label]
                if len(desc) > 30:  # Truncate long descriptions
                    desc = desc[:27] + "..."
                new_labels.append(f"{label}\n({desc})")
            else:
                new_labels.append(label)
        
        # Add x-axis labels with rotation for better fit
        plt.xticks(range(len(new_labels)), new_labels, rotation=45, ha="right")
        
        # Add custom legend with better positioning
        plt.legend(title="Metric", loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
        
        # Add grid for easier comparison
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add title and labels
        plt.title("Query Processing Technique Performance Comparison", fontsize=14, pad=20)
        plt.xlabel("Query Processing Technique", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        
        # Find and mark best technique for MRR
        if "technique" in query_df.columns and "metric_mrr" in query_df.columns:
            mrr_by_technique = query_df.groupby("technique")["metric_mrr"].mean()
            best_technique = mrr_by_technique.idxmax()
            best_mrr = mrr_by_technique.max()
            
            # Add annotation for best technique
            plt.annotate(
                f"Best: {best_technique}\nMRR: {best_mrr:.3f}",
                xy=(list(old_labels).index(best_technique), best_mrr),
                xytext=(0, 30),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="gold", ec="black", alpha=0.8),
                fontsize=10
            )
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(query_dir, "fig8_query_technique_comparison.png"), dpi=300)
        plt.close()
    
    # Figure 9: Efficiency vs. Performance
    if "processing_time_sec" in query_df.columns and "metric_mrr" in query_df.columns:
        if query_df["processing_time_sec"].isnull().all():
            print("No processing time data available for efficiency plot")
        else:
            plt.figure(figsize=set_figure_size("medium"))
            
            # Improve clarity by calculating queries per second
            query_df["queries_per_second"] = 1.0 / query_df["processing_time_sec"].clip(lower=0.001)
            
            # Get unique techniques for color coding
            techniques = query_df["technique"].unique()
            colors = get_color_palette(len(techniques))
            technique_colors = dict(zip(techniques, colors))
            
            # Create enhanced scatter plot
            for technique in techniques:
                subset = query_df[query_df["technique"] == technique]
                if not subset.empty and not subset["processing_time_sec"].isnull().all():
                    plt.scatter(
                        subset["processing_time_sec"],
                        subset["metric_mrr"],
                        s=150,
                        alpha=0.8,
                        label=technique,
                        color=technique_colors[technique],
                        edgecolor="white",
                        linewidth=1.5
                    )
            
            # Add text labels with enhanced styling
            for _, row in query_df.iterrows():
                if pd.notna(row["processing_time_sec"]) and pd.notna(row["metric_mrr"]):
                    plt.annotate(
                        row["technique"],
                        (row["processing_time_sec"], row["metric_mrr"]),
                        fontsize=9,
                        xytext=(5, 5),
                        textcoords="offset points",
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
            
            # Add efficiency frontier
            if not query_df.empty:
                # Get data points for frontier
                frontier_df = query_df.dropna(subset=["processing_time_sec", "metric_mrr"])
                frontier_df = frontier_df.sort_values("processing_time_sec")
                
                max_mrr = 0
                frontier_points = []
                
                for _, row in frontier_df.iterrows():
                    if row["metric_mrr"] > max_mrr:
                        max_mrr = row["metric_mrr"]
                        frontier_points.append((row["processing_time_sec"], row["metric_mrr"]))
                
                # Plot frontier line
                if len(frontier_points) > 1:
                    frontier_x, frontier_y = zip(*frontier_points)
                    plt.plot(frontier_x, frontier_y, 'k--', linewidth=2, label="Efficiency Frontier")
            
            # Add quadrant labels for easier interpretation
            x_mid = query_df["processing_time_sec"].median()
            y_mid = query_df["metric_mrr"].median()
            
            plt.annotate(
                "Fast & Effective\n(Optimal)",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.3)
            )
            
            plt.annotate(
                "Slow & Effective",
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                ha='right',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.3)
            )
            
            plt.annotate(
                "Fast & Ineffective",
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                fontsize=10,
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.3)
            )
            
            plt.annotate(
                "Slow & Ineffective\n(Avoid)",
                xy=(0.95, 0.05),
                xycoords='axes fraction',
                fontsize=10,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.3)
            )
            
            # Enhance axes and labels
            plt.title("Query Processing Efficiency vs. Performance", fontsize=14, pad=20)
            plt.xlabel("Processing Time (seconds)", fontsize=12)
            plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add secondary y-axis for queries per second
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            max_time = query_df["processing_time_sec"].max()
            min_time = query_df["processing_time_sec"].min()
            
            # Generate suitable time points for queries per second
            time_points = np.linspace(min_time * 1.1, max_time * 0.9, 5)
            qps_points = 1.0 / time_points
            
            ax2.set_ylabel("Queries Per Second", color="green", fontsize=12)
            ax2.tick_params(axis='y', colors="green")
            ax2.set_yscale('log')
            
            # Add legend with optimal positioning
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                legend = plt.legend(
                    handles, labels,
                    title="Technique",
                    loc="best",
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="gray"
                )
                legend.get_title().set_fontweight('bold')
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(query_dir, "fig9_query_efficiency_vs_performance.png"), dpi=300)
            plt.close()
    
    # Generate improved summary table
    if check_empty_data(query_df, metrics):
        print("Not enough data to generate query processing performance table")
    else:
        # Calculate summary statistics
        summary = query_df.groupby(["technique", "description"])[
            metrics
        ].agg(['mean', 'std']).reset_index()
        
        # Flatten the MultiIndex in columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Rename columns for better readability
        column_mapping = {
            "technique_": "Technique",
            "description_": "Description",
            "metric_precision_at_3_mean": "Precision@3",
            "metric_precision_at_3_std": "Precision_std",
            "metric_recall_at_3_mean": "Recall@3",
            "metric_recall_at_3_std": "Recall_std",
            "metric_mrr_mean": "MRR",
            "metric_mrr_std": "MRR_std"
        }
        
        # Apply column renaming
        for old_col, new_col in column_mapping.items():
            if old_col in summary.columns:
                summary = summary.rename(columns={old_col: new_col})
        
        # Find best performance for each metric
        best_precision_idx = summary["Precision@3"].idxmax() if "Precision@3" in summary.columns else None
        best_recall_idx = summary["Recall@3"].idxmax() if "Recall@3" in summary.columns else None
        best_mrr_idx = summary["MRR"].idxmax() if "MRR" in summary.columns else None
        
        # Save as CSV
        summary.to_csv(
            os.path.join(query_dir, "table4_query_processing_performance.csv"),
            index=False, 
            float_format="%.4f"
        )
        
        # Generate enhanced LaTeX table
        with open(os.path.join(query_dir, "table4_query_processing_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Query Processing Techniques}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|l|p{5cm}|c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Technique} & \\textbf{Description} & \\multicolumn{2}{c|}{\\textbf{Precision@3}} & \\multicolumn{2}{c|}{\\textbf{Recall@3}} & \\multicolumn{2}{c|}{\\textbf{MRR}} \\\\ \n")
            f.write("\\hline\n")
            
            for idx, row in summary.iterrows():
                # Prepare formatting for best values
                precision_format = "\\textbf{%.4f}" if idx == best_precision_idx else "%.4f"
                recall_format = "\\textbf{%.4f}" if idx == best_recall_idx else "%.4f"
                mrr_format = "\\textbf{%.4f}" if idx == best_mrr_idx else "%.4f"
                
                # Clean and truncate description if needed
                description = row['Description']
                if isinstance(description, str) and len(description) > 80:
                    description = description[:77] + "..."
                
                # Handle possible null values
                if pd.isna(description):
                    description = ""
                
                f.write(f"{row['Technique']} & {description} & ")
                f.write(f"{precision_format % row['Precision@3']} & {row['Precision_std']:.4f} & ")
                f.write(f"{recall_format % row['Recall@3']} & {row['Recall_std']:.4f} & ")
                f.write(f"{mrr_format % row['MRR']} & {row['MRR_std']:.4f} \\\\ \n")

            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:query_processing}\n")
            f.write("\\end{center}\n")
            f.write("\\end{table}\n")

def generate_reranking_figures(df: pd.DataFrame, output_dir: str):
    """Generate enhanced figures for reranking experiments"""
    set_ieee_style()
    
    # Filter to reranking experiment results
    reranking_df = df[df["experiment"] == "reranking"].copy()
    
    # Create directory for reranking figures
    reranking_dir = os.path.join(output_dir, "reranking")
    os.makedirs(reranking_dir, exist_ok=True)
    
    # Figure 10: Reranking Method Comparison
    metrics = ["metric_precision_at_3", "metric_recall_at_3", "metric_mrr"]
    
    # Check if we have enough data
    if check_empty_data(reranking_df, metrics, ["reranking_method"]):
        print("Not enough data to generate reranking method comparison figure")
    else:
        plot_data = pd.melt(
            reranking_df, 
            id_vars=["reranking_method", "description"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean up metric names
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "").str.replace("_", " ").str.title()
        
        # Determine figure size based on number of methods
        fig_size = "wide" if len(plot_data["reranking_method"].unique()) > 4 else "medium"
        
        # Create figure with enhanced styling
        plt.figure(figsize=set_figure_size(fig_size))
        
        # Create grouped bar chart
        g = sns.catplot(
            data=plot_data,
            x="reranking_method",
            y="Score",
            hue="Metric",
            kind="bar",
            height=5,
            aspect=1.5,
            palette=get_color_palette(len(plot_data["Metric"].unique())),
            legend=False,  # We'll add a custom legend
            errorbar=("ci", 95)  # Add confidence intervals
        )
        
        # Add value labels to bars
        for ax in g.axes.flat:
            add_value_labels(ax, formatter=lambda x: f"{x:.2f}")
        
        # Get the method descriptions for later use
        method_descriptions = {}
        for method, desc in zip(reranking_df["reranking_method"], reranking_df["description"]):
            if method not in method_descriptions and not pd.isna(desc):
                method_descriptions[method] = desc
        
        # Adjust x-axis labels with descriptions
        ax = plt.gca()
        old_labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = []
        
        # Truncate descriptions to reasonable length for x-axis
        for label in old_labels:
            if label in method_descriptions:
                desc = method_descriptions[label]
                if len(desc) > 30:  # Truncate long descriptions
                    desc = desc[:27] + "..."
                new_labels.append(f"{label}\n({desc})")
            else:
                new_labels.append(label)
        
        # Add x-axis labels with rotation for better fit
        plt.xticks(range(len(new_labels)), new_labels, rotation=45, ha="right")
        
        # Add custom legend with better positioning
        plt.legend(title="Metric", loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
        
        # Add grid for easier comparison
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add title and labels
        plt.title("Reranking Method Performance Comparison", fontsize=14, pad=20)
        plt.xlabel("Reranking Method", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        
        # Find baseline method if it exists
        if "no_reranking" in reranking_df["reranking_method"].values:
            baseline_idx = list(old_labels).index("no_reranking")
            
            # Highlight baseline bar with distinct pattern
            bars = [patch for patch in plt.gca().patches if isinstance(patch, plt.Rectangle)]
            
            # Group bars by metric (assuming 3 metrics)
            metrics_count = len(metrics)
            bars_per_method = metrics_count
            
            # Add hatching to baseline bars
            for i in range(bars_per_method):
                baseline_bar_idx = baseline_idx * bars_per_method + i
                if baseline_bar_idx < len(bars):
                    bars[baseline_bar_idx].set_hatch('////')
                    bars[baseline_bar_idx].set_edgecolor('black')
                    bars[baseline_bar_idx].set_linewidth(1.5)
            
            # Add annotation for baseline
            plt.annotate(
                "Baseline\n(No Reranking)",
                xy=(baseline_idx, 0),
                xytext=(0, -40),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                fontsize=10
            )
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(reranking_dir, "fig10_reranking_method_comparison.png"), dpi=300)
        plt.close()
    
    # Figure 11: Reranking Improvement over Baseline
    # Get baseline performance (no_reranking)
    if "no_reranking" in reranking_df["reranking_method"].values and "metric_mrr" in reranking_df.columns:
        baseline_mrr = reranking_df[reranking_df["reranking_method"] == "no_reranking"]["metric_mrr"].mean()
        
        # Calculate improvement over baseline
        improvement_df = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
        
        if not improvement_df.empty:
            improvement_df["improvement"] = improvement_df["metric_mrr"] - baseline_mrr
            improvement_df["relative_improvement"] = improvement_df["improvement"] / baseline_mrr * 100
            
            # Calculate standard error for relative improvement
            improvement_stats = improvement_df.groupby("reranking_method").agg({
                "relative_improvement": ["mean", "std", "count"]
            })
            
            # Flatten multi-index columns
            improvement_stats.columns = ["_".join(col).strip("_") for col in improvement_stats.columns]
            
            # Calculate standard error
            improvement_stats["stderr"] = improvement_stats["relative_improvement_std"] / np.sqrt(improvement_stats["relative_improvement_count"])
            
            # Calculate 95% confidence intervals
            improvement_stats["ci_lower"] = improvement_stats["relative_improvement_mean"] - 1.96 * improvement_stats["stderr"]
            improvement_stats["ci_upper"] = improvement_stats["relative_improvement_mean"] + 1.96 * improvement_stats["stderr"]
            
            # Reset index for plotting
            improvement_stats = improvement_stats.reset_index()
            
            # Sort by improvement for better visualization
            improvement_stats = improvement_stats.sort_values("relative_improvement_mean", ascending=False)
            
            plt.figure(figsize=set_figure_size("medium"))
            
            # Create enhanced bar plot
            bars = plt.bar(
                improvement_stats["reranking_method"],
                improvement_stats["relative_improvement_mean"],
                yerr=[
                    improvement_stats["relative_improvement_mean"] - improvement_stats["ci_lower"],
                    improvement_stats["ci_upper"] - improvement_stats["relative_improvement_mean"]
                ],
                capsize=5,
                color=get_color_palette(len(improvement_stats)),
                edgecolor="white",
                linewidth=1.5,
                error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1)
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                )
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Add visual cues for improvement vs degradation
            plt.axhspan(0, max(improvement_stats["ci_upper"]) * 1.1, alpha=0.1, color="green", label="Improvement")
            plt.axhspan(min(improvement_stats["ci_lower"]) * 1.1 if min(improvement_stats["ci_lower"]) < 0 else -1, 0, alpha=0.1, color="red", label="Degradation")
            
            # Enhance appearance
            plt.title("Reranking Performance Improvement over Baseline", fontsize=14, pad=20)
            plt.xlabel("Reranking Method", fontsize=12)
            plt.ylabel("Relative Improvement (%)", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha="right")
            
            # Add legend
            plt.legend(loc="upper right")
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(reranking_dir, "fig11_reranking_improvement.png"), dpi=300)
            plt.close()
    
    # Figure 12: Reranking Time vs. Performance Improvement
    if ("reranking_time_sec" in reranking_df.columns and 
        "no_reranking" in reranking_df["reranking_method"].values and 
        "metric_mrr" in reranking_df.columns):
        
        baseline_mrr = reranking_df[reranking_df["reranking_method"] == "no_reranking"]["metric_mrr"].mean()
        
        # Calculate improvement and efficiency
        improvement_df = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
        
        if not improvement_df.empty and not improvement_df["reranking_time_sec"].isnull().all():
            improvement_df["improvement"] = improvement_df["metric_mrr"] - baseline_mrr
            improvement_df["relative_improvement"] = improvement_df["improvement"] / baseline_mrr * 100
            
            # Calculate average values per reranking method
            avg_improvement = improvement_df.groupby("reranking_method").agg({
                "reranking_time_sec": "mean",
                "relative_improvement": "mean"
            }).reset_index()
            
            plt.figure(figsize=set_figure_size("medium"))
            
            # Create enhanced scatter plot
            scatter = plt.scatter(
                avg_improvement["reranking_time_sec"],
                avg_improvement["relative_improvement"],
                s=200,
                c=range(len(avg_improvement)),
                cmap="viridis",
                alpha=0.8,
                edgecolor="white",
                linewidth=1.5
            )
            
            # Add text labels with better positioning and styling
            for _, row in avg_improvement.iterrows():
                plt.annotate(
                    row["reranking_method"],
                    (row["reranking_time_sec"], row["relative_improvement"]),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add quadrant divisions with interpretations
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Find reasonable time threshold - median or specific threshold
            time_threshold = avg_improvement["reranking_time_sec"].median()
            plt.axvline(x=time_threshold, color='black', linestyle='-', alpha=0.3)
            
            # Add quadrant labels
            plt.annotate(
                "Fast & Better\n(Optimal)",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.3)
            )
            
            plt.annotate(
                "Slow & Better",
                xy=(0.95, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                ha='right',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.3)
            )
            
            plt.annotate(
                "Fast & Worse\n(Use baseline)",
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                fontsize=10,
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", alpha=0.3)
            )
            
            plt.annotate(
                "Slow & Worse\n(Avoid)",
                xy=(0.95, 0.05),
                xycoords='axes fraction',
                fontsize=10,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="red", alpha=0.3)
            )
            
            # Add median time annotation
            plt.annotate(
                f"Median time: {time_threshold:.3f}s",
                xy=(time_threshold, avg_improvement["relative_improvement"].min()),
                xytext=(0, -30),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=9
            )
            
            # Enhance axes and labels
            plt.title("Reranking Efficiency vs. Performance Improvement", fontsize=14, pad=20)
            plt.xlabel("Reranking Time (seconds)", fontsize=12)
            plt.ylabel("Relative Improvement (%)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Calculate and add efficiency frontier
            if len(avg_improvement) > 1:
                # Sort by time
                frontier_df = avg_improvement.sort_values("reranking_time_sec")
                
                max_improvement = float('-inf')
                frontier_points = []
                
                for _, row in frontier_df.iterrows():
                    if row["relative_improvement"] > max_improvement:
                        max_improvement = row["relative_improvement"]
                        frontier_points.append((row["reranking_time_sec"], row["relative_improvement"]))
                
                if frontier_points:
                    frontier_x, frontier_y = zip(*frontier_points)
                    plt.plot(frontier_x, frontier_y, 'r--', linewidth=2, label="Efficiency Frontier")
                    
                    # Add legend
                    plt.legend(loc="upper right")
            
            # Find and mark the best method (highest improvement)
            best_idx = avg_improvement["relative_improvement"].idxmax()
            best_method = avg_improvement.iloc[best_idx]["reranking_method"]
            best_improvement = avg_improvement.iloc[best_idx]["relative_improvement"]
            best_time = avg_improvement.iloc[best_idx]["reranking_time_sec"]
            
            # Add annotation for best method
            plt.annotate(
                f"Best: {best_method}\n+{best_improvement:.1f}%",
                xy=(best_time, best_improvement),
                xytext=(0, 30),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="gold", ec="black", alpha=0.8),
                fontsize=10
            )
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(reranking_dir, "fig12_reranking_efficiency_vs_improvement.png"), dpi=300)
            plt.close()
    
    # Generate enhanced summary table
    if check_empty_data(reranking_df, metrics):
        print("Not enough data to generate reranking performance table")
    else:
        # Calculate summary statistics
        summary = reranking_df.groupby(["reranking_method", "description"])[
            metrics
        ].agg(['mean', 'std']).reset_index()
        
        # Flatten the MultiIndex in columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Rename columns for better readability
        column_mapping = {
            "reranking_method_": "Method",
            "description_": "Description",
            "metric_precision_at_3_mean": "Precision@3",
            "metric_precision_at_3_std": "Precision_std",
            "metric_recall_at_3_mean": "Recall@3",
            "metric_recall_at_3_std": "Recall_std",
            "metric_mrr_mean": "MRR",
            "metric_mrr_std": "MRR_std"
        }
        
        # Apply column renaming
        for old_col, new_col in column_mapping.items():
            if old_col in summary.columns:
                summary = summary.rename(columns={old_col: new_col})
        
        # Find best performance for each metric
        best_precision_idx = summary["Precision@3"].idxmax() if "Precision@3" in summary.columns else None
        best_recall_idx = summary["Recall@3"].idxmax() if "Recall@3" in summary.columns else None
        best_mrr_idx = summary["MRR"].idxmax() if "MRR" in summary.columns else None
        
        # Save as CSV
        summary.to_csv(
            os.path.join(reranking_dir, "table5_reranking_performance.csv"),
            index=False, 
            float_format="%.4f"
        )
        
        # Generate enhanced LaTeX table
        with open(os.path.join(reranking_dir, "table5_reranking_performance.tex"), "w") as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\caption{Performance Comparison of Different Reranking Methods}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{|l|p{5cm}|c@{$\\pm$}c|c@{$\\pm$}c|c@{$\\pm$}c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Method} & \\textbf{Description} & \\multicolumn{2}{c|}{\\textbf{Precision@3}} & \\multicolumn{2}{c|}{\\textbf{Recall@3}} & \\multicolumn{2}{c|}{\\textbf{MRR}} \\\\ \n")
            f.write("\\hline\n")
            
            for idx, row in summary.iterrows():
                # Prepare formatting for best values
                precision_format = "\\textbf{%.4f}" if idx == best_precision_idx else "%.4f"
                recall_format = "\\textbf{%.4f}" if idx == best_recall_idx else "%.4f"
                mrr_format = "\\textbf{%.4f}" if idx == best_mrr_idx else "%.4f"
                
                # Clean and truncate description if needed
                description = row['Description']
                if isinstance(description, str) and len(description) > 80:
                    description = description[:77] + "..."
                
                # Handle possible null values
                if pd.isna(description):
                    description = ""
                
                f.write(f"{row['Method']} & {description} & ")
                f.write(f"{precision_format % row['Precision@3']} & {row['Precision_std']:.4f} & ")
                f.write(f"{recall_format % row['Recall@3']} & {row['Recall_std']:.4f} & ")
                f.write(f"{mrr_format % row['MRR']} & {row['MRR_std']:.4f} \\\\ \n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:reranking}\n")
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