# analysis/analyze_results.py
# COMPLETE FILE - Incorporates fixes for directory handling, NameErrors, and KeyError: nan in analyze_* functions

"""
Results Analysis

This script analyzes experimental results from different RAG configurations,
generates visualizations, and provides insights for the research paper.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np
import json
import traceback # Added for better error printing if needed

# Add project root if necessary (assuming it's needed based on previous context)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- analyze_experiment_results Function (WITH ALL PREVIOUS FIXES) ---
def analyze_experiment_results(results_dirs: Optional[List[str]] = None, output_dir: str = "figures"):
    """
    Analyze and visualize results from experiments (Corrected Directory Handling & NameError Fix)

    Args:
        results_dirs: List of result directories to analyze (if None, searches in results/)
        output_dir: Directory to save output figures and analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    actual_experiment_dirs_to_process = [] # List to store paths like 'results/chunking_...'
    base_dirs_to_scan = []
    if results_dirs is None:
        default_base = "results"
        if os.path.exists(default_base) and os.path.isdir(default_base):
             base_dirs_to_scan = [default_base]
        else: print(f"Default results directory '{default_base}' not found."); return
    else: base_dirs_to_scan = results_dirs

    for base_dir in base_dirs_to_scan:
        if not os.path.exists(base_dir): print(f"Warning: Provided directory '{base_dir}' not found."); continue
        if not os.path.isdir(base_dir): print(f"Warning: Provided path '{base_dir}' is not a directory."); continue
        try:
            found_subdirs_in_base = False
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and "experiment" in item and os.path.exists(os.path.join(item_path, "results.csv")):
                    actual_experiment_dirs_to_process.append(item_path); found_subdirs_in_base = True
            if not found_subdirs_in_base and os.path.exists(os.path.join(base_dir, "results.csv")):
                 print(f"Note: Treating '{base_dir}' as a single experiment directory."); actual_experiment_dirs_to_process.append(base_dir)
        except Exception as e: print(f"Error scanning directory '{base_dir}': {e}")

    if not actual_experiment_dirs_to_process: print("No valid experiment result directories found."); return

    all_results = []
    for result_dir in actual_experiment_dirs_to_process:
        result_csv = os.path.join(result_dir, "results.csv")
        if not os.path.exists(result_csv): continue # Skip dirs without results.csv silently

        # --- Define experiment_name (Moved definition here, includes NameError fix) ---
        try:
             dir_name = os.path.basename(result_dir)
             dir_parts = dir_name.split('_'); timestamp_index = -1
             for i, part in enumerate(dir_parts):
                  if part.isdigit() and len(part) >= 8: timestamp_index = i; break
             if timestamp_index > 0: experiment_name = "_".join(dir_parts[:timestamp_index])
             else: experiment_name = dir_name if "experiment" in dir_name else dir_parts[0]
        except Exception as e_name: print(f"Warning: Could not derive experiment name from dir '{result_dir}': {e_name}"); experiment_name = "unknown"
        # --- End Define experiment_name ---

        try: # Load results df
            df = pd.read_csv(result_csv)
            df["experiment"] = experiment_name.replace("_experiment", "") # Simplify name
            all_results.append(df)
        except Exception as e: print(f"Error reading CSV {result_csv}: {e}")

    if not all_results: print("No results found to analyze"); return
    combined_df = pd.concat(all_results, ignore_index=True)
    try: # Save combined results
        combined_df.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)
    except Exception as e_comb: print(f"Warning: Failed to save combined results: {e_comb}")

    # --- Call analysis functions (Keep as is) ---
    if "chunking" in combined_df["experiment"].values: analyze_chunking_results(combined_df, output_dir)
    if "embedding" in combined_df["experiment"].values: analyze_embedding_results(combined_df, output_dir)
    if "retrieval" in combined_df["experiment"].values: analyze_retrieval_results(combined_df, output_dir)
    if "query_processing" in combined_df["experiment"].values: analyze_query_processing_results(combined_df, output_dir)
    if "reranking" in combined_df["experiment"].values: analyze_reranking_results(combined_df, output_dir)
    if "generation" in combined_df["experiment"].values: analyze_generation_results(combined_df, output_dir)
    generate_comparative_analysis(combined_df, output_dir)
# --- End analyze_experiment_results ---


# --- analyze_chunking_results Function (WITH ALL PREVIOUS FIXES - Verbatim) ---
def analyze_chunking_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from chunking experiments (With idxmax NaN checks)"""
    chunking_df = df[df["experiment"] == "chunking"].copy()
    if chunking_df.empty: print("No chunking experiment results found"); return
    print("Analyzing chunking experiment results...")
    chunking_dir = os.path.join(output_dir, "chunking_analysis"); os.makedirs(chunking_dir, exist_ok=True)
    try: # Plotting
        plot_metrics_by_category(chunking_df, x_col="chunking_strategy", title="Retrieval Performance by Chunking Strategy", output_file=os.path.join(chunking_dir, "chunking_strategy_comparison.png"))
        fixed_df = chunking_df[chunking_df["chunking_strategy"] == "fixed"].copy()
        if not fixed_df.empty:
            plot_metrics_by_category(fixed_df, x_col="chunk_size", title="Retrieval Performance by Chunk Size", output_file=os.path.join(chunking_dir, "chunk_size_comparison.png"))
            if "chunk_overlap" in fixed_df.columns and fixed_df["chunk_overlap"].nunique() > 1:
                 fixed_df["chunk_config"] = fixed_df.apply(lambda row: f"{row['chunk_size']}/{row['chunk_overlap']}", axis=1)
                 plot_metrics_by_category(fixed_df, x_col="chunk_config", title="Retrieval Performance by Chunk Size/Overlap", output_file=os.path.join(chunking_dir, "chunk_overlap_comparison.png"))
        plot_metrics_by_category(chunking_df, x_col="retrieval_method", hue_col="chunking_strategy", title="Retrieval Methods Comparison by Chunking Strategy", output_file=os.path.join(chunking_dir, "retrieval_method_by_chunking.png"))
        if "num_chunks" in chunking_df.columns:
            plt.figure(figsize=(10, 6))
            plot_df = chunking_df.drop_duplicates(subset=["chunking_strategy", "chunk_size", "chunk_overlap"])
            sns.barplot(data=plot_df, x="chunking_strategy", y="num_chunks")
            plt.title("Number of Chunks Generated by Strategy"); plt.ylabel("Number of Chunks"); plt.xticks(rotation=45); plt.tight_layout()
            plt.savefig(os.path.join(chunking_dir, "chunks_by_strategy.png")); plt.close()
    except Exception as e_plot: print(f"Warning: Failed during chunking plot generation: {e_plot}")
    try: # Summary
        summary = chunking_df.groupby(["chunking_strategy", "retrieval_method"])[[col for col in chunking_df.columns if col.startswith("metric_")]].mean().reset_index()
        summary.to_csv(os.path.join(chunking_dir, "chunking_summary.csv"), index=False)
    except Exception as e_summary: print(f"Warning: Failed to create/save chunking summary table: {e_summary}")
    print("\nBest configurations for chunking experiment:"); metric_columns_to_check = [col for col in chunking_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in chunking_df.columns and not chunking_df[metric].isna().all():
                best_idx = chunking_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = chunking_df.loc[best_idx]
                    print(f"Best {metric}:"); print(f" Chunking: {best_config.get('chunking_strategy', 'N/A')}")
                    if best_config.get('chunking_strategy') == 'fixed': print(f" Size: {best_config.get('chunk_size', 'N/A')}"); print(f" Overlap: {best_config.get('chunk_overlap', 'N/A')}")
                    print(f" Retrieval: {best_config.get('retrieval_method', 'N/A')}"); print(f" Score: {best_config.get(metric, np.nan):.4f}\n")
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; mrr_col = None # Recommendations
    if "metric_mrr" in chunking_df.columns and not chunking_df["metric_mrr"].isna().all(): mrr_col = "metric_mrr"
    else: potential_metrics = [col for col in metric_columns_to_check if col in chunking_df.columns and not chunking_df[col].isna().all()]; mrr_col = potential_metrics[0] if potential_metrics else None
    if mrr_col:
        try: # Best overall
             best_idx = chunking_df[mrr_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = chunking_df.loc[best_idx]; rec = {"chunking_strategy": best_config.get("chunking_strategy"), "retrieval_method": best_config.get("retrieval_method"), "metric": mrr_col, "score": float(best_config.get(mrr_col, np.nan))}
                 if best_config.get("chunking_strategy") == "fixed":
                      try: rec["chunk_size"] = int(best_config.get("chunk_size")); rec["chunk_overlap"] = int(best_config.get("chunk_overlap"))
                      except: pass
                 recommendations["best_overall"] = rec
        except Exception as e_rec1: print(f"Error generating best overall chunking recommendation: {e_rec1}")
        method_recommendations = {} # By method
        try:
            for method in chunking_df["retrieval_method"].dropna().unique():
                method_df = chunking_df[chunking_df["retrieval_method"] == method]
                if not method_df.empty and mrr_col in method_df.columns and not method_df[mrr_col].isna().all():
                    best_idx = method_df[mrr_col].idxmax()
                    if not pd.isna(best_idx):
                        best_config = method_df.loc[best_idx]; rec = {"chunking_strategy": best_config.get("chunking_strategy"), "score": float(best_config.get(mrr_col, np.nan))}
                        if best_config.get("chunking_strategy") == "fixed":
                            try: rec["chunk_size"] = int(best_config.get("chunk_size")); rec["chunk_overlap"] = int(best_config.get("chunk_overlap"))
                            except: pass
                        method_recommendations[str(method)] = rec
        except Exception as e_rec2: print(f"Error generating chunking recommendations by retrieval method: {e_rec2}")
        recommendations["by_retrieval_method"] = method_recommendations
    try: # Save recommendations
        with open(os.path.join(chunking_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save chunking recommendations JSON: {e_json}")
# --- End analyze_chunking_results ---


# --- analyze_embedding_results Function (WITH ALL PREVIOUS FIXES - Verbatim) ---
def analyze_embedding_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from embedding experiments (With idxmax NaN checks)"""
    embedding_df = df[df["experiment"] == "embedding"].copy()
    if embedding_df.empty: print("No embedding experiment results found"); return
    print("Analyzing embedding experiment results...")
    embedding_dir = os.path.join(output_dir, "embedding_analysis"); os.makedirs(embedding_dir, exist_ok=True)
    try: # Plotting
         plot_metrics_by_category(embedding_df, x_col="embedding_model", title="Retrieval Performance by Embedding Model", output_file=os.path.join(embedding_dir, "embedding_model_comparison.png"))
         if "embedding_time_sec" in embedding_df.columns:
             plt.figure(figsize=(10, 6)); sns.barplot(data=embedding_df, x="embedding_model", y="embedding_time_sec")
             plt.title("Embedding Generation Time by Model"); plt.xticks(rotation=45); plt.ylabel("Time (seconds)"); plt.tight_layout()
             plt.savefig(os.path.join(embedding_dir, "embedding_time_comparison.png")); plt.close()
         if "embedding_time_sec" in embedding_df.columns and "metric_mrr" in embedding_df.columns:
             valid_time_df = embedding_df[(embedding_df["embedding_time_sec"] > 0) & (embedding_df["embedding_time_sec"].notna())].copy()
             if not valid_time_df.empty:
                 x_axis_col = "docs_per_second"; x_label = "Documents per Second"
                 if "docs_per_second" not in valid_time_df.columns:
                      print("Warning: 'docs_per_second' column not found. Using proxy.")
                      avg_times = valid_time_df.groupby('embedding_model')['embedding_time_sec'].mean()
                      valid_time_df['calculated_efficiency'] = valid_time_df['embedding_model'].map(1.0 / avg_times)
                      x_axis_col = 'calculated_efficiency'; x_label = "Runs per Second (Proxy)"
                 if x_axis_col in valid_time_df.columns and "metric_mrr" in valid_time_df.columns and not valid_time_df["metric_mrr"].isna().all():
                      plt.figure(figsize=(10, 6)); sns.scatterplot(data=valid_time_df, x=x_axis_col, y="metric_mrr", hue="embedding_model", s=100, alpha=0.7)
                      plt.title("Embedding Efficiency vs. Performance"); plt.xlabel(x_label); plt.ylabel("Mean Reciprocal Rank (MRR)"); plt.grid(True, linestyle='--', alpha=0.7)
                      plt.tight_layout(); plt.savefig(os.path.join(embedding_dir, "embedding_efficiency_vs_performance.png")); plt.close()
                 else: print("Warning: Cannot plot efficiency vs performance - required columns invalid.")
             else: print("Warning: Cannot plot efficiency vs performance - no valid embedding times.")
    except Exception as e_plot: print(f"Warning: Failed during embedding plot generation: {e_plot}")
    try: # Summary
        efficiency_cols = []; group_cols = ["embedding_model", "embedding_dim"]; metric_cols = [col for col in embedding_df.columns if col.startswith("metric_")]
        if "docs_per_second" in embedding_df.columns and not embedding_df["docs_per_second"].isna().all(): efficiency_cols.append("docs_per_second")
        elif "calculated_efficiency" in embedding_df.columns and not embedding_df["calculated_efficiency"].isna().all(): efficiency_cols.append("calculated_efficiency")
        if "embedding_time_sec" in embedding_df.columns and not embedding_df["embedding_time_sec"].isna().all(): efficiency_cols.append("embedding_time_sec")
        valid_group_cols = [col for col in group_cols if col in embedding_df.columns]; valid_metric_cols = [col for col in metric_cols if col in embedding_df.columns]; valid_efficiency_cols = [col for col in efficiency_cols if col in embedding_df.columns]
        if valid_group_cols and (valid_metric_cols or valid_efficiency_cols):
             summary = embedding_df.groupby(valid_group_cols)[valid_metric_cols + valid_efficiency_cols].mean().reset_index()
             summary.to_csv(os.path.join(embedding_dir, "embedding_summary.csv"), index=False)
        else: print("Warning: Could not create embedding summary - required columns missing.")
    except Exception as e_summary: print(f"Warning: Failed to create/save embedding summary: {e_summary}")
    print("\nBest configurations for embedding experiment:"); metric_columns_to_check = [col for col in embedding_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in embedding_df.columns and not embedding_df[metric].isna().all():
                best_idx = embedding_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = embedding_df.loc[best_idx]
                    print(f"Best {metric}:"); print(f" Model: {best_config.get('embedding_model', 'N/A')}"); print(f" Dimension: {best_config.get('embedding_dim', 'N/A')}"); print(f" Score: {best_config.get(metric, np.nan):.4f}\n")
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; mrr_col = None; efficiency_col = None # Recommendations
    if "metric_mrr" in embedding_df.columns and not embedding_df["metric_mrr"].isna().all(): mrr_col = "metric_mrr"
    else: potential_metrics = [col for col in metric_columns_to_check if col in embedding_df.columns and not embedding_df[col].isna().all()]; mrr_col = potential_metrics[0] if potential_metrics else None
    if mrr_col: # Best performance
         try:
             best_idx = embedding_df[mrr_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = embedding_df.loc[best_idx]
                 recommendations["best_performance"] = {"embedding_model": best_config.get("embedding_model"), "embedding_dim": int(best_config.get("embedding_dim")) if pd.notna(best_config.get("embedding_dim")) else None, "metric": mrr_col, "score": float(best_config.get(mrr_col, np.nan))}
             else: print(f"Warning: Could not determine best overall embedding model based on {mrr_col}.")
         except Exception as e_rec1: print(f"Error generating best performance embedding recommendation: {e_rec1}")
    if "docs_per_second" in embedding_df.columns and not embedding_df["docs_per_second"].isna().all(): efficiency_col = "docs_per_second"
    elif "calculated_efficiency" in embedding_df.columns and not embedding_df["calculated_efficiency"].isna().all(): efficiency_col = "calculated_efficiency"
    if efficiency_col: # Best efficiency
         try:
             fastest_idx = embedding_df[efficiency_col].idxmax()
             if not pd.isna(fastest_idx):
                 fastest_config = embedding_df.loc[fastest_idx]
                 rec = {"embedding_model": fastest_config.get("embedding_model"), "embedding_dim": int(fastest_config.get("embedding_dim")) if pd.notna(fastest_config.get("embedding_dim")) else None, efficiency_col: float(fastest_config.get(efficiency_col, np.nan))}
                 if mrr_col: rec["score"] = float(fastest_config.get(mrr_col, np.nan))
                 recommendations["best_efficiency"] = rec
             else: print("Warning: Could not determine most efficient embedding model.")
         except Exception as e_rec2: print(f"Error generating best efficiency embedding recommendation: {e_rec2}")
    balance_possible = False # Best balance
    if mrr_col and efficiency_col and mrr_col in embedding_df.columns and efficiency_col in embedding_df.columns and not embedding_df[mrr_col].isna().all() and not embedding_df[efficiency_col].isna().all():
         try:
             max_score = embedding_df[mrr_col].max(); min_score = embedding_df[mrr_col].min(); max_eff = embedding_df[efficiency_col].max(); min_eff = embedding_df[efficiency_col].min(); score_range = max_score - min_score; eff_range = max_eff - min_eff
             if score_range > 0 and eff_range > 0:
                 embedding_df["norm_score"] = (embedding_df[mrr_col] - min_score) / score_range; embedding_df["norm_eff"] = (embedding_df[efficiency_col] - min_eff) / eff_range; embedding_df["balance"] = 0.5 * embedding_df["norm_score"] + 0.5 * embedding_df["norm_eff"]; balance_possible = True
             else: print("Warning: Cannot normalize for balance calc (zero range).")
         except Exception as e_norm: print(f"Warning: Error during normalization for balance: {e_norm}")
    if balance_possible and "balance" in embedding_df.columns and not embedding_df["balance"].isna().all():
         try:
             balanced_idx = embedding_df["balance"].idxmax()
             if not pd.isna(balanced_idx):
                 balanced_config = embedding_df.loc[balanced_idx]
                 recommendations["best_balance"] = {"embedding_model": balanced_config.get("embedding_model"), "embedding_dim": int(balanced_config.get("embedding_dim")) if pd.notna(balanced_config.get("embedding_dim")) else None, "score": float(balanced_config.get(mrr_col, np.nan)), efficiency_col: float(balanced_config.get(efficiency_col, np.nan))}
             else: print("Warning: Could not determine best balanced embedding model.")
         except Exception as e_rec3: print(f"Error generating best balance embedding recommendation: {e_rec3}")
    try: # Save recommendations
        with open(os.path.join(embedding_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save embedding recommendations: {e_json}")
# --- End analyze_embedding_results ---


# --- analyze_retrieval_results Function (WITH LATEST FIXES - Verbatim) ---
def analyze_retrieval_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from retrieval method experiments (With idxmax NaN checks)"""
    retrieval_df = df[df["experiment"] == "retrieval"].copy()
    if retrieval_df.empty: print("No retrieval experiment results found"); return
    print("Analyzing retrieval experiment results...")
    retrieval_dir = os.path.join(output_dir, "retrieval_analysis"); os.makedirs(retrieval_dir, exist_ok=True)
    try: # Plotting
        plot_metrics_by_category(retrieval_df, x_col="retrieval_method", title="Retrieval Performance by Method", output_file=os.path.join(retrieval_dir, "retrieval_method_comparison.png"))
        query_expansion_df = retrieval_df[retrieval_df["query_expansion"].notna()].copy()
        if not query_expansion_df.empty:
            plot_metrics_by_category(query_expansion_df, x_col="retrieval_method", hue_col="query_expansion", title="Impact of Query Expansion", output_file=os.path.join(retrieval_dir, "query_expansion_impact.png"))
            if query_expansion_df["query_expansion"].nunique() > 1:
                plot_metrics_by_category(query_expansion_df, x_col="query_expansion", title="Impact of Expansion Method", output_file=os.path.join(retrieval_dir, "expansion_method_comparison.png"))
        hybrid_df = retrieval_df[retrieval_df["retrieval_type"] == "hybrid"].copy()
        if not hybrid_df.empty and "alpha" in hybrid_df.columns:
             plt.figure(figsize=(10, 6)); metrics = [col for col in hybrid_df.columns if col.startswith("metric_")]
             for metric in metrics:
                 if not hybrid_df[metric].isna().all(): alpha_means = hybrid_df.groupby("alpha")[metric].mean().reset_index(); plt.plot(alpha_means["alpha"], alpha_means[metric], marker='o', label=metric.replace("metric_", ""))
             plt.title("Impact of Alpha Parameter on Hybrid Retrieval"); plt.xlabel("Alpha (Vector Weight)"); plt.ylabel("Score"); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.tight_layout()
             plt.savefig(os.path.join(retrieval_dir, "hybrid_alpha_comparison.png")); plt.close()
        if "retrieval_time_sec" in retrieval_df.columns:
             plt.figure(figsize=(10, 6)); sns.barplot(data=retrieval_df, x="retrieval_method", y="retrieval_time_sec")
             plt.title("Retrieval Time by Method"); plt.xticks(rotation=45); plt.ylabel("Time (seconds)"); plt.tight_layout()
             plt.savefig(os.path.join(retrieval_dir, "retrieval_time_comparison.png")); plt.close()
        if "retrieval_time_sec" in retrieval_df.columns and "metric_mrr" in retrieval_df.columns:
             valid_time_df = retrieval_df[(retrieval_df["retrieval_time_sec"] > 0) & (retrieval_df["retrieval_time_sec"].notna())].copy()
             if not valid_time_df.empty:
                 valid_time_df["queries_per_second"] = 1.0 / valid_time_df["retrieval_time_sec"]
                 if "metric_mrr" in valid_time_df.columns and not valid_time_df["metric_mrr"].isna().all():
                      plt.figure(figsize=(10, 6)); sns.scatterplot(data=valid_time_df, x="queries_per_second", y="metric_mrr", hue="retrieval_method", s=100, alpha=0.7)
                      plt.title("Retrieval Efficiency vs. Performance"); plt.xlabel("Queries per Second"); plt.ylabel("Mean Reciprocal Rank (MRR)"); plt.grid(True, linestyle='--', alpha=0.7)
                      plt.tight_layout(); plt.savefig(os.path.join(retrieval_dir, "retrieval_efficiency_vs_performance.png")); plt.close()
                 else: print("Warning: Cannot plot retrieval efficiency vs performance - MRR invalid.")
             else: print("Warning: Cannot plot retrieval efficiency vs performance - time invalid.")
    except Exception as e_plot: print(f"Warning: Failed during retrieval plot generation: {e_plot}")
    try: # Summary
        group_cols = ["retrieval_method", "retrieval_type", "query_expansion"]; metric_cols = [col for col in retrieval_df.columns if col.startswith("metric_")]; efficiency_cols = []
        if "retrieval_time_sec" in retrieval_df.columns and not retrieval_df["retrieval_time_sec"].isna().all(): efficiency_cols.append("retrieval_time_sec")
        if "queries_per_second" in retrieval_df.columns and not retrieval_df["queries_per_second"].isna().all(): efficiency_cols.append("queries_per_second")
        elif "retrieval_time_sec" in efficiency_cols: retrieval_df["qps_calc"] = 1.0 / retrieval_df["retrieval_time_sec"]; efficiency_cols.append("qps_calc")
        valid_group_cols = [col for col in group_cols if col in retrieval_df.columns]; valid_metric_cols = [col for col in metric_cols if col in retrieval_df.columns]; valid_efficiency_cols = [col for col in efficiency_cols if col in retrieval_df.columns]
        if valid_group_cols and (valid_metric_cols or valid_efficiency_cols):
             summary = retrieval_df.groupby(valid_group_cols)[valid_metric_cols + valid_efficiency_cols].mean().reset_index()
             if "alpha" in retrieval_df.columns and "retrieval_method" in summary.columns:
                 hybrid_alphas = retrieval_df[retrieval_df["retrieval_type"] == "hybrid"].groupby("retrieval_method")["alpha"].first()
                 if not hybrid_alphas.empty: summary = summary.merge(hybrid_alphas.reset_index(), on="retrieval_method", how="left")
             summary.to_csv(os.path.join(retrieval_dir, "retrieval_summary.csv"), index=False)
        else: print("Warning: Could not create retrieval summary - columns missing.")
    except Exception as e_summary: print(f"Warning: Failed to create/save retrieval summary: {e_summary}")
    print("\nBest configurations for retrieval experiment:"); metric_columns_to_check = [col for col in retrieval_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in retrieval_df.columns and not retrieval_df[metric].isna().all():
                best_idx = retrieval_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = retrieval_df.loc[best_idx]
                    print(f"Best {metric}:"); print(f" Method: {best_config.get('retrieval_method', 'N/A')}"); print(f" Score: {best_config.get(metric, np.nan):.4f}\n")
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; mrr_col = None; # Recommendations
    if "metric_mrr" in retrieval_df.columns and not retrieval_df["metric_mrr"].isna().all(): mrr_col = "metric_mrr"
    else: potential_metrics = [col for col in metric_columns_to_check if col in retrieval_df.columns and not retrieval_df[col].isna().all()]; mrr_col = potential_metrics[0] if potential_metrics else None
    if mrr_col: # Best performance
         try:
             best_idx = retrieval_df[mrr_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = retrieval_df.loc[best_idx]; rec = {"retrieval_method": best_config.get("retrieval_method"), "retrieval_type": best_config.get("retrieval_type"), "query_expansion": str(best_config.get("query_expansion", "none")) if pd.notna(best_config.get("query_expansion")) else "none", "metric": mrr_col, "score": float(best_config.get(mrr_col, np.nan))}
                 if best_config.get("retrieval_type") == "hybrid" and pd.notna(best_config.get("alpha")):
                      try: rec["alpha"] = float(best_config.get("alpha"))
                      except: pass
                 recommendations["best_performance"] = rec
             else: print(f"Warning: Could not determine best overall retrieval method based on {mrr_col}.")
         except Exception as e_rec1: print(f"Error generating best performance retrieval recommendation: {e_rec1}")
    type_recommendations = {}; # By Type
    if mrr_col and "retrieval_type" in retrieval_df.columns:
         try:
             for r_type in retrieval_df["retrieval_type"].dropna().unique():
                 type_df = retrieval_df[retrieval_df["retrieval_type"] == r_type]
                 if not type_df.empty and mrr_col in type_df.columns and not type_df[mrr_col].isna().all():
                     best_idx = type_df[mrr_col].idxmax()
                     if not pd.isna(best_idx):
                         best_config = type_df.loc[best_idx]; rec = {"retrieval_method": best_config.get("retrieval_method"), "query_expansion": str(best_config.get("query_expansion", "none")) if pd.notna(best_config.get("query_expansion")) else "none", "score": float(best_config.get(mrr_col, np.nan))}
                         if r_type == "hybrid" and pd.notna(best_config.get("alpha")):
                              try: rec["alpha"] = float(best_config.get("alpha"))
                              except: pass
                         type_recommendations[str(r_type)] = rec
         except Exception as e_rec2: print(f"Error generating retrieval recommendations by type: {e_rec2}")
         recommendations["by_retrieval_type"] = type_recommendations
    if mrr_col and "query_expansion" in retrieval_df.columns and "retrieval_type" in retrieval_df.columns: # Expansion Impact
         expansion_impact = {}
         try:
             baseline_exists = "none" in retrieval_df["query_expansion"].astype(str).unique()
             if baseline_exists:
                 for base_method_type in retrieval_df["retrieval_type"].dropna().unique():
                     base_df = retrieval_df[retrieval_df["retrieval_type"] == base_method_type]; base_df["query_expansion_str"] = base_df["query_expansion"].astype(str)
                     if base_df.empty or mrr_col not in base_df.columns: continue
                     no_exp_df = base_df[base_df["query_expansion_str"] == "none"]
                     if not no_exp_df.empty and not no_exp_df[mrr_col].isna().all(): no_exp_score = no_exp_df[mrr_col].mean()
                     else: continue
                     exp_df = base_df[base_df["query_expansion_str"] != "none"]
                     if not exp_df.empty and not exp_df[mrr_col].isna().all():
                         best_exp_idx = exp_df[mrr_col].idxmax()
                         if not pd.isna(best_exp_idx):
                             best_exp_config = exp_df.loc[best_exp_idx]; exp_score = best_exp_config.get(mrr_col); best_exp_type = best_exp_config.get("query_expansion_str")
                             if pd.notna(exp_score) and pd.notna(no_exp_score) and no_exp_score != 0:
                                  improvement = (exp_score - no_exp_score) / abs(no_exp_score)
                                  expansion_impact[str(base_method_type)] = {"best_expansion": best_exp_type, "improvement": float(improvement), "base_score": float(no_exp_score), "with_expansion_score": float(exp_score)}
             else: print("Warning: Cannot calculate expansion impact - 'none' expansion results missing.")
         except Exception as e_rec3: print(f"Error generating query expansion impact analysis: {e_rec3}")
         recommendations["query_expansion_impact"] = expansion_impact
    hybrid_df_rec = retrieval_df[retrieval_df["retrieval_type"] == "hybrid"].copy() # Best Alpha
    if mrr_col and not hybrid_df_rec.empty and "alpha" in hybrid_df_rec.columns and not hybrid_df_rec["alpha"].isna().all():
         try:
             alpha_scores = hybrid_df_rec.groupby("alpha")[mrr_col].mean().reset_index()
             if not alpha_scores.empty and not alpha_scores[mrr_col].isna().all():
                 best_alpha_idx = alpha_scores[mrr_col].idxmax()
                 if not pd.isna(best_alpha_idx):
                     best_alpha_row = alpha_scores.loc[best_alpha_idx]
                     recommendations["best_hybrid_alpha"] = {"alpha": float(best_alpha_row.get("alpha", np.nan)), "score": float(best_alpha_row.get(mrr_col, np.nan))}
                 else: print("Warning: Could not determine best hybrid alpha.")
         except Exception as e_rec4: print(f"Error generating best hybrid alpha recommendation: {e_rec4}")
    try: # Save recommendations
        with open(os.path.join(retrieval_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save retrieval recommendations: {e_json}")
# --- End analyze_retrieval_results ---


# --- analyze_query_processing_results Function (Verbatim from paste-3.txt) ---
def analyze_query_processing_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from query processing experiments"""
    # [ ... Function code exactly as provided in paste-3.txt ... ]
    # Filter to just query_processing experiment results
    query_df = df[df["experiment"] == "query_processing"].copy()
    if query_df.empty: print("No query processing experiment results found"); return
    print("Analyzing query processing experiment results...")
    query_dir = os.path.join(output_dir, "query_processing_analysis"); os.makedirs(query_dir, exist_ok=True)
    try: # Plotting
        plot_metrics_by_category(query_df, x_col="technique", title="Retrieval Performance by Query Processing Technique", output_file=os.path.join(query_dir, "query_technique_comparison.png"))
        if "processing_time_sec" in query_df.columns:
            plt.figure(figsize=(10, 6)); sns.barplot(data=query_df, x="technique", y="processing_time_sec")
            plt.title("Query Processing Time by Technique"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(query_dir, "query_processing_time.png")); plt.close()
        if "processing_time_sec" in query_df.columns and "metric_mrr" in query_df.columns:
            valid_time_df = query_df[(query_df["processing_time_sec"] > 0) & (query_df["processing_time_sec"].notna())].copy()
            if not valid_time_df.empty:
                 valid_time_df["queries_per_second"] = 1.0 / valid_time_df["processing_time_sec"]
                 if "metric_mrr" in valid_time_df.columns and not valid_time_df["metric_mrr"].isna().all():
                      plt.figure(figsize=(10, 6)); sns.scatterplot(data=valid_time_df, x="queries_per_second", y="metric_mrr", hue="technique", s=100, alpha=0.7)
                      plt.title("Query Processing Efficiency vs. Performance"); plt.xlabel("Queries per Second"); plt.ylabel("Mean Reciprocal Rank (MRR)"); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
                      plt.savefig(os.path.join(query_dir, "query_efficiency_vs_performance.png")); plt.close()
                 else: print("Warning: Cannot plot query efficiency vs performance - MRR invalid.")
            else: print("Warning: Cannot plot query efficiency vs performance - time invalid.")
    except Exception as e_plot: print(f"Warning: Failed during query processing plot generation: {e_plot}")
    try: # Summary
        group_cols = ["technique", "description"]; metric_cols = [col for col in query_df.columns if col.startswith("metric_")]; efficiency_cols = []
        if "processing_time_sec" in query_df.columns and not query_df["processing_time_sec"].isna().all(): efficiency_cols.append("processing_time_sec")
        if "queries_per_second" in query_df.columns and not query_df["queries_per_second"].isna().all(): efficiency_cols.append("queries_per_second")
        elif "processing_time_sec" in efficiency_cols: query_df["qps_calc"] = 1.0 / query_df["processing_time_sec"]; efficiency_cols.append("qps_calc")
        valid_group_cols = [col for col in group_cols if col in query_df.columns]; valid_metric_cols = [col for col in metric_cols if col in query_df.columns]; valid_efficiency_cols = [col for col in efficiency_cols if col in query_df.columns]
        if valid_group_cols and (valid_metric_cols or valid_efficiency_cols):
             summary = query_df.groupby(valid_group_cols)[valid_metric_cols + valid_efficiency_cols].mean().reset_index()
             summary.to_csv(os.path.join(query_dir, "query_processing_summary.csv"), index=False)
        else: print("Warning: Could not create query processing summary - columns missing.")
    except Exception as e_summary: print(f"Warning: Failed to create/save query processing summary: {e_summary}")
    print("\nBest configurations for query processing experiment:"); metric_columns_to_check = [col for col in query_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in query_df.columns and not query_df[metric].isna().all():
                best_idx = query_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = query_df.loc[best_idx]
                    print(f"Best {metric}:"); print(f" Technique: {best_config.get('technique', 'N/A')} ({best_config.get('description', 'N/A')})"); print(f" Score: {best_config.get(metric, np.nan):.4f}")
                    if "processing_time_sec" in best_config and pd.notna(best_config['processing_time_sec']): print(f" Processing time: {best_config['processing_time_sec']:.2f} seconds")
                    print()
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; mrr_col = None; # Recommendations
    if "metric_mrr" in query_df.columns and not query_df["metric_mrr"].isna().all(): mrr_col = "metric_mrr"
    else: potential_metrics = [col for col in metric_columns_to_check if col in query_df.columns and not query_df[col].isna().all()]; mrr_col = potential_metrics[0] if potential_metrics else None
    if mrr_col: # Best performance
         try:
             best_idx = query_df[mrr_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = query_df.loc[best_idx]; rec = {"technique": best_config.get("technique"), "description": best_config.get("description"), "metric": mrr_col, "score": float(best_config.get(mrr_col, np.nan))}
                 if "processing_time_sec" in best_config and pd.notna(best_config["processing_time_sec"]): rec["processing_time_sec"] = float(best_config["processing_time_sec"])
                 recommendations["best_performance"] = rec
             else: print(f"Warning: Could not determine best overall query technique based on {mrr_col}.")
         except Exception as e_rec1: print(f"Error generating best performance query recommendation: {e_rec1}")
    if "processing_time_sec" in query_df.columns and not query_df["processing_time_sec"].isna().all(): # Best efficiency
         try:
             fastest_idx = query_df["processing_time_sec"].idxmin()
             if not pd.isna(fastest_idx):
                 fastest_config = query_df.loc[fastest_idx]; rec = {"technique": fastest_config.get("technique"), "description": fastest_config.get("description"), "processing_time_sec": float(fastest_config.get("processing_time_sec", np.nan))}
                 if mrr_col: rec["score"] = float(fastest_config.get(mrr_col, np.nan))
                 recommendations["best_efficiency"] = rec
             else: print("Warning: Could not determine most efficient query technique.")
         except Exception as e_rec2: print(f"Error generating best efficiency query recommendation: {e_rec2}")
    balance_possible = False # Best balance
    if mrr_col and "processing_time_sec" in query_df.columns and not query_df[mrr_col].isna().all() and not query_df["processing_time_sec"].isna().all():
        try:
            max_score = query_df[mrr_col].max(); min_score = query_df[mrr_col].min(); min_time = query_df["processing_time_sec"].min(); max_time = query_df["processing_time_sec"].max(); score_range = max_score - min_score; time_range = max_time - min_time
            if score_range > 0 and time_range > 0:
                 query_df["norm_score"] = (query_df[mrr_col] - min_score) / score_range; query_df["norm_speed"] = 1 - ((query_df["processing_time_sec"] - min_time) / time_range)
                 query_df["balance"] = 0.5 * query_df["norm_score"] + 0.5 * query_df["norm_speed"]; balance_possible = True
            else: print("Warning: Cannot normalize query results for balance calc (zero range).")
        except Exception as e_norm: print(f"Warning: Error during normalization for query balance: {e_norm}")
    if balance_possible and "balance" in query_df.columns and not query_df["balance"].isna().all():
        try:
             balanced_idx = query_df["balance"].idxmax()
             if not pd.isna(balanced_idx):
                 balanced_config = query_df.loc[balanced_idx]
                 recommendations["best_balance"] = {"technique": balanced_config.get("technique"), "description": balanced_config.get("description"), "score": float(balanced_config.get(mrr_col, np.nan)), "processing_time_sec": float(balanced_config.get("processing_time_sec", np.nan))}
             else: print("Warning: Could not determine best balanced query technique.")
        except Exception as e_rec3: print(f"Error generating best balance query recommendation: {e_rec3}")
    try: # Save recommendations
        with open(os.path.join(query_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save query processing recommendations: {e_json}")
# --- End analyze_query_processing_results ---


# --- analyze_reranking_results Function (Verbatim from paste-3.txt) ---
def analyze_reranking_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from reranking experiments"""
    # [ ... Function code exactly as provided in paste-3.txt ... ]
    reranking_df = df[df["experiment"] == "reranking"].copy()
    if reranking_df.empty: print("No reranking experiment results found"); return
    print("Analyzing reranking experiment results...")
    reranking_dir = os.path.join(output_dir, "reranking_analysis"); os.makedirs(reranking_dir, exist_ok=True)
    try: # Plotting
        plot_metrics_by_category(reranking_df, x_col="reranking_method", title="Retrieval Performance by Reranking Method", output_file=os.path.join(reranking_dir, "reranking_method_comparison.png"))
        if "reranking_time_sec" in reranking_df.columns:
             plt.figure(figsize=(10, 6)); sns.barplot(data=reranking_df, x="reranking_method", y="reranking_time_sec")
             plt.title("Reranking Time by Method"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(os.path.join(reranking_dir, "reranking_time_comparison.png")); plt.close()
        if "metric_mrr" in reranking_df.columns and "reranking_time_sec" in reranking_df.columns:
            baseline_df = reranking_df[reranking_df["reranking_method"] == "no_reranking"]
            if not baseline_df.empty:
                baseline_score = baseline_df["metric_mrr"].mean()
                if pd.notna(baseline_score) and baseline_score != 0: # Ensure valid baseline
                     reranking_only = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
                     if not reranking_only.empty:
                         reranking_only["improvement"] = reranking_only["metric_mrr"] - baseline_score
                         reranking_only["relative_improvement"] = reranking_only["improvement"] / abs(baseline_score) # Use abs
                         valid_time_df = reranking_only[(reranking_only["reranking_time_sec"] > 0) & (reranking_only["reranking_time_sec"].notna())]
                         if not valid_time_df.empty and not valid_time_df["relative_improvement"].isna().all():
                              plt.figure(figsize=(10, 6)); sns.scatterplot(data=valid_time_df, x="reranking_time_sec", y="relative_improvement", hue="reranking_method", s=100, alpha=0.7)
                              plt.title("Reranking Improvement vs. Time Cost"); plt.xlabel("Reranking Time (s)"); plt.ylabel("Relative MRR Improvement"); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
                              plt.savefig(os.path.join(reranking_dir, "reranking_improvement_vs_time.png")); plt.close()
                         else: print("Warning: Cannot plot reranking improvement vs time - data invalid.")
                else: print("Warning: Cannot calculate reranking improvement - baseline score invalid.")
    except Exception as e_plot: print(f"Warning: Failed during reranking plot generation: {e_plot}")
    try: # Summary
        group_cols = ["reranking_method", "description"]; metric_cols = [col for col in reranking_df.columns if col.startswith("metric_")]; efficiency_cols = []
        if "reranking_time_sec" in reranking_df.columns and not reranking_df["reranking_time_sec"].isna().all(): efficiency_cols.append("reranking_time_sec")
        if "queries_per_second" in reranking_df.columns and not reranking_df["queries_per_second"].isna().all(): efficiency_cols.append("queries_per_second")
        elif "reranking_time_sec" in efficiency_cols: reranking_df["qps_calc"] = 1.0 / reranking_df["reranking_time_sec"]; efficiency_cols.append("qps_calc")
        valid_group_cols = [col for col in group_cols if col in reranking_df.columns]; valid_metric_cols = [col for col in metric_cols if col in reranking_df.columns]; valid_efficiency_cols = [col for col in efficiency_cols if col in reranking_df.columns]
        if valid_group_cols and (valid_metric_cols or valid_efficiency_cols):
             summary = reranking_df.groupby(valid_group_cols)[valid_metric_cols + valid_efficiency_cols].mean().reset_index()
             summary.to_csv(os.path.join(reranking_dir, "reranking_summary.csv"), index=False)
        else: print("Warning: Could not create reranking summary - columns missing.")
    except Exception as e_summary: print(f"Warning: Failed to create/save reranking summary: {e_summary}")
    print("\nBest configurations for reranking experiment:"); metric_columns_to_check = [col for col in reranking_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in reranking_df.columns and not reranking_df[metric].isna().all():
                best_idx = reranking_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = reranking_df.loc[best_idx]
                    print(f"Best {metric}:"); print(f" Method: {best_config.get('reranking_method', 'N/A')} ({best_config.get('description', 'N/A')})"); print(f" Score: {best_config.get(metric, np.nan):.4f}")
                    if "reranking_time_sec" in best_config and pd.notna(best_config['reranking_time_sec']): print(f" Reranking time: {best_config['reranking_time_sec']:.2f} seconds")
                    print()
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; mrr_col = None # Recommendations
    if "metric_mrr" in reranking_df.columns and not reranking_df["metric_mrr"].isna().all(): mrr_col = "metric_mrr"
    else: potential_metrics = [col for col in metric_columns_to_check if col in reranking_df.columns and not reranking_df[col].isna().all()]; mrr_col = potential_metrics[0] if potential_metrics else None
    if mrr_col: # Best performance
         try:
             best_idx = reranking_df[mrr_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = reranking_df.loc[best_idx]; rec = {"reranking_method": best_config.get("reranking_method"), "description": best_config.get("description"), "metric": mrr_col, "score": float(best_config.get(mrr_col, np.nan))}
                 if "reranking_time_sec" in best_config and pd.notna(best_config["reranking_time_sec"]): rec["reranking_time_sec"] = float(best_config["reranking_time_sec"])
                 recommendations["best_performance"] = rec
             else: print(f"Warning: Could not determine best overall reranking method based on {mrr_col}.")
         except Exception as e_rec1: print(f"Error generating best performance reranking recommendation: {e_rec1}")
    if "no_reranking" in reranking_df["reranking_method"].values and mrr_col: # Improvements
        improvements = {}
        try:
            baseline_score_series = reranking_df[reranking_df["reranking_method"] == "no_reranking"][mrr_col]
            if not baseline_score_series.empty and not baseline_score_series.isna().all():
                no_reranking_score = baseline_score_series.mean()
                for method in reranking_df["reranking_method"].dropna().unique():
                    if method == "no_reranking": continue
                    method_score_series = reranking_df[reranking_df["reranking_method"] == method][mrr_col]
                    if not method_score_series.empty and not method_score_series.isna().all():
                        method_score = method_score_series.mean()
                        if pd.notna(method_score) and pd.notna(no_reranking_score) and no_reranking_score != 0:
                            rel_improvement = (method_score - no_reranking_score) / abs(no_reranking_score)
                            imp_rec = {"absolute_improvement": float(method_score - no_reranking_score), "relative_improvement": float(rel_improvement), "score": float(method_score)}
                            if "reranking_time_sec" in reranking_df.columns:
                                method_time_series = reranking_df[reranking_df["reranking_method"] == method]["reranking_time_sec"]
                                if not method_time_series.empty and not method_time_series.isna().all(): imp_rec["reranking_time_sec"] = float(method_time_series.mean())
                            improvements[str(method)] = imp_rec
            else: print("Warning: Cannot calculate reranking improvements - baseline score missing/invalid.")
        except Exception as e_imp: print(f"Error calculating reranking improvements: {e_imp}")
        recommendations["improvements_over_baseline"] = improvements
    if "reranking_time_sec" in reranking_df.columns and mrr_col and "no_reranking" in reranking_df["reranking_method"].values: # Most Efficient
        try:
            reranking_only = reranking_df[reranking_df["reranking_method"] != "no_reranking"].copy()
            baseline_score_series = reranking_df[reranking_df["reranking_method"] == "no_reranking"][mrr_col]
            if not reranking_only.empty and not baseline_score_series.empty and not baseline_score_series.isna().all():
                no_reranking_score = baseline_score_series.mean()
                valid_rerank_df = reranking_only[(reranking_only["reranking_time_sec"] > 0) & (reranking_only["reranking_time_sec"].notna()) & (reranking_only[mrr_col].notna())]
                if not valid_rerank_df.empty:
                     valid_rerank_df["improvement"] = valid_rerank_df[mrr_col] - no_reranking_score
                     valid_rerank_df["imp_per_sec"] = valid_rerank_df["improvement"] / valid_rerank_df["reranking_time_sec"]
                     if not valid_rerank_df["imp_per_sec"].isna().all():
                          efficient_idx = valid_rerank_df["imp_per_sec"].idxmax()
                          if not pd.isna(efficient_idx):
                              efficient_config = valid_rerank_df.loc[efficient_idx]
                              recommendations["most_efficient"] = {"reranking_method": efficient_config.get("reranking_method"), "description": efficient_config.get("description"), "improvement_per_second": float(efficient_config.get("imp_per_sec", np.nan)), "score": float(efficient_config.get(mrr_col, np.nan)), "reranking_time_sec": float(efficient_config.get("reranking_time_sec", np.nan))}
                          else: print("Warning: Could not determine most efficient reranker (idxmax failed).")
                     else: print("Warning: Could not determine most efficient reranker (all imp_per_sec NaN).")
                else: print("Warning: Cannot calculate reranking efficiency - data invalid.")
            else: print("Warning: Cannot calculate reranking efficiency - baseline invalid.")
        except Exception as e_eff: print(f"Error generating most efficient reranking recommendation: {e_eff}")
    try: # Save recommendations
        with open(os.path.join(reranking_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save reranking recommendations: {e_json}")
# --- End analyze_reranking_results ---


# --- analyze_generation_results Function (Verbatim from paste-3.txt) ---
def analyze_generation_results(df: pd.DataFrame, output_dir: str):
    """Analyze results from generation experiments"""
    # [ ... Function code exactly as provided in paste-3.txt ... ]
    generation_df = df[df["experiment"] == "generation"].copy()
    if generation_df.empty: print("No generation experiment results found"); return
    print("Analyzing generation experiment results...")
    generation_dir = os.path.join(output_dir, "generation_analysis"); os.makedirs(generation_dir, exist_ok=True)
    try: # Plotting
        if "prompt_template" in generation_df.columns:
             plot_metrics_by_category(generation_df, x_col="prompt_template", title="Generation Performance by Prompt Template", output_file=os.path.join(generation_dir, "prompt_template_comparison.png"))
        if "retrieval_config" in generation_df.columns:
             plot_metrics_by_category(generation_df, x_col="retrieval_config", title="Generation Performance by Retrieval Configuration", output_file=os.path.join(generation_dir, "retrieval_config_comparison.png"))
        if ("prompt_template" in generation_df.columns and "retrieval_config" in generation_df.columns and "metric_answer_precision" in generation_df.columns and not generation_df["metric_answer_precision"].isna().all()):
             try:
                 pivot_df = generation_df.pivot_table(index="prompt_template", columns="retrieval_config", values="metric_answer_precision", aggfunc="mean")
                 plt.figure(figsize=(10, 8)); sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
                 plt.title("Answer Precision by Prompt Template and Retrieval Configuration"); plt.tight_layout()
                 plt.savefig(os.path.join(generation_dir, "prompt_retrieval_heatmap.png")); plt.close()
             except Exception as e_heat: print(f"Warning: Failed to generate heatmap: {e_heat}")
    except Exception as e_plot: print(f"Warning: Failed during generation plot generation: {e_plot}")
    try: # Summary
        group_cols = []; metric_cols = [col for col in generation_df.columns if col.startswith("metric_")]; efficiency_cols = []
        if "prompt_template" in generation_df.columns: group_cols.append("prompt_template")
        if "prompt_description" in generation_df.columns: group_cols.append("prompt_description")
        if "retrieval_config" in generation_df.columns: group_cols.append("retrieval_config")
        if "retrieval_description" in generation_df.columns: group_cols.append("retrieval_description")
        if "generation_time_sec" in generation_df.columns and not generation_df["generation_time_sec"].isna().all(): efficiency_cols.append("generation_time_sec")
        if "queries_per_second" in generation_df.columns and not generation_df["queries_per_second"].isna().all(): efficiency_cols.append("queries_per_second")
        elif "generation_time_sec" in efficiency_cols: generation_df["qps_calc"] = 1.0 / generation_df["generation_time_sec"]; efficiency_cols.append("qps_calc")
        valid_group_cols = [col for col in group_cols if col in generation_df.columns]; valid_metric_cols = [col for col in metric_cols if col in generation_df.columns]; valid_efficiency_cols = [col for col in efficiency_cols if col in generation_df.columns]
        if valid_group_cols and (valid_metric_cols or valid_efficiency_cols):
             summary = generation_df.groupby(valid_group_cols)[valid_metric_cols + valid_efficiency_cols].mean().reset_index()
             summary.to_csv(os.path.join(generation_dir, "generation_summary.csv"), index=False)
        else: print("Warning: Could not create generation summary - columns missing.")
    except Exception as e_summary: print(f"Warning: Failed to create/save generation summary: {e_summary}")
    print("\nBest configurations for generation experiment:"); metric_columns_to_check = [col for col in generation_df.columns if col.startswith("metric_")]
    for metric in metric_columns_to_check: # Find Best Loop
        try:
            if metric in generation_df.columns and not generation_df[metric].isna().all():
                best_idx = generation_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = generation_df.loc[best_idx]
                    print(f"Best {metric}:")
                    if "prompt_template" in best_config: print(f" Prompt: {best_config.get('prompt_template', 'N/A')} ({best_config.get('prompt_description', 'N/A')})")
                    if "retrieval_config" in best_config: print(f" Retrieval: {best_config.get('retrieval_config', 'N/A')} ({best_config.get('retrieval_description', 'N/A')})")
                    print(f" Score: {best_config.get(metric, np.nan):.4f}")
                    if "generation_time_sec" in best_config and pd.notna(best_config['generation_time_sec']): print(f" Generation time: {best_config['generation_time_sec']:.2f} seconds")
                    print()
                else: print(f"Could not determine best configuration for {metric}: No valid scores found.\n")
            else: print(f"Could not determine best configuration for {metric}: Column missing or all NaN.\n")
        except Exception as e_best: print(f"Error finding best configuration for {metric}: {e_best}\n")
    recommendations = {}; answer_precision_col = None # Recommendations
    if "metric_answer_precision" in generation_df.columns and not generation_df["metric_answer_precision"].isna().all(): answer_precision_col = "metric_answer_precision"
    else: potential_metrics = [col for col in metric_columns_to_check if col in generation_df.columns and not generation_df[col].isna().all()]; answer_precision_col = potential_metrics[0] if potential_metrics else None
    if answer_precision_col: # Best performance
         try:
             best_idx = generation_df[answer_precision_col].idxmax()
             if not pd.isna(best_idx):
                 best_config = generation_df.loc[best_idx]; rec = {"metric": answer_precision_col, "score": float(best_config.get(answer_precision_col, np.nan))}
                 if "prompt_template" in best_config: rec["prompt_template"] = best_config.get("prompt_template"); rec["prompt_description"] = best_config.get("prompt_description")
                 if "retrieval_config" in best_config: rec["retrieval_config"] = best_config.get("retrieval_config"); rec["retrieval_description"] = best_config.get("retrieval_description")
                 if "generation_time_sec" in best_config and pd.notna(best_config["generation_time_sec"]): rec["generation_time_sec"] = float(best_config["generation_time_sec"])
                 recommendations["best_performance"] = rec
             else: print(f"Warning: Could not determine best overall generation config based on {answer_precision_col}.")
         except Exception as e_rec1: print(f"Error generating best performance generation recommendation: {e_rec1}")
    metric_recommendations = {}; important_metrics = ["metric_answer_precision", "metric_faithfulness", "metric_citation_rate", "metric_answer_correctness"] # By Metric
    for metric in important_metrics:
        if metric in generation_df.columns and not generation_df[metric].isna().all():
            try:
                best_idx = generation_df[metric].idxmax()
                if not pd.isna(best_idx):
                    best_config = generation_df.loc[best_idx]; metric_name = metric.replace("metric_", ""); rec = {"score": float(best_config.get(metric, np.nan))}
                    if "prompt_template" in best_config: rec["prompt_template"] = best_config.get("prompt_template")
                    if "retrieval_config" in best_config: rec["retrieval_config"] = best_config.get("retrieval_config")
                    metric_recommendations[metric_name] = rec
            except Exception as e_rec_metric: print(f"Error generating recommendation for metric {metric}: {e_rec_metric}")
    recommendations["by_metric"] = metric_recommendations
    try: # Save recommendations
        with open(os.path.join(generation_dir, "recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json: print(f"Warning: Failed to save generation recommendations: {e_json}")
# --- End analyze_generation_results ---


# --- generate_comparative_analysis Function (Verbatim from paste-3.txt) ---
def generate_comparative_analysis(df: pd.DataFrame, output_dir: str):
    """Generate overall comparative analysis across all experiments"""
    # [ ... Function code exactly as provided in paste-3.txt ... ]
    comparative_dir = os.path.join(output_dir, "comparative_analysis"); os.makedirs(comparative_dir, exist_ok=True)
    print("Generating comparative analysis across experiments...")
    experiments = df["experiment"].unique(); recommendations = {}
    for experiment in experiments: # Load recs
        exp_dir = os.path.join(output_dir, f"{experiment}_analysis"); rec_file = os.path.join(exp_dir, "recommendations.json")
        if os.path.exists(rec_file):
             try:
                 with open(rec_file, "r") as f: recommendations[experiment] = json.load(f)
             except Exception as e_load_rec: print(f"Warning: Failed to load recommendations for {experiment}: {e_load_rec}")
    try: # Save combined recs
        with open(os.path.join(comparative_dir, "all_recommendations.json"), "w") as f: json.dump(recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_json_all: print(f"Warning: Failed to save combined recommendations: {e_json_all}")
    metric_columns = [col for col in df.columns if col.startswith("metric_")]; common_metrics = [] # Common metrics plot
    for metric in metric_columns:
        present_in_all = True
        for experiment in experiments:
             if metric not in df[df["experiment"] == experiment].columns: present_in_all = False; break
        if present_in_all: common_metrics.append(metric)
    for metric in common_metrics:
        best_scores = []
        for experiment in experiments:
            exp_df = df[df["experiment"] == experiment]
            if not exp_df.empty and metric in exp_df.columns and not exp_df[metric].isna().all():
                 best_score = exp_df[metric].max(); best_scores.append({"Experiment": experiment, "Best Score": best_score})
        if best_scores:
            try:
                best_df = pd.DataFrame(best_scores); plt.figure(figsize=(10, 6)); sns.barplot(data=best_df, x="Experiment", y="Best Score")
                plt.title(f"Best {metric.replace('metric_', '')} Across Experiments"); plt.xticks(rotation=45); plt.tight_layout()
                plt.savefig(os.path.join(comparative_dir, f"{metric}_comparison.png")); plt.close()
            except Exception as e_comp_plot: print(f"Warning: Failed to plot common metric {metric}: {e_comp_plot}")
    pipeline_recommendations = { # Pipeline recs
        "best_chunking": recommendations.get("chunking", {}).get("best_overall", {}), "best_embedding": recommendations.get("embedding", {}).get("best_performance", {}),
        "best_retrieval": recommendations.get("retrieval", {}).get("best_performance", {}), "best_reranking": recommendations.get("reranking", {}).get("best_performance", {}),
        "best_generation": recommendations.get("generation", {}).get("best_performance", {})}
    try: # Save pipeline recs
        with open(os.path.join(comparative_dir, "pipeline_recommendations.json"), "w") as f: json.dump(pipeline_recommendations, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    except Exception as e_pipe_json: print(f"Warning: Failed to save pipeline recommendations: {e_pipe_json}")
    performance_impact = {}; # Performance impact analysis
    if "chunking" in recommendations and "best_overall" in recommendations["chunking"]:
        try:
            best_chunking = recommendations["chunking"]["best_overall"]; baseline_chunking = "fixed"; chunking_df = df[df["experiment"] == "chunking"]
            if not chunking_df.empty and "metric_mrr" in chunking_df.columns:
                baseline_score_series = chunking_df[(chunking_df["chunking_strategy"] == baseline_chunking) & (chunking_df["retrieval_method"] == "vector")]["metric_mrr"]
                best_score_series = chunking_df[(chunking_df["chunking_strategy"] == best_chunking.get("chunking_strategy", "")) & (chunking_df["retrieval_method"] == best_chunking.get("retrieval_method", ""))]["metric_mrr"]
                if not baseline_score_series.empty and not best_score_series.empty and not baseline_score_series.isna().all() and not best_score_series.isna().all():
                    baseline_score = baseline_score_series.mean(); best_score = best_score_series.mean()
                    if baseline_score != 0: performance_impact["chunking"] = {"baseline": baseline_chunking, "best": best_chunking.get("chunking_strategy", ""), "absolute_improvement": float(best_score - baseline_score), "relative_improvement": float((best_score - baseline_score) / abs(baseline_score))}
        except Exception as e_imp_c: print(f"Warning: Error calculating chunking impact: {e_imp_c}")
    if "retrieval" in recommendations and "best_performance" in recommendations["retrieval"]:
        try:
            best_retrieval = recommendations["retrieval"]["best_performance"]; baseline_retrieval_type = "vector"; baseline_expansion = "none"; retrieval_df = df[df["experiment"] == "retrieval"]
            if not retrieval_df.empty and "metric_mrr" in retrieval_df.columns:
                baseline_score_series = retrieval_df[(retrieval_df["retrieval_type"] == baseline_retrieval_type) & (retrieval_df["query_expansion"].astype(str) == baseline_expansion)]["metric_mrr"]
                best_score_series = retrieval_df[(retrieval_df["retrieval_method"] == best_retrieval.get("retrieval_method", ""))]["metric_mrr"]
                if not baseline_score_series.empty and not best_score_series.empty and not baseline_score_series.isna().all() and not best_score_series.isna().all():
                     baseline_score = baseline_score_series.mean(); best_score = best_score_series.mean()
                     if baseline_score != 0: performance_impact["retrieval"] = {"baseline": baseline_retrieval_type, "best": best_retrieval.get("retrieval_method", ""), "absolute_improvement": float(best_score - baseline_score), "relative_improvement": float((best_score - baseline_score) / abs(baseline_score))}
        except Exception as e_imp_r: print(f"Warning: Error calculating retrieval impact: {e_imp_r}")
    if "reranking" in recommendations and "best_performance" in recommendations["reranking"]:
        try:
            best_reranking = recommendations["reranking"]["best_performance"]; baseline_reranking = "no_reranking"; reranking_df = df[df["experiment"] == "reranking"]
            if not reranking_df.empty and "metric_mrr" in reranking_df.columns:
                baseline_score_series = reranking_df[(reranking_df["reranking_method"] == baseline_reranking)]["metric_mrr"]
                best_score_series = reranking_df[(reranking_df["reranking_method"] == best_reranking.get("reranking_method", ""))]["metric_mrr"]
                if not baseline_score_series.empty and not best_score_series.empty and not baseline_score_series.isna().all() and not best_score_series.isna().all():
                    baseline_score = baseline_score_series.mean(); best_score = best_score_series.mean()
                    if baseline_score != 0: performance_impact["reranking"] = {"baseline": baseline_reranking, "best": best_reranking.get("reranking_method", ""), "absolute_improvement": float(best_score - baseline_score), "relative_improvement": float((best_score - baseline_score) / abs(baseline_score))}
        except Exception as e_imp_rr: print(f"Warning: Error calculating reranking impact: {e_imp_rr}")
    if performance_impact: # Save and plot impact
        try:
            with open(os.path.join(comparative_dir, "performance_impact.json"), "w") as f: json.dump(performance_impact, f, indent=2, default=lambda x: None if pd.isna(x) else x)
            impact_data = [{"Component": comp, "Rel Imp (%)": imp.get("relative_improvement", 0) * 100} for comp, imp in performance_impact.items()]
            if impact_data:
                impact_df = pd.DataFrame(impact_data); plt.figure(figsize=(10, 6)); sns.barplot(data=impact_df, x="Component", y="Rel Imp (%)")
                plt.title("Relative Performance Improvement by RAG Component"); plt.ylabel("Relative Improvement (%)"); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(comparative_dir, "component_impact.png")); plt.close()
        except Exception as e_impact_final: print(f"Warning: Error saving/plotting performance impact: {e_impact_final}")
# --- End generate_comparative_analysis ---


# --- plot_metrics_by_category Function (Verbatim from paste-3.txt) ---
def plot_metrics_by_category(df: pd.DataFrame, x_col: str, hue_col: Optional[str] = None, title: str = "Metrics Comparison", output_file: str = "metrics_plot.png"):
    """Plot metrics grouped by a category (Verbatim from paste-3.txt)"""
    # [ ... Function code exactly as provided in paste-3.txt ... ]
    metric_cols = [col for col in df.columns if col.startswith("metric_")]
    if not metric_cols: print("No metric columns found for plotting"); return
    # Ensure required columns exist before melting
    id_vars = [x_col] + ([hue_col] if hue_col and hue_col in df.columns else [])
    value_vars = [col for col in metric_cols if col in df.columns]
    if not value_vars or not all(col in df.columns for col in id_vars): print(f"Warning: Missing columns needed for plot '{title}'. Skipping."); return
    try: # Add try-except around plotting itself
        plot_data = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name="Metric", value_name="Score")
        plot_data["Metric"] = plot_data["Metric"].str.replace("metric_", "")
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=plot_data, x=x_col, y="Score", hue=hue_col if hue_col else "Metric")
        plt.title(title); plt.xticks(rotation=45)
        # Set legend title safely
        legend_title = hue_col if hue_col else "Metric"
        try: # Handle potential error if legend handle is None
             handles, labels = ax.get_legend_handles_labels()
             if handles: # Only show legend if there are items
                  ax.legend(handles=handles, labels=labels, title=legend_title)
             elif ax.get_legend() is not None: # Remove empty legend if it exists
                  ax.get_legend().remove()
        except AttributeError: pass # Ignore if legend methods fail
        plt.tight_layout(); plt.savefig(output_file); plt.close()
    except Exception as e_plot: print(f"Warning: Failed to generate plot '{title}': {e_plot}")
# --- End plot_metrics_by_category ---


# --- Keep if __name__ == "__main__": block EXACTLY as in paste-3.txt ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, help="Directory with experiment results") # Default handled by function
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save output")
    args = parser.parse_args()
    if args.results_dir: results_dirs = [args.results_dir]
    else: results_dirs = None # Pass None to use default "results"
    analyze_experiment_results(results_dirs=results_dirs, output_dir=args.output_dir)
# --- End Main ---

