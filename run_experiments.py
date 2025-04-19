"""
End-to-End Experiment Runner

This script runs a complete set of RAG experiments in sequence,
collecting results for all components to facilitate research paper writing.
"""

import os
import sys
import argparse
import time
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("experiment_runner")

def run_experiment(experiment_module: str, args: List[str]) -> bool:
    """
    Run a specific experiment as a subprocess
    
    Args:
        experiment_module: Python module path to run
        args: Command line arguments to pass
    
    Returns:
        Success status (True if experiment ran successfully)
    """
    command = ["python", "-m", experiment_module] + args
    
    logger.info(f"Running experiment: {experiment_module}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        # Run the experiment as a subprocess
        start_time = time.time()
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Get output and error
        stdout, stderr = process.communicate()
        
        # Log output
        if stdout:
            for line in stdout.splitlines():
                logger.info(f"[{experiment_module}] {line}")
        
        # Check if there was an error
        if process.returncode != 0:
            logger.error(f"Experiment {experiment_module} failed with code {process.returncode}")
            if stderr:
                for line in stderr.splitlines():
                    logger.error(f"[{experiment_module}] {line}")
            return False
        
        # Log success
        elapsed_time = time.time() - start_time
        logger.info(f"Experiment {experiment_module} completed successfully in {elapsed_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment {experiment_module}: {str(e)}")
        return False

def run_all_experiments(
    corpus_file: str,
    output_dir: str,
    sample_size: int = 50,
    num_queries: int = 5,
    skip_modules: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Run all experiments in sequence
    
    Args:
        corpus_file: Path to corpus file
        output_dir: Directory to save results
        sample_size: Number of documents to use in experiments
        num_queries: Number of queries to evaluate
        skip_modules: List of experiment modules to skip
    
    Returns:
        Dictionary with experiment results
    """
    if skip_modules is None:
        skip_modules = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiments to run
    experiments = [
        {
            "module": "experiments.test_framework",
            "args": []
        },
        {
            "module": "experiments.chunking_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        },
        {
            "module": "experiments.embedding_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        },
        {
            "module": "experiments.retrieval_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        },
        {
            "module": "experiments.query_processing_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        },
        {
            "module": "experiments.reranking_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        },
        {
            "module": "experiments.generation_experiment",
            "args": [
                "--corpus", corpus_file,
                "--sample-size", str(sample_size),
                "--num-queries", str(num_queries),
                "--output-dir", output_dir
            ]
        }
    ]
    
    # Keep track of results
    results = {}
    
    # Run all experiments
    for experiment in experiments:
        module = experiment["module"]
        args = experiment["args"]
        
        # Skip if in skip_modules
        if module in skip_modules:
            logger.info(f"Skipping experiment: {module}")
            results[module] = "skipped"
            continue
        
        # Run the experiment
        success = run_experiment(module, args)
        results[module] = success
        
        # Break if an experiment fails and it's a dependency for later experiments
        if not success and module in ["experiments.test_framework", "experiments.chunking_experiment"]:
            logger.error(f"Critical experiment {module} failed. Stopping further experiments.")
            break
    
    # Run analysis
    if all(results.get(m, False) for m in ["experiments.chunking_experiment", "experiments.embedding_experiment"]):
        logger.info("Running result analysis...")
        
        analysis_args = ["--results-dir", output_dir, "--output-dir", os.path.join(output_dir, "figures")]
        analysis_success = run_experiment("analysis.analyze_results", analysis_args)
        results["analysis.analyze_results"] = analysis_success
        
        if analysis_success:
            logger.info("Generating paper figures...")
            figure_args = ["--results-dir", output_dir, "--output-dir", os.path.join(output_dir, "paper_figures")]
            figure_success = run_experiment("analysis.paper_figures", figure_args)
            results["analysis.paper_figures"] = figure_success
    
    # Save results
    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n--- Experiment Run Summary ---")
    for module, success in results.items():
        status = "SUCCESS" if success == True else "FAILED" if success == False else "SKIPPED"
        logger.info(f"{module}: {status}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run all RAG experiments")
    
    parser.add_argument("--corpus", type=str, required=True,
                      help="Path to corpus file (pickle format)")
    
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save results")
    
    parser.add_argument("--sample-size", type=int, default=50,
                      help="Number of documents to use in experiments")
    
    parser.add_argument("--num-queries", type=int, default=5,
                      help="Number of queries to evaluate")
    
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                      help="List of experiment modules to skip")
    
    args = parser.parse_args()
    
    # Validate corpus file
    if not os.path.exists(args.corpus):
        logger.error(f"Corpus file not found: {args.corpus}")
        return 1
    
    # Run experiments
    try:
        results = run_all_experiments(
            corpus_file=args.corpus,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            num_queries=args.num_queries,
            skip_modules=args.skip
        )
        
        # Determine exit code based on success
        failed_experiments = [m for m, s in results.items() if s == False]
        if failed_experiments:
            logger.warning(f"Some experiments failed: {', '.join(failed_experiments)}")
            return 1
        else:
            logger.info("All experiments completed successfully!")
            return 0
        
    except Exception as e:
        logger.error(f"Error in experiment run: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())