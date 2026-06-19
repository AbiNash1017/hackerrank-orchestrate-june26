"""
Main Terminal Entry Point
Parses command-line arguments and triggers the processing pipeline on the test dataset.
"""
import os
import argparse
import sys
from dotenv import load_dotenv

# Ensure the current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="HackerRank Orchestrate - Multi-Modal Evidence Review Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("dataset", "claims.csv"),
        help="Path to the input claims CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Path to save the output predictions CSV file"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Path to the dataset directory (containing user_history, evidence_requirements, etc.)"
    )
    
    args = parser.parse_args()
    
    # 1. Resolve paths
    # If paths are relative, resolve them from current working directory
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    dataset_root = os.path.abspath(args.dataset_root)
    
    print(f"============================================================")
    print(f"HackerRank Orchestrate — Multi-Modal Evidence Review")
    print(f"============================================================")
    print(f"Input path:    {input_path}")
    print(f"Output path:   {output_path}")
    print(f"Dataset root:  {dataset_root}")
    print(f"============================================================")
    
    # 2. Check prerequisites
    if not os.path.exists(input_path):
        print(f"ERROR: Input claims file not found at: {input_path}")
        sys.exit(1)
        
    if not os.path.exists(dataset_root):
        print(f"ERROR: Dataset directory not found at: {dataset_root}")
        sys.exit(1)
        
    # Check for .env file or GEMINI_API_KEY
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment or .env file.")
        print("Please create a 'code/.env' file containing GOOGLE_API_KEY=your_key")
        sys.exit(1)
        
    # 3. Run Pipeline
    try:
        run_pipeline(
            claims_csv_path=input_path,
            output_csv_path=output_path,
            dataset_root=dataset_root
        )
    except Exception as e:
        print(f"CRITICAL: Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
