"""
Evaluation Main Entry Point
Runs the VLM pipeline on dataset/sample_claims.csv and computes accuracy and quality metrics.
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure the parent and current directories are in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import load_user_history, load_evidence_requirements, process_claim_row
from vlm_client import GeminiVLMClient
from metrics import compute_all_metrics

def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    
    # 1. Paths
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(code_dir)
    dataset_root = os.path.join(repo_root, "dataset")
    sample_csv_path = os.path.join(dataset_root, "sample_claims.csv")
    
    print(f"============================================================")
    print(f"Evaluation Pipeline - Multi-Modal Evidence Review")
    print(f"============================================================")
    print(f"Sample claims path: {sample_csv_path}")
    print(f"Dataset root:       {dataset_root}")
    print(f"============================================================")
    
    if not os.path.exists(sample_csv_path):
        print(f"ERROR: Sample claims CSV not found at: {sample_csv_path}")
        sys.exit(1)
        
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment or .env file.")
        sys.exit(1)
        
    # 2. Load dependencies
    print("Loading lookup tables...")
    user_history_path = os.path.join(dataset_root, "user_history.csv")
    evidence_req_path = os.path.join(dataset_root, "evidence_requirements.csv")
    
    history_lookup = load_user_history(user_history_path)
    requirements_lookup = load_evidence_requirements(evidence_req_path)
    
    vlm_client = GeminiVLMClient()
    
    # Load sample ground truth
    sample_df = pd.read_csv(sample_csv_path)
    print(f"Loaded {len(sample_df)} sample claims.")
    
    # 3. Process claims
    predictions = []
    print("\nRunning evaluation on sample claims...")
    for idx, row in sample_df.iterrows():
        print(f"[{idx+1}/{len(sample_df)}] Processing user: {row['user_id']} ({row['claim_object']})...")
        row_dict = row.to_dict()
        pred_row = process_claim_row(
            row=row_dict,
            history_lookup=history_lookup,
            requirements_lookup=requirements_lookup,
            vlm_client=vlm_client,
            dataset_root=dataset_root
        )
        predictions.append(pred_row)
        
    predictions_df = pd.DataFrame(predictions)
    
    # 4. Compute metrics
    print("\nComputing metrics...")
    try:
        metrics = compute_all_metrics(predictions_df, sample_df)
        
        print("\nEvaluation Results:")
        print("------------------------------------------------------------")
        for metric, val in metrics.items():
            if metric == "aligned_rows":
                print(f"{metric:30s}: {int(val)}")
            else:
                print(f"{metric:30s}: {val:.2%}")
        print("------------------------------------------------------------")
        
        # Save metrics to local text file for reference
        metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_evaluation_results.txt")
        with open(metrics_file, "w") as f:
            f.write("Evaluation Results:\n")
            f.write("===================\n")
            for metric, val in metrics.items():
                if metric == "aligned_rows":
                    f.write(f"{metric}: {int(val)}\n")
                else:
                    f.write(f"{metric}: {val:.4f}\n")
        print(f"Saved summary metrics to: {metrics_file}")
        
    except Exception as e:
        print(f"ERROR computing metrics: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
