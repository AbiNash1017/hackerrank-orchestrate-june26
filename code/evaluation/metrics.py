"""
Evaluation Metrics helper library.
Computes accuracies, Jaccard similarities, and combined scores for system evaluation.
"""
from typing import Dict, List, Set, Any
import pandas as pd

def parse_set(value: Any) -> Set[str]:
    """Helper to parse semicolon-separated values into a set."""
    if pd.isna(value):
        return {"none"}
    val_str = str(value).strip().lower()
    if val_str in ["none", ""]:
        return {"none"}
    return {item.strip() for item in val_str.replace(";", ",").split(",") if item.strip()}

def compute_jaccard(pred: str, gold: str) -> float:
    """Computes Jaccard similarity between two semicolon-separated lists."""
    set_pred = parse_set(pred)
    set_gold = parse_set(gold)
    
    # If both are {"none"}, similarity is 1.0
    if set_pred == {"none"} and set_gold == {"none"}:
        return 1.0
        
    intersection = set_pred.intersection(set_gold)
    union = set_pred.union(set_gold)
    
    # If union is empty (should not happen with default {"none"}), return 1.0 if they match
    if not union:
        return 1.0 if set_pred == set_gold else 0.0
        
    return len(intersection) / len(union)

def compute_all_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes all classification and set-based metrics.
    Ensures dataframes are aligned on user_id and image_paths.
    """
    # Align dataframes on user_id (or case index) to ensure proper comparison
    # We will merge them to match rows accurately
    pred_df = predictions_df.copy()
    gold_df = ground_truth_df.copy()
    
    # Prefix columns to avoid collisions
    pred_df = pred_df.add_prefix("pred_")
    gold_df = gold_df.add_prefix("gold_")
    
    # Merge on user_id
    merged = pd.merge(
        pred_df, gold_df,
        left_on="pred_user_id",
        right_on="gold_user_id",
        how="inner"
    )
    
    if len(merged) == 0:
        raise ValueError("No matching rows found between predictions and ground truth.")
        
    total_rows = len(merged)
    
    # Classification Exact Match accuracies
    exact_match_fields = [
        ("evidence_standard_met", "evidence_standard_met"),
        ("valid_image", "valid_image"),
        ("severity", "severity"),
        ("issue_type", "issue_type"),
        ("object_part", "object_part"),
        ("claim_status", "claim_status")
    ]
    
    metrics = {"aligned_rows": float(total_rows)}
    
    for pred_col, gold_col in exact_match_fields:
        pred_vals = merged[f"pred_{pred_col}"].astype(str).str.strip().str.lower()
        gold_vals = merged[f"gold_{gold_col}"].astype(str).str.strip().str.lower()
        correct = (pred_vals == gold_vals).sum()
        metrics[f"{pred_col}_accuracy"] = correct / total_rows
        
    # Jaccard similarities
    jaccard_fields = [
        ("risk_flags", "risk_flags"),
        ("supporting_image_ids", "supporting_image_ids")
    ]
    
    for pred_col, gold_col in jaccard_fields:
        similarities = []
        for _, row in merged.iterrows():
            sim = compute_jaccard(row[f"pred_{pred_col}"], row[f"gold_{gold_col}"])
            similarities.append(sim)
        metrics[f"{pred_col}_jaccard"] = sum(similarities) / total_rows
        
    # Compute Weighted Overall Score
    # Weights sum to 1.0
    weights = {
        "claim_status_accuracy": 0.35,
        "evidence_standard_met_accuracy": 0.15,
        "valid_image_accuracy": 0.10,
        "issue_type_accuracy": 0.10,
        "object_part_accuracy": 0.10,
        "severity_accuracy": 0.10,
        "risk_flags_jaccard": 0.10
    }
    
    overall_score = 0.0
    for key, weight in weights.items():
        if key in metrics:
            overall_score += metrics[key] * weight
            
    metrics["overall_weighted_score"] = overall_score
    
    return metrics
