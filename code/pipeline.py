"""
Pipeline Orchestrator Layer
Loads data, maps evidence requirements, executes the safety and VLM layer, and collects results.
"""
import os
import pandas as pd
from typing import Dict, Any, List
from tqdm import tqdm

from safety import detect_prompt_injection, sanitize_claim
from prompt_builder import build_system_prompt, build_user_prompt
from vlm_client import GeminiVLMClient
from postprocessor import postprocess_result

def load_user_history(path: str) -> Dict[str, Dict[str, Any]]:
    """Loads user history from CSV and returns a mapping from user_id to details."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"User history file not found at: {path}")
    df = pd.read_csv(path)
    # Convert dataframe to a dict of dicts keyed by user_id
    history = {}
    for _, row in df.iterrows():
        user_id = str(row["user_id"]).strip()
        history[user_id] = {
            "past_claim_count": row.get("past_claim_count", 0),
            "accept_claim": row.get("accept_claim", 0),
            "manual_review_claim": row.get("manual_review_claim", 0),
            "rejected_claim": row.get("rejected_claim", 0),
            "last_90_days_claim_count": row.get("last_90_days_claim_count", 0),
            "history_flags": str(row.get("history_flags", "none")).strip(),
            "history_summary": str(row.get("history_summary", "")).strip(),
        }
    return history

def load_evidence_requirements(path: str) -> List[Dict[str, Any]]:
    """Loads evidence requirements from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evidence requirements file not found at: {path}")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def get_applicable_requirements(
    claim_object: str,
    user_claim: str,
    num_images: int,
    requirements: List[Dict[str, Any]]
) -> str:
    """
    Intelligently maps which evidence requirements apply to the claim based on the
    object type, the user's claim text, and number of images.
    """
    applicable = []
    claim_lower = user_claim.lower()
    obj = claim_object.lower()
    
    for req in requirements:
        req_id = req["requirement_id"]
        req_obj = req["claim_object"].lower()
        applies_to = req["applies_to"].lower()
        min_evidence = req["minimum_image_evidence"]
        
        # 1. General rules that always apply
        if req_id == "REQ_GENERAL_OBJECT_PART" or req_id == "REQ_REVIEW_TRUST":
            applicable.append(f"- {req_id}: {min_evidence}")
            continue
            
        if req_id == "REQ_GENERAL_MULTI_IMAGE" and num_images > 1:
            applicable.append(f"- {req_id}: {min_evidence}")
            continue
            
        # 2. Match based on object type and keywords
        if req_obj == obj or req_obj == "all":
            # Car rules
            if obj == "car":
                if req_id == "REQ_CAR_BODY_PANEL" and ("dent" in claim_lower or "scratch" in claim_lower or "scrape" in claim_lower or "hail" in claim_lower):
                    applicable.append(f"- {req_id}: {min_evidence}")
                elif req_id == "REQ_CAR_GLASS_LIGHT_MIRROR" and any(w in claim_lower for w in ["crack", "broken", "mirror", "light", "glass", "windshield", "headlight", "taillight"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                elif req_id == "REQ_CAR_IDENTITY_OR_SIDE" and (num_images > 1 or any(w in claim_lower for w in ["side", "identity", "different", "angle", "view"])):
                    applicable.append(f"- {req_id}: {min_evidence}")
            
            # Laptop rules
            elif obj == "laptop":
                if req_id == "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD" and any(w in claim_lower for w in ["screen", "keyboard", "trackpad", "key", "display", "spill", "stain"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                elif req_id == "REQ_LAPTOP_BODY_HINGE_PORT" and any(w in claim_lower for w in ["hinge", "lid", "corner", "body", "port", "base", "dent"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                    
            # Package rules
            elif obj == "package":
                if req_id == "REQ_PACKAGE_EXTERIOR" and any(w in claim_lower for w in ["crushed", "torn", "seal", "open", "box", "flap"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                elif req_id == "REQ_PACKAGE_LABEL_OR_STAIN" and any(w in claim_lower for w in ["water", "stain", "label", "wet"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                elif req_id == "REQ_PACKAGE_CONTENTS" and any(w in claim_lower for w in ["contents", "item", "missing", "inside", "empty"]):
                    applicable.append(f"- {req_id}: {min_evidence}")
                    
    # If nothing specific was matched, fallback to all rules for that object type
    if len(applicable) <= 3:  # Only has the 2 general ones + maybe multi_image
        for req in requirements:
            req_id = req["requirement_id"]
            req_obj = req["claim_object"].lower()
            min_evidence = req["minimum_image_evidence"]
            if req_obj == obj and f"- {req_id}: {min_evidence}" not in applicable:
                applicable.append(f"- {req_id}: {min_evidence}")
                
    return "\n".join(applicable)

def resolve_image_paths(paths_str: str, dataset_root: str) -> List[str]:
    """Resolves relative image paths to absolute paths, handling subfolders."""
    resolved = []
    # Split by semicolon
    parts = [p.strip() for p in paths_str.split(";") if p.strip()]
    for p in parts:
        # Check standard relative paths (e.g. images/test/case_001/img_1.jpg)
        # Try relative to dataset_root
        trial1 = os.path.join(dataset_root, p)
        # Try relative to repo root (which is parent of dataset_root if dataset_root is hackerearth/dataset)
        repo_root = os.path.dirname(dataset_root)
        trial2 = os.path.join(repo_root, p)
        # Try appending dataset prefix to path if it doesn't have it
        trial3 = os.path.join(dataset_root, "dataset", p)
        
        if os.path.exists(trial1):
            resolved.append(trial1)
        elif os.path.exists(trial2):
            resolved.append(trial2)
        elif os.path.exists(trial3):
            resolved.append(trial3)
        else:
            # Check if dataset_root itself is the repo root and dataset/images/... is standard
            trial4 = os.path.join(dataset_root, "dataset", p)
            if os.path.exists(trial4):
                resolved.append(trial4)
            else:
                # If still not found, return trial1 path so it fails with clean FileNotFoundError later
                resolved.append(trial1)
    return resolved

def process_claim_row(
    row: Dict[str, Any],
    history_lookup: Dict[str, Dict[str, Any]],
    requirements_lookup: List[Dict[str, Any]],
    vlm_client: GeminiVLMClient,
    dataset_root: str
) -> Dict[str, Any]:
    """
    Executes safety pre-processing, prompt construction, VLM query, and post-processing
    for a single row of the claims dataframe.
    """
    user_id = str(row["user_id"]).strip()
    image_paths_str = str(row["image_paths"]).strip()
    user_claim = str(row["user_claim"]).strip()
    claim_object = str(row["claim_object"]).strip()
    
    # 1. Pre-processing Safety Layer: Detect prompt injection
    text_instruction_present = detect_prompt_injection(user_claim)
    sanitized_claim = sanitize_claim(user_claim)
    
    # 2. Get User History Context
    user_hist = history_lookup.get(user_id, {
        "history_flags": "none",
        "history_summary": "No prior claim history."
    })
    history_flags = user_hist["history_flags"]
    history_summary = user_hist["history_summary"]
    
    # 3. Resolve image paths
    absolute_image_paths = resolve_image_paths(image_paths_str, dataset_root)
    num_images = len(absolute_image_paths)
    
    # 4. Map Applicable requirements
    applicable_reqs_text = get_applicable_requirements(
        claim_object, user_claim, num_images, requirements_lookup
    )
    
    # 5. Build Prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        claim_object=claim_object,
        user_claim=sanitized_claim,
        history_summary=history_summary,
        history_flags=history_flags,
        applicable_requirements=applicable_reqs_text
    )
    
    # 6. Determine thinking level
    # Enable medium/high thinking for adversarial or low-light/ambiguous text cases
    thinking_level = "none"
    if text_instruction_present:
        thinking_level = "high"
    elif "blurry" in user_claim.lower() or "dark" in user_claim.lower() or "not sure" in user_claim.lower():
        thinking_level = "medium"
        
    # 7. Execute VLM call
    try:
        raw_vlm_output = vlm_client.analyze_claim(
            image_paths=absolute_image_paths,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking_level=thinking_level
        )
    except Exception as e:
        # Fallback dictionary if VLM fails completely
        print(f"CRITICAL VLM Failure for user {user_id}: {str(e)}")
        raw_vlm_output = {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": f"VLM execution failed: {str(e)}",
            "risk_flags": ["manual_review_required"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": f"VLM execution failed: {str(e)}",
            "supporting_image_ids": [],
            "valid_image": True,
            "severity": "unknown",
            "text_instruction_present": text_instruction_present
        }
        
    # 8. Post-processing Validation Layer
    processed_output = postprocess_result(
        raw_result=raw_vlm_output,
        claim_object=claim_object,
        user_history_flags=history_flags,
        text_instruction_present=text_instruction_present
    )
    
    # 9. Merge with pass-through inputs to form full output row
    output_row = {
        "user_id": user_id,
        "image_paths": image_paths_str,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": processed_output["evidence_standard_met"],
        "evidence_standard_met_reason": processed_output["evidence_standard_met_reason"],
        "risk_flags": processed_output["risk_flags"],
        "issue_type": processed_output["issue_type"],
        "object_part": processed_output["object_part"],
        "claim_status": processed_output["claim_status"],
        "claim_status_justification": processed_output["claim_status_justification"],
        "supporting_image_ids": processed_output["supporting_image_ids"],
        "valid_image": processed_output["valid_image"],
        "severity": processed_output["severity"]
    }
    
    return output_row

def run_pipeline(claims_csv_path: str, output_csv_path: str, dataset_root: str):
    """Runs the full pipeline end-to-end on claims_csv_path and writes output_csv_path."""
    print(f"Loading datasets...")
    claims_df = pd.read_csv(claims_csv_path)
    
    user_history_path = os.path.join(dataset_root, "user_history.csv")
    evidence_req_path = os.path.join(dataset_root, "evidence_requirements.csv")
    
    history_lookup = load_user_history(user_history_path)
    requirements_lookup = load_evidence_requirements(evidence_req_path)
    
    vlm_client = GeminiVLMClient()
    
    print(f"Loaded {len(claims_df)} claims. Initiating Gemini pipeline...")
    
    results = []
    # Process sequentially with tqdm progress bar
    for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
        row_dict = row.to_dict()
        processed_row = process_claim_row(
            row=row_dict,
            history_lookup=history_lookup,
            requirements_lookup=requirements_lookup,
            vlm_client=vlm_client,
            dataset_root=dataset_root
        )
        results.append(processed_row)
        
    # Write to DataFrame
    output_cols = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity"
    ]
    output_df = pd.DataFrame(results, columns=output_cols)
    
    # Save output CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Pipeline complete! Output written to {output_csv_path}")
    print(f"Token stats: Input={vlm_client.total_input_tokens}, Output={vlm_client.total_output_tokens}")
    print(f"Estimated run cost: ${vlm_client.total_cost:.4f}")
