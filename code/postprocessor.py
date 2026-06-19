"""
Post-processing and Validation Layer
Enforces enum constraints, applies custom business rules, and formats output fields.
"""
from typing import Dict, Any, List

# Allowed Enum Lists
CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
]

SEVERITIES = ["none", "low", "medium", "high", "unknown"]

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
]

LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
]

PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
]

ALLOWED_RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
]

def clean_enum(val: str, allowed: List[str], default: str = "unknown") -> str:
    """Helper to clean and validate enum values."""
    if not val:
        return default
    val_clean = str(val).strip().lower()
    if val_clean in allowed:
        return val_clean
    # Try mapping variations (e.g. "front bumper" -> "front_bumper")
    val_mapped = val_clean.replace(" ", "_").replace("-", "_")
    if val_mapped in allowed:
        return val_mapped
    return default

def get_allowed_parts(claim_object: str) -> List[str]:
    """Returns the list of allowed parts based on the object type."""
    obj = str(claim_object).strip().lower()
    if obj == "car":
        return CAR_PARTS
    elif obj == "laptop":
        return LAPTOP_PARTS
    elif obj == "package":
        return PACKAGE_PARTS
    return ["unknown"]

def postprocess_result(
    raw_result: Dict[str, Any],
    claim_object: str,
    user_history_flags: str = "none",
    text_instruction_present: bool = False
) -> Dict[str, Any]:
    """
    Validates, corrects, and formats raw VLM output against constraints and business rules.
    """
    processed = {}
    
    # 1. Base boolean fields as strings ("true" / "false")
    valid_image_raw = raw_result.get("valid_image")
    if isinstance(valid_image_raw, bool):
        valid_image = valid_image_raw
    else:
        valid_image = str(valid_image_raw).strip().lower() == "true"
        
    evidence_standard_raw = raw_result.get("evidence_standard_met")
    if isinstance(evidence_standard_raw, bool):
        evidence_standard_met = evidence_standard_raw
    else:
        evidence_standard_met = str(evidence_standard_raw).strip().lower() == "true"
        
    # Rule Override: If valid_image is false, then evidence_standard_met MUST be false
    if not valid_image:
        evidence_standard_met = False
        
    processed["valid_image"] = "true" if valid_image else "false"
    processed["evidence_standard_met"] = "true" if evidence_standard_met else "false"
    
    # 2. Text reasons & justifications (strip and clean)
    processed["evidence_standard_met_reason"] = str(raw_result.get("evidence_standard_met_reason", "")).strip()
    processed["claim_status_justification"] = str(raw_result.get("claim_status_justification", "")).strip()
    
    # 3. Handle Enums (issue_type, severity, object_part, claim_status)
    processed["issue_type"] = clean_enum(raw_result.get("issue_type"), ISSUE_TYPES, "unknown")
    processed["severity"] = clean_enum(raw_result.get("severity"), SEVERITIES, "unknown")
    
    allowed_parts = get_allowed_parts(claim_object)
    processed["object_part"] = clean_enum(raw_result.get("object_part"), allowed_parts, "unknown")
    
    processed["claim_status"] = clean_enum(raw_result.get("claim_status"), CLAIM_STATUSES, "not_enough_information")
    
    # Rule Override: if evidence standard is not met and status is supported, demote to not_enough_information
    if not evidence_standard_met and processed["claim_status"] == "supported":
        processed["claim_status"] = "not_enough_information"
        
    # 4. Process risk_flags (ensure list format, add history & text instruction flags)
    raw_flags = raw_result.get("risk_flags", [])
    if isinstance(raw_flags, str):
        # In case the VLM returned it as a string
        if raw_flags.lower() in ["none", ""]:
            flags_list = []
        else:
            flags_list = [f.strip() for f in raw_flags.replace(";", ",").split(",") if f.strip()]
    elif isinstance(raw_flags, list):
        flags_list = [str(f).strip() for f in raw_flags]
    else:
        flags_list = []
        
    # Clean flags and keep only allowed ones
    cleaned_flags = []
    for f in flags_list:
        cf = clean_enum(f, ALLOWED_RISK_FLAGS, "")
        if cf and cf != "none":
            cleaned_flags.append(cf)
            
    # Overlay: user history flags
    if user_history_flags and user_history_flags.lower() != "none":
        hist_flags = [hf.strip().lower() for hff in user_history_flags.split(";") for hf in hff.split(",") if hf.strip()]
        for hf in hist_flags:
            if hf in ALLOWED_RISK_FLAGS:
                cleaned_flags.append(hf)
                
    # Overlay: text instruction present (from pre-processing safety layer or VLM)
    if text_instruction_present or raw_result.get("text_instruction_present") is True:
        cleaned_flags.append("text_instruction_present")
        
    # Remove duplicates while preserving order
    unique_flags = []
    for f in cleaned_flags:
        if f not in unique_flags:
            unique_flags.append(f)
            
    if not unique_flags:
        processed["risk_flags"] = "none"
    else:
        processed["risk_flags"] = ";".join(unique_flags)
        
    # 5. Process supporting_image_ids (semicolon-separated stems)
    raw_imgs = raw_result.get("supporting_image_ids", [])
    if isinstance(raw_imgs, str):
        if raw_imgs.lower() in ["none", ""]:
            img_list = []
        else:
            img_list = [i.strip() for i in raw_imgs.replace(";", ",").split(",") if i.strip()]
    elif isinstance(raw_imgs, list):
        img_list = [str(i).strip() for i in raw_imgs]
    else:
        img_list = []
        
    # Clean image stems (e.g. "img_1.jpg" -> "img_1")
    cleaned_imgs = []
    for img in img_list:
        img_clean = img.split(".")[0].strip()
        if img_clean and img_clean.lower() != "none":
            cleaned_imgs.append(img_clean)
            
    if not cleaned_imgs or processed["claim_status"] == "not_enough_information":
        processed["supporting_image_ids"] = "none"
    else:
        processed["supporting_image_ids"] = ";".join(cleaned_imgs)
        
    return processed
