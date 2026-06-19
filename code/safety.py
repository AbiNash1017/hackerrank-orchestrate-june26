"""
Safety and Pre-processing Layer
Detects prompt-injection attempts and sanitizes claim text before sending to the VLM.
"""
import re

# Common prompt injection indicators
INJECTION_KEYWORDS = [
    r"ignore (?:all )?instructions",
    r"ignore (?:previous|prior|past) (?:instructions|rules|guidelines|directives)",
    r"approve (?:the claim )?immediately",
    r"skip (?:manual )?review",
    r"mark (?:as )?supported",
    r"force (?:support|approval)",
    r"override (?:the )?(?:system|rules|verification|review)",
    r"bypass (?:the )?(?:verification|system|rules)",
    r"note (?:in |says |on )(?:photo|image|text|attached|note) (?:approved|approve|follow)",
    r"keep reopening (?:tickets|claims) until approved",
    r"you must approve",
    r"system message:",
    r"system prompt:",
]

def detect_prompt_injection(user_claim: str) -> bool:
    """
    Detects if the user claim contains prompt-injection attempts.
    """
    if not user_claim:
        return False
    
    claim_lower = user_claim.lower()
    
    for pattern in INJECTION_KEYWORDS:
        if re.search(pattern, claim_lower):
            return True
            
    return False

def sanitize_claim(user_claim: str) -> str:
    """
    Sanitizes user claim to prevent prompt injection.
    Wraps text and escapes brackets/quotes so the VLM treats it as raw untrusted string.
    """
    if not user_claim:
        return ""
    
    # Replace common injection attempt words with slightly altered forms if we want to be safe,
    # or simply escape JSON/quotes, or wrap the prompt in strong XML-like delimiters.
    # We will primarily wrap it in strong delimiters and let the system prompt handle it,
    # but we can also strip or flag obvious malicious phrases.
    sanitized = user_claim.replace('"', '\\"').replace('\n', ' ').strip()
    return sanitized
