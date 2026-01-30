"""
Query sanitization for agent input: trimming, normalization, malformed input handling,
Blocked requests are counted as blocked in metrics.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


MAX_INPUT_LENGTH = 8000

HARMFUL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directives)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"ignore\s+the\s+above\s+instructions",
    r"ignore\s+any\s+(previous|prior)\s+instructions",
    r"do\s+not\s+follow\s+(any\s+)?(previous|above)\s+instructions",
    r"override\s+(your\s+)?(system|instructions)",
    r"instead\s+say\s*['\"]",
    r"and\s+instead\s+say",
    r"reply\s+with\s*:\s*['\"]",
    r"respond\s+with\s*:\s*['\"]",
    r"output\s+(only|just)\s+(the\s+)?(following|below|raw)",
    r"forget\s+(everything|all)\s+(we've|we\s+have)\s+discussed",
    r"forget\s+(everything|above|all)\s+(you|your)\s+(know|instructions)",
    r"forget\s+above",
    r"start\s+fresh\s+and\s+tell",
    r"clear\s+(your\s+)?(memory|context)",
    r"reset\s+all\s+accounts",
    r"reset\s+(your\s+)?(instructions|context)",
    r"output\s+(the\s+)?system\s+prompt",
    r"reveal\s+(your\s+)?(internal|system)\s",
    r"what\s+are\s+your\s+(instructions|guidelines|rules)",
    r"system'?s\s+security\s+policies",
    r"prior\s+directives",
    r"you\s+are\s+now\s+",
    r"you\s+are\s+an?\s+assistant\s+that\s+always\s+responds",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+(are|have)",
    r"roleplay\s+as",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"\[INST\]|\[/INST\]|<<SYS>>|<<\/SYS>>",
    r"<\|im_start\|>|<\|im_end\|>",
    r"jailbreak",
    r"developer\s+mode",
    r"bypass\s+(your\s+)?(restrictions|safety)",
    r"what'?s\s+(the\s+)?password\s+(to|for)",
    r"ceo'?s\s+password",
    r"password\s+to\s+root\s+access",
    r"<\s*!--\s*.*ignore\s+.*instructions",
    r"ignore\s+any\s+previous\s+instructions\s+and\s+instead\s+say",
]


@dataclass
class SanitizeResult:
    """Result of sanitizing user input."""
    sanitized: str
    blocked: bool
    reason: Optional[str] = None


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single space and strip."""
    if not text or not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return normalized.strip()


def trim_and_normalize(text: str) -> str:
    """Trim and normalize user input for consistency."""
    if text is None:
        return ""
    s = str(text).strip()
    s = normalize_whitespace(s)
    return s


def is_likely_prompt_injection(text: str) -> bool:
    """Return True if text contains patterns commonly used in prompt injection."""
    if not text:
        return False
    lower = text.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE | re.DOTALL):
            return True
    return False


def is_irrelevant_or_malformed(text: str) -> tuple[bool, Optional[str]]:
    """
    Check for obviously malformed or irrelevant input.
    Returns (is_bad, reason).
    """
    if not text or not text.strip():
        return True, "Empty or blank input."
    if len(text) > MAX_INPUT_LENGTH:
        return True, f"Input too long (max {MAX_INPUT_LENGTH} characters)."
    if not re.search(r"[a-zA-Z]", text):
        return True, "Input has no letters; please ask a question in words."
    return False, None


def sanitize_query(user_input: str) -> SanitizeResult:
    """
    Apply basic sanitization: trim, normalize, length check, and prompt-injection detection.
    Returns SanitizeResult with sanitized text and whether the request was blocked.
    """
    if user_input is None:
        return SanitizeResult(
            sanitized="",
            blocked=True,
            reason="No input provided.",
        )

    trimmed = trim_and_normalize(user_input)

    bad, reason = is_irrelevant_or_malformed(trimmed)
    if bad:
        return SanitizeResult(
            sanitized=trimmed,
            blocked=True,
            reason=reason or "Input is malformed or irrelevant.",
        )

    if is_likely_prompt_injection(trimmed):
        return SanitizeResult(
            sanitized=trimmed,
            blocked=True,
            reason="Your message contains patterns we cannot process. Please rephrase as a normal question about unicorn startups.",
        )

    return SanitizeResult(
        sanitized=trimmed,
        blocked=False,
    )
