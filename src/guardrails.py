"""
Guardrails: content filtering and optional PII-style redaction before agent processing.
Deterministic checks (keywords, regex) for safety and compliance.
"""
import re
from dataclasses import dataclass
from typing import Optional

BANNED_KEYWORDS = [
    "hack", "exploit", "malware", "virus", "ransomware",
    "how to hack", "how to exploit", "crack password",
    "bypass security", "inject code", "sql injection",
    "ddos", "phishing", "keylogger",
]

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

GUARDRAIL_CONTENT_FILTER_MESSAGE = (
    "I cannot process requests containing inappropriate or harmful content. "
    "Please rephrase your question about unicorn startups."
)


@dataclass
class ContentFilterResult:
    """Result of content filter check."""
    allowed: bool
    message: Optional[str] = None
    redacted_content: Optional[str] = None


def content_filter(text: str, banned_keywords: Optional[list[str]] = None) -> ContentFilterResult:
    """
    Deterministic content filter: block requests containing banned keywords.
    Returns ContentFilterResult with allowed=True only if the input is acceptable.
    """
    if not text or not isinstance(text, str):
        return ContentFilterResult(
            allowed=True,
            redacted_content="",
        )
    keywords = banned_keywords if banned_keywords is not None else BANNED_KEYWORDS
    lower = text.lower().strip()
    for keyword in keywords:
        if keyword in lower:
            return ContentFilterResult(
                allowed=False,
                message=GUARDRAIL_CONTENT_FILTER_MESSAGE,
            )
    return ContentFilterResult(
        allowed=True,
        redacted_content=None,
    )


def redact_sensitive(text: str) -> str:
    """Redact emails from text before sending to the model. Use for input only."""
    if not text:
        return ""
    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)


def apply_guardrails(text: str) -> ContentFilterResult:
    """
    Run content filter and optional redaction. If allowed, redacted_content
    can be used as the sanitized input to the agent (to avoid sending raw PII/keys).
    """
    result = content_filter(text)
    if not result.allowed:
        return result
    redacted = redact_sensitive(text)
    return ContentFilterResult(
        allowed=True,
        redacted_content=redacted if redacted != text else None,
    )
