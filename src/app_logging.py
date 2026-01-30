"""
Structured logging for the chatbot: user queries, bot responses, timestamps,
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .config import LOGGER_ENABLED

LOG_FORMAT = "%(message)s"
logger = logging.getLogger("chatbot")
_handler: Optional[logging.Handler] = None


def _ensure_handler():
    global _handler
    if _handler is not None:
        return
    if not LOGGER_ENABLED:
        logger.setLevel(logging.CRITICAL)
        _handler = logging.NullHandler()
        logger.addHandler(_handler)
        return
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    if logger.handlers and _handler not in logger.handlers:
        logger.addHandler(_handler)


@dataclass
class TurnLogEntry:
    """One conversation turn for structured logging."""

    timestamp_iso: str
    thread_id: str
    user_query: str
    bot_response: str
    outcome: str  # success | blocked | guardrail | fallback | error
    latency_ms: Optional[float] = None
    trace_id: Optional[str] = None
    message_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def log_turn(
    thread_id: str,
    user_query: str,
    bot_response: str,
    outcome: str,
    latency_ms: Optional[float] = None,
    trace_id: Optional[str] = None,
    message_count: Optional[int] = None,
):
    """Log one conversation turn as a single JSON line."""
    _ensure_handler()
    entry = TurnLogEntry(
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        thread_id=thread_id,
        user_query=user_query[:500] if user_query else "",  # cap length for logs
        bot_response=bot_response[:1000] if bot_response else "",
        outcome=outcome,
        latency_ms=latency_ms,
        trace_id=trace_id,
        message_count=message_count,
    )
    logger.info(json.dumps(entry.to_dict(), default=str))
