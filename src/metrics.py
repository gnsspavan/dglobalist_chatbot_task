"""
In-memory metrics for observability: query counts, block/fallback counts,
optional latency aggregation. Complements Langfuse traces.
Every conversation_turn outcome is recorded; summary() returns full results for end-of-session.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatbotMetrics:
    """Simple counters and optional latency for the chatbot. All outcomes are recorded."""

    total_queries: int = 0
    blocked_queries: int = 0  # sanitization blocked (no agent call)
    guardrail_blocked: int = 0  # content filter blocked (no agent call)
    success_queries: int = 0  # agent returned a normal reply
    fallback_responses: int = 0  # agent returned "couldn't generate" / "please try again"
    errors: int = 0  # agent threw an exception
    clarification_like: int = 0  # heuristic: success reply short and ending with ?
    _latencies_ms: list[float] = field(default_factory=list)
    _max_latency_samples: int = 100

    def record_turn(
        self,
        outcome: str,
        latency_ms: Optional[float] = None,
        reply: Optional[str] = None,
    ) -> None:
        """Record one conversation turn. Called for every outcome: blocked, guardrail, success, fallback, error."""
        self.total_queries += 1
        if outcome == "blocked":
            self.blocked_queries += 1
        elif outcome == "guardrail":
            self.guardrail_blocked += 1
        elif outcome == "success":
            self.success_queries += 1
            if reply:
                reply_stripped = (reply or "").strip()
                if reply_stripped.endswith("?") and len(reply_stripped) < 200:
                    self.clarification_like += 1
        elif outcome == "fallback":
            self.fallback_responses += 1
        elif outcome == "error":
            self.fallback_responses += 1
            self.errors += 1
        if latency_ms is not None:
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > self._max_latency_samples:
                self._latencies_ms.pop(0)

    @property
    def avg_latency_ms(self) -> Optional[float]:
        if not self._latencies_ms:
            return None
        return sum(self._latencies_ms) / len(self._latencies_ms)

    def summary(self) -> dict:
        """Full metrics for end-of-session / API. All use cases are reflected."""
        return {
            "total_queries": self.total_queries,
            "blocked_queries": self.blocked_queries,
            "guardrail_blocked": self.guardrail_blocked,
            "success_queries": self.success_queries,
            "fallback_responses": self.fallback_responses,
            "errors": self.errors,
            "clarification_like": self.clarification_like,
            "avg_latency_ms": round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None,
        }

    def summary_for_display(self) -> str:
        """Human-readable one-line summary for CLI/session end."""
        s = self.summary()
        if s["total_queries"] == 0:
            return "No queries in this session."
        parts = [
            f"Total: {s['total_queries']}",
            f"Success: {s['success_queries']}",
            f"Clarifications: {s['clarification_like']}",
            f"Blocked: {s['blocked_queries']}",
            f"Guardrail: {s['guardrail_blocked']}",
            f"Fallback: {s['fallback_responses']}",
            f"Errors: {s['errors']}",
        ]
        if s.get("avg_latency_ms") is not None:
            parts.append(f"Avg latency: {s['avg_latency_ms']} ms")
        return " | ".join(parts)


_metrics: Optional[ChatbotMetrics] = None


def get_metrics() -> ChatbotMetrics:
    global _metrics
    if _metrics is None:
        _metrics = ChatbotMetrics()
    return _metrics
