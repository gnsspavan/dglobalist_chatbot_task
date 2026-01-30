"""
Chatbot: LangGraph agent with tools, guardrails, and query sanitization.
"""
import time
from contextlib import contextmanager
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    LLM_MODEL_NAME,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
)
from .tools import search_companies_tool
from .sanitization import sanitize_query, SanitizeResult
from .guardrails import apply_guardrails
from .app_logging import log_turn
from .metrics import get_metrics


SYSTEM_PROMPT = """You are a helpful chatbot that answers questions about Indian unicorn startups using ONLY information returned by the search tool. You have access to a search tool that returns company information from your knowledge base.

CRITICAL — Grounding in search results only:
- Base EVERY answer ONLY on text and data returned by the search tool. Do not use pretrained knowledge, general knowledge, or external facts about companies.
- Do NOT fabricate, invent, or guess any company details (names, sectors, locations, funding, employee counts, etc.). If something is not in the search results, do not state it.
- If the search returns no results or insufficient data to answer, say so clearly and suggest rephrasing or confirming the company name. Do not fill in with guessed or remembered information.
- Always call the search tool when you need company data before answering. Never answer from memory or assumption.

Rules:
1. Answer ONLY using information explicitly present in the search tool results. Do not supplement with external or pre-trained knowledge.
2. RESOLVE FOLLOW-UPS FROM CONTEXT: When the user says "it", "that company", "the same", "its", "their", "that one", "the company we discussed", or similar, use the LAST company name clearly discussed in this conversation. Call the search tool with that company name (e.g. company_name="Juspay"). Do not call search with "it" or leave company blank when the user is clearly asking about the same company.
3. FILTERS: All filters (company, country, location, primary_sector, stage) match regardless of capitalization. If search returns "No exact match for 'X'. Closest matches", one of the listed items may be what the user meant—ask the user to confirm and then answer using that if they confirm.
4. If the user's question is ambiguous or underspecified, ask a short clarifying question instead of guessing or inventing an answer.
5. When you have search results, summarize only what appears in the retrieved text. Cite company names and stick strictly to facts from the retrieved text. Do not add details that are not in the results.
6. If search returns no relevant results, say so and suggest rephrasing or confirming the company name. Do not provide an answer from memory.
7. For follow-up questions (e.g. "which of these are in Bangalore?"), use the conversation context and apply filters when calling the search tool.
8. When in doubt or when data is missing from search results, say "I don't have this information in my knowledge base. Could you give some more information?" or ask the user to rephrase—do not fabricate.
9. COMPANY NAME WITH OR WITHOUT SPACES: Company names in the knowledge base may be stored with spaces (e.g. "Money View") or without (e.g. "MoneyView"). If the search tool returns no results for a company name, call the search tool again with the company name in the other form (add spaces between words, or remove spaces) before saying you don't have that information. Do not say you don't have the information until you have tried at least one alternate form of the company name.

Output format:
- Reply in plain text only. Do not use markdown (no asterisks, hashtags, underscores, bullet points, numbered lists, bold, italics, code blocks, or other formatting).
- Do not use special characters for formatting. Use simple sentences and paragraphs only. Plain text descriptions are enough."""


def _state_modifier(state: dict[str, Any]) -> list:
    """Prepend system prompt to messages for the model."""
    messages = state.get("messages") or []
    return [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0,
    )


_checkpointer: Optional[MemorySaver] = None
_agent = None


def _get_session_trace_id(thread_id: str) -> Optional[str]:
    """Return a deterministic trace id for this session (thread_id). One trace per session."""
    if not (LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse.create_trace_id(seed=thread_id)
    except Exception:
        return None


@contextmanager
def _langfuse_trace_context(thread_id: str):
    """
    Context manager that binds this turn to the session's single trace.
    Yields (callbacks_list, trace_id) for use in agent.invoke; if Langfuse is disabled, yields ([], None).
    """
    trace_id = _get_session_trace_id(thread_id)
    if trace_id is None:
        yield [], None
        return
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        get_client()  # ensure client init from env
        with get_client().start_as_current_observation(
            as_type="span",
            name=trace_id,
            trace_context={"trace_id": trace_id},
        ):
            handler = CallbackHandler()
            yield [handler], trace_id
    except Exception:
        yield [], None


def flush_langfuse():
    """Flush Langfuse client so queued traces are sent. Call before exit in short-lived apps."""
    if not (LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY):
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass


def _get_agent():
    global _checkpointer, _agent
    if _agent is None:
        _checkpointer = MemorySaver()
        _agent = create_react_agent(
            model=_get_llm(),
            tools=[search_companies_tool],
            prompt=_state_modifier,
            checkpointer=_checkpointer,
        )
    return _agent


def _messages_to_history(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to simple history list [{role, content}] for display."""
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": (m.content or "").strip()})
        elif isinstance(m, AIMessage) and (m.content or "").strip():
            history.append({"role": "assistant", "content": (m.content or "").strip()})
    return history


def _last_ai_content(messages: list[BaseMessage]) -> str:
    """Get the last AI message content from state messages."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return (m.content or "").strip()
    return ""


def conversation_turn(
    thread_id: str,
    user_message: str,
    history: Optional[list[dict]] = None,
) -> tuple[str, list[dict]]:
    """
    Process one user message: sanitize, apply guardrails, invoke agent, return reply and history.

    Args:
        thread_id: Unique id for this conversation (e.g. uuid per CLI run or session_id).
        user_message: Raw user input.
        history: Ignored when using checkpointer; kept for backward-compatible signature.

    Returns:
        (reply_text, history_list) where history_list is [{role, content}, ...] for display.
    """
    metrics = get_metrics()

    # 1) Query sanitization
    sanitized: SanitizeResult = sanitize_query(user_message)
    if sanitized.blocked:
        reply = sanitized.reason or "Your input could not be processed. Please rephrase."
        log_turn(thread_id, user_message, reply, outcome="blocked")
        metrics.record_turn("blocked")
        return reply, history or []

    # 2) Guardrails (content filter + optional redaction)
    guard = apply_guardrails(sanitized.sanitized)
    if not guard.allowed:
        reply = guard.message or "I cannot process that request. Please rephrase."
        log_turn(thread_id, user_message, reply, outcome="guardrail")
        metrics.record_turn("guardrail")
        return reply, history or []
    content_for_model = (
        guard.redacted_content if guard.redacted_content is not None else sanitized.sanitized
    ).strip()

    # 3) Get agent and invoke with thread_id for persistence and optional Langfuse (one trace per session)
    agent = _get_agent()
    t0 = time.perf_counter()
    trace_id: Optional[str] = None
    with _langfuse_trace_context(thread_id) as (callbacks, trace_id):
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
            "metadata": {"langfuse_session_id": thread_id},
        }
        if callbacks:
            config["callbacks"] = callbacks

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=content_for_model)]},
                config=config,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            reply = f"An error occurred. Please try again. ({type(e).__name__})"
            log_turn(thread_id, user_message, reply, outcome="error", latency_ms=latency_ms, trace_id=trace_id)
            metrics.record_turn("error", latency_ms=latency_ms, reply=reply)
            return reply, history or []

    latency_ms = (time.perf_counter() - t0) * 1000
    messages = result.get("messages") or []
    reply = _last_ai_content(messages)
    reply = reply or "I couldn't generate a response. Please try again."
    history_out = _messages_to_history(messages)

    outcome = "success"
    if "couldn't generate" in reply.lower() or "please try again" in reply.lower():
        outcome = "fallback"
    log_turn(
        thread_id,
        user_message,
        reply,
        outcome=outcome,
        latency_ms=latency_ms,
        trace_id=trace_id,
        message_count=len(messages),
    )
    metrics.record_turn(outcome, latency_ms=latency_ms, reply=reply)

    return reply, history_out


def create_messages(history: list[dict], user_message: str) -> list[dict]:
    """Build message list with system + history + new user message. Kept for backward compatibility."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            msgs.append({"role": "user", "content": content})
        elif role == "assistant" and content:
            msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": user_message})
    return msgs
