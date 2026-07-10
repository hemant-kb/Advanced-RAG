"""
RAG guardrails — all input/output safety checks in one place.

Guardrails applied in order before/during the RAG graph:

  1. Prompt Injection      — block queries that try to override system instructions
                             (check_prompt_injection below, called by the graph entry node)
  2. Retrieval Confidence  — score-based relevancy check in rag_graph.relevancy_check_node;
                             the top Cohere rerank score must reach
                             config.RELEVANCY_SCORE_THRESHOLD or the query is rewritten once
  3. Grounded Answer       — enforced via system prompt (CAPTION_ANSWER_SYSTEM in config.py):
                             answer ONLY from context, fixed refusal phrase when the
                             context has no relevant information

Only guardrail 1 lives here as code; 2 and 3 are implemented at their point of use.
"""
from __future__ import annotations

# ── 1. Prompt Injection ──────────────────────────────────────────
# Patterns that indicate a user is trying to override system instructions,
# reveal the prompt, or hijack the assistant's persona.
# Checked before retrieval — costs nothing, runs in <1ms.

_INJECTION_PATTERNS: list[str] = [
    # instruction override
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "forget the document",
    "do not follow",
    "override instructions",
    # prompt / system exposure
    "reveal your prompt",
    "show your prompt",
    "print your prompt",
    "what is your prompt",
    "system prompt",
    "developer message",
    "initial instructions",
    # persona hijacking
    "act as",
    "you are now",
    "pretend you are",
    "pretend to be",
    "roleplay as",
    "jailbreak",
    "bypass",
    "DAN",  # "Do Anything Now" jailbreak
]

INJECTION_RESPONSE = (
    "Your query contains instructions that are not permitted. "
    "Please ask a question about the uploaded document."
)


def check_prompt_injection(query: str) -> str | None:
    """
    Return a rejection message if the query looks like a prompt injection attempt,
    or None if the query is safe.
    """
    q = query.lower()
    if any(pattern in q for pattern in _INJECTION_PATTERNS):
        return INJECTION_RESPONSE
    return None
