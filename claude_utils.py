# llm_utils.py
"""
Helpers for calling Claude (Anthropic) and parsing JSON replies.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List

import anthropic

# ── Custom exceptions ──────────────────────────────────────────────────
class LLMError(Exception):
    """Base class for all LLM‑related errors."""


class LLMAPIError(LLMError):
    """Wrapping lower‑level API/HTTP errors."""

class LLMParseError(Exception):
    pass

def call_model(
    client: anthropic.Anthropic,
    messages: List[Dict[str, str]],
    system: str,
    temperature: float = 1.0,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> Any:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            temperature=temperature,
        )
        return resp

    except LLMError:
        # re‑throw our own subclasses unchanged
        raise
    except Exception as exc:
        # Wrap anything else
        raise LLMAPIError(f"Error calling Anthropic API: {exc}") from exc
