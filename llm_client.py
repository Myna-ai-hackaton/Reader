"""
llm_client.py
=============

A tiny OpenAI-compatible client wrapper.

Works against:
    - A local Podman AI Lab model exposed at OPENAI_BASE_URL=http://localhost:PORT/v1
    - The real OpenAI cloud API (omit OPENAI_BASE_URL, set OPENAI_API_KEY)

Why not just use the OpenAI client directly everywhere? Two reasons:
    1. Local Podman models often wrap their JSON output in prose ("Sure, here is
       the JSON: { ... }"). `extract_json_object` defends against that.
    2. Centralising the configuration means the rest of the codebase never has
       to think about env vars or model names.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Load .env once at import time. Real env vars still win over .env values.
load_dotenv()


# -----------------------------------------------------------------------------
# Client + model wiring
# -----------------------------------------------------------------------------

def get_client() -> OpenAI:
    """
    Build an OpenAI client configured for whatever endpoint is in the env.

    - If OPENAI_BASE_URL is set, point the client at it (Podman AI Lab or any
      other OpenAI-compatible server). API key is usually irrelevant for local
      servers, so it defaults to "not-needed".
    - Otherwise, talk to the real OpenAI cloud and use OPENAI_API_KEY as-is.
    """
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")

    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key)
    return OpenAI(api_key=api_key)


def get_model() -> str:
    """Return the configured model name, with a sensible cloud default."""
    return os.getenv("LLM_MODEL", "gpt-4.1-mini")


# -----------------------------------------------------------------------------
# Chat helpers
# -----------------------------------------------------------------------------

def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """
    Single-turn chat: one system message + one user message → string response.

    Returns the assistant's text content, or an empty string if the API
    returned no content for any reason.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def chat_stream(system: str, user: str, temperature: float = 0.1):
    """
    Same as `chat` but yields text chunks as the model produces them.

    Designed for `st.write_stream(chat_stream(...))` so the UI feels
    responsive while the model is still generating. Each yielded value is
    a short string fragment (a few tokens). The generator ends when the
    model's response is complete.

    If the streaming endpoint errors mid-response, the partial text that
    arrived before the error is still yielded; the caller can decide
    whether the partial answer is usable. Network errors at the start of
    the stream propagate as exceptions because there is nothing useful
    to yield.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    for chunk in response:
        # Defensive: some providers occasionally emit chunks with no
        # `choices` (keep-alive frames). Skip those without raising.
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


# -----------------------------------------------------------------------------
# JSON-mode helpers (resilient to chatty local models)
# -----------------------------------------------------------------------------

def extract_json_object(text: str) -> dict[str, Any]:
    """
    Pull a JSON object out of an LLM response that may include surrounding prose.

    Strategy:
        1. Try parsing the whole string. Many models do return clean JSON.
        2. If that fails, locate the first '{' and the last '}' and try the
           substring between them. This is forgiving of leading apologies and
           trailing explanations.
        3. If both attempts fail, raise ValueError. Callers decide how to recover.
    """
    text = text.strip()

    # Attempt 1: the whole response is JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: take the widest {...} slice we can find.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSON object from model response: {exc}"
            ) from exc

    raise ValueError("No JSON object found in model response.")


def chat_json(system: str, user: str, temperature: float = 0.0) -> dict[str, Any]:
    """
    Like `chat`, but parse the response as a JSON object before returning it.

    Uses temperature=0.0 by default because JSON outputs benefit from
    deterministic decoding. Raises ValueError on unparseable responses.
    """
    raw = chat(system=system, user=user, temperature=temperature)
    return extract_json_object(raw)