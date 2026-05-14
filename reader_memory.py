"""
reader_memory.py
================

Responsibilities (and ONLY these):
    1. Choose which memory JSON file to use (real Writer output, mock, or override).
    2. Load that file as arbitrary JSON.
    3. Serialize that arbitrary JSON to a text blob suitable for an LLM prompt.

This module is intentionally schema-agnostic. It does NOT:
    - Inspect or validate the JSON's structure.
    - Look for specific fields (entries, commits, business_summary, etc.).
    - Search, filter, or chunk the JSON.

The Reader Agent itself hands the raw serialized JSON to the LLM and lets the
LLM decide how to use it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Path constants
# -----------------------------------------------------------------------------

READER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = READER_DIR.parent

# The Writer pipeline owns this file inside a sibling `writer/` checkout.
# Kept for backwards compatibility with the original monorepo layout, but
# the canonical location is now `.myna/system_memory_index.json` inside any
# Myna-enabled repo (see github_source.MEMORY_RELATIVE_PATH).
REAL_MEMORY_PATH = PROJECT_ROOT / "writer" / "data" / "system_memory_index.json"

# Local fallback so Reader works even before any repo is connected.
MOCK_MEMORY_PATH = READER_DIR / "Myna" / "mock_system_memory_index.json"


# -----------------------------------------------------------------------------
# Path selection
# -----------------------------------------------------------------------------

def choose_memory_path(
    path: str | None = None,
    connected_memory_path: Path | str | None = None,
) -> Path:
    """
    Decide which JSON memory file to load.

    Precedence:
        1. Explicit `path` argument (user manual override from the UI).
        2. `connected_memory_path` — the `.myna/system_memory_index.json`
           found inside a freshly connected GitHub repo. Provided by the UI
           after a successful `github_source.connect_repo(...)` call.
        3. The legacy sibling-`writer/` path, if it exists on disk.
        4. The bundled mock memory file (last-resort fallback so the agent
           always has something to work with).
    """
    if path:
        return Path(path)
    if connected_memory_path:
        return Path(connected_memory_path)
    if REAL_MEMORY_PATH.exists():
        return REAL_MEMORY_PATH
    return MOCK_MEMORY_PATH


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_raw_memory(
    path: str | None = None,
    connected_memory_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Load the memory JSON file as an arbitrary Python object.

    Returns a dict with a uniform shape so callers do not have to handle
    multiple exception types:

        {
            "source_path": "<absolute path to the file we tried to read>",
            "raw":         <parsed JSON value, or None on error>,
            "error":       <human-readable error string, or None on success>
        }
    """
    memory_path = choose_memory_path(path, connected_memory_path)

    if not memory_path.exists():
        return {
            "source_path": str(memory_path),
            "raw": None,
            "error": f"Memory file does not exist: {memory_path}",
        }

    try:
        with memory_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:  # noqa: BLE001 — surface any parse error to the UI
        return {
            "source_path": str(memory_path),
            "raw": None,
            "error": f"Failed to parse memory JSON: {exc}",
        }

    return {
        "source_path": str(memory_path),
        "raw": raw,
        "error": None,
    }


# -----------------------------------------------------------------------------
# Serialization for the LLM prompt
# -----------------------------------------------------------------------------

def memory_to_prompt_text(raw: Any, max_chars: int = 50_000) -> dict[str, Any]:
    """
    Convert arbitrary JSON-compatible data into a pretty-printed text blob
    that can be embedded directly into an LLM prompt.

    If the rendered text exceeds `max_chars`, it is truncated and a clear
    marker is appended so the LLM knows it does not have the full file.

    Returns:
        {
            "text":          <serialized JSON string, possibly truncated>,
            "char_count":    <length of the returned text>,
            "was_truncated": <bool>,
        }
    """
    text = json.dumps(raw, ensure_ascii=False, indent=2)

    was_truncated = False
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[MEMORY JSON TRUNCATED]"
        was_truncated = True

    return {
        "text": text,
        "char_count": len(text),
        "was_truncated": was_truncated,
    }