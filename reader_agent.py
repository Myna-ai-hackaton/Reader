"""
reader_agent.py
===============

The Reader Agent. This is the brain of the Reader pipeline.

Flow per user query:

    1.  Load arbitrary JSON memory from disk (real Writer output or mock).
    2.  Serialize that JSON to text and hand it directly to the LLM.
    3.  Ask the LLM: "Is this enough to answer? If not, which files / commits /
        keywords should I look at in the actual Git repo?"
    4.  If the LLM asks for a deep-dive AND a valid repo path is available,
        run focused, read-only Git commands using the LLM's hints.
    5.  Ask the LLM to write a final answer using:
            - the arbitrary JSON memory
            - its own memory analysis
            - optional deep-dive evidence
    6.  Return a rich result dict that the Streamlit UI can render verbatim.

Design rules:
    - We do NOT assume any specific schema for the JSON memory.
    - We do NOT search or chunk the JSON ourselves; the LLM sees the raw text.
    - The Reader never writes to Writer files. The only file it may produce is
      `reader/data/RELEASE_NOTES.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_client import chat, chat_json
from reader_memory import load_raw_memory, memory_to_prompt_text
from repo_tools import (
    focused_repo_deep_dive,
    format_deep_dive_for_prompt,
    validate_repo,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

READER_DIR = Path(__file__).resolve().parent
RELEASE_NOTES_PATH = READER_DIR / "data" / "RELEASE_NOTES.md"


# -----------------------------------------------------------------------------
# System prompts
# -----------------------------------------------------------------------------

ANALYZE_MEMORY_SYSTEM = """\
You are an AI Git Project Manager Reader Agent.

You receive:
1. A user query.
2. An arbitrary JSON memory file produced by another LLM (the "Writer").

You must inspect the JSON memory and decide:
- whether the memory contains enough information to answer the query,
- whether a focused Git repo deep-dive is needed,
- if deep-dive is needed, which files / commits / keywords are most likely relevant.

Return ONLY a single valid JSON object with this exact shape:
{
  "can_answer_from_memory": true,
  "needs_repo_deep_dive": false,
  "query_type": "release_notes|qa_risk|developer_forensics|pm_summary|general",
  "audience": "pm|qa|developer|mixed",
  "relevant_memory_evidence": ["short evidence item 1", "short evidence item 2"],
  "likely_files": ["path/to/file.py"],
  "likely_commits": ["abc1234"],
  "likely_keywords": ["auth", "login"],
  "reason": "short explanation"
}

Rules:
- Do not assume a fixed JSON schema. The memory could have ANY structure.
  Treat it as arbitrary structured notes about a codebase.
- If the query asks for exact code location, implementation details, root
  cause analysis, or line-level details, set needs_repo_deep_dive = true.
- If the memory is too vague or missing key facts to answer the query, set
  needs_repo_deep_dive = true.
- If there is enough memory to give a high-level PM / QA / release-notes
  answer, set can_answer_from_memory = true.
- Pull likely_files, likely_commits, and likely_keywords from whatever the
  memory actually contains — do not invent them.
- Output ONLY the JSON object. No prose before or after.
"""


FINAL_ANSWER_SYSTEM = """\
You are an AI Git Project Manager.

You answer PM / QA / developer questions about a software project using:
1. An arbitrary JSON memory file produced by another LLM (the "Writer").
2. A memory analysis describing what is and isn't in that memory.
3. Optionally, focused Git repo deep-dive evidence (commit patches, file
   histories, grep hits) gathered to fill gaps in the memory.

Rules:
- Do not assume a fixed schema for the memory. It may have any structure.
- Use ONLY the evidence provided. Do not invent commits, files, tickets, PRs,
  or features that are not in the evidence.
- Cite commit hashes and file paths whenever you reference them, taking them
  verbatim from the memory or the repo evidence.
- If the evidence is insufficient to answer fully, say clearly what is missing
  and what would resolve it.
- Tailor the tone to the audience:
    * PMs: focus on product / business impact, user-facing changes, risk.
    * QA:  give concrete test recommendations and risk areas.
    * Developers: include technical details, files, commits, and
                  implementation clues.
- Keep the answer well-structured with short headings or bullets when useful.
"""


# -----------------------------------------------------------------------------
# Step 1: ask the LLM to analyse the memory
# -----------------------------------------------------------------------------

def _fallback_keywords_from_query(query: str) -> list[str]:
    """
    If the analysis-LLM call fails, derive a few naive keywords from the
    user's query so a deep-dive can still attempt something useful.

    This is deliberately dumb (no stemming, no synonyms) — it only exists as
    a safety net for malformed model responses.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in",
        "on", "for", "with", "is", "are", "was", "were", "be", "been", "being",
        "what", "which", "who", "whom", "where", "when", "why", "how", "did",
        "do", "does", "done", "i", "you", "we", "they", "it", "this", "that",
        "these", "those", "my", "our", "your", "their", "should", "could",
        "would", "can", "will", "shall", "may", "might", "about", "from",
    }
    words = [w.strip(".,;:!?\"'()[]{}").lower() for w in query.split()]
    return [w for w in words if w and w not in stopwords and len(w) > 2][:8]


def analyze_memory_with_llm(query: str, memory_text: str) -> dict[str, Any]:
    """
    Ask the LLM to read the raw JSON memory and decide:
    can it answer from memory? does it need a deep-dive? what to look at?

    Returns the parsed JSON object. On parse failure, returns a safe fallback
    that requests a deep-dive (since we genuinely don't know what the model
    saw).
    """
    user_prompt = (
        f"User query:\n{query}\n\n"
        "Arbitrary JSON memory (produced by another LLM, schema unknown):\n"
        "```json\n"
        f"{memory_text}\n"
        "```\n\n"
        "Return ONLY the JSON object described in the system prompt."
    )

    try:
        return chat_json(
            system=ANALYZE_MEMORY_SYSTEM,
            user=user_prompt,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — bad JSON from model is recoverable
        return {
            "can_answer_from_memory": False,
            "needs_repo_deep_dive": True,
            "query_type": "general",
            "audience": "mixed",
            "relevant_memory_evidence": [],
            "likely_files": [],
            "likely_commits": [],
            "likely_keywords": _fallback_keywords_from_query(query),
            "reason": (
                f"Fallback analysis because model did not return valid JSON: {exc}"
            ),
        }


# -----------------------------------------------------------------------------
# Step 2: ask the LLM to write the final answer
# -----------------------------------------------------------------------------

def generate_final_answer(
    query: str,
    memory_text: str,
    memory_analysis: dict[str, Any],
    deep_dive_context: str | None,
) -> str:
    """
    Compose the final answer prompt and let the LLM write the response.

    The model sees everything we know: the raw memory, the meta-analysis it
    just produced, and the optional deep-dive evidence. It is responsible for
    synthesising these into a coherent answer.
    """
    import json as _json  # local alias so we don't shadow anything

    deep_dive_block = deep_dive_context or "No repo deep-dive was performed."

    user_prompt = (
        f"User query:\n{query}\n\n"
        "Arbitrary JSON memory (schema unknown — read as structured notes):\n"
        "```json\n"
        f"{memory_text}\n"
        "```\n\n"
        "Memory analysis (from the previous step):\n"
        "```json\n"
        f"{_json.dumps(memory_analysis, ensure_ascii=False, indent=2)}\n"
        "```\n\n"
        "Focused repo deep-dive evidence:\n"
        f"{deep_dive_block}\n\n"
        "Write the final answer for the appropriate audience. Cite commits "
        "and file paths verbatim when you use them. If something cannot be "
        "answered from the available evidence, say so explicitly."
    )

    return chat(
        system=FINAL_ANSWER_SYSTEM,
        user=user_prompt,
        temperature=0.2,
    )


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------

def run_reader_agent(
    query: str,
    memory_path: str | None = None,
    repo_path: str | None = None,
    max_memory_chars: int = 50_000,
    connected_memory_path: str | None = None,
) -> dict[str, Any]:
    """
    Run one full query end-to-end.

    Args:
        query:                 The user's natural-language question.
        memory_path:           Manual override for the memory JSON file
                               (Advanced mode). Wins over everything else.
        repo_path:             Path to a local Git checkout for deep-dives.
                               When None or invalid, no deep-dive is performed.
        max_memory_chars:      Soft cap on how much memory JSON is shown
                               to the LLM.
        connected_memory_path: Path to `.myna/system_memory_index.json`
                               inside a freshly connected GitHub clone.
                               Used when `memory_path` is not set.

    Returns:
        {
            "answer":          <final answer string>,
            "memory_analysis": <LLM analysis dict>,
            "deep_dive_result": <dict or None>,
            "trace":           <list[str] of step-by-step status messages>,
            "memory_metadata": {
                "source_path":          <which file was loaded>,
                "memory_chars":         <length of the serialized memory>,
                "memory_was_truncated": <bool>,
            },
        }
    """
    trace: list[str] = []

    # --- Load memory --------------------------------------------------------
    loaded = load_raw_memory(
        path=memory_path,
        connected_memory_path=connected_memory_path,
    )
    trace.append(f"Loaded memory from: {loaded['source_path']}")

    if loaded["error"]:
        # Surface the error directly to the UI; no point calling the LLM.
        trace.append(f"Memory load error: {loaded['error']}")
        return {
            "answer": (
                "Could not run the Reader Agent because the memory file "
                f"could not be loaded.\n\nError: {loaded['error']}"
            ),
            "memory_analysis": {},
            "deep_dive_result": None,
            "trace": trace,
            "memory_metadata": {
                "source_path": loaded["source_path"],
                "error": loaded["error"],
            },
        }

    # --- Serialize for the LLM ---------------------------------------------
    memory_payload = memory_to_prompt_text(
        loaded["raw"], max_chars=max_memory_chars
    )
    memory_text = memory_payload["text"]
    trace.append(
        f"Serialized memory: {memory_payload['char_count']} chars "
        f"(truncated={memory_payload['was_truncated']})."
    )

    # --- Step 1: meta-analysis ---------------------------------------------
    memory_analysis = analyze_memory_with_llm(query, memory_text)
    trace.append(
        "Memory analysis reason: "
        f"{memory_analysis.get('reason', '(no reason provided)')}"
    )

    # --- Step 2: optional focused deep-dive --------------------------------
    deep_dive_result: dict[str, Any] | None = None
    deep_dive_context: str | None = None

    if memory_analysis.get("needs_repo_deep_dive"):
        if repo_path and validate_repo(repo_path):
            trace.append(
                f"Deep-dive requested; valid repo at {repo_path}. Running "
                "focused Git commands."
            )
            deep_dive_result = focused_repo_deep_dive(
                repo_path=repo_path,
                files=memory_analysis.get("likely_files", []) or [],
                commits=memory_analysis.get("likely_commits", []) or [],
                keywords=memory_analysis.get("likely_keywords", []) or [],
            )
            deep_dive_context = format_deep_dive_for_prompt(deep_dive_result)
            trace.append(
                "Deep-dive complete: "
                f"{len(deep_dive_result.get('commit_patches', []))} commits, "
                f"{len(deep_dive_result.get('file_logs', []))} file logs, "
                f"{len(deep_dive_result.get('grep_results', []))} grep hits."
            )
        else:
            trace.append(
                "Deep-dive requested but no valid Git repo path is "
                "available; answering from memory only."
            )
    else:
        trace.append("Memory deemed sufficient; skipping repo deep-dive.")

    # --- Step 3: final answer ----------------------------------------------
    answer = generate_final_answer(
        query=query,
        memory_text=memory_text,
        memory_analysis=memory_analysis,
        deep_dive_context=deep_dive_context,
    )
    trace.append("Generated final answer.")

    return {
        "answer": answer,
        "memory_analysis": memory_analysis,
        "deep_dive_result": deep_dive_result,
        "trace": trace,
        "memory_metadata": {
            "source_path": loaded["source_path"],
            "memory_chars": memory_payload["char_count"],
            "memory_was_truncated": memory_payload["was_truncated"],
        },
    }


# -----------------------------------------------------------------------------
# Convenience: generate RELEASE_NOTES.md
# -----------------------------------------------------------------------------

def generate_release_notes(
    memory_path: str | None = None,
    repo_path: str | None = None,
    connected_memory_path: str | None = None,
) -> dict[str, Any]:
    """
    Special-cased query that asks the agent to produce release notes and
    writes the answer to `reader/data/RELEASE_NOTES.md`.

    Returns the same result shape as `run_reader_agent`, with an extra trace
    entry confirming the file was written.
    """
    query = (
        "Draft release notes from the current project memory. Group them "
        "into New Features, Bug Fixes, Internal Improvements, and QA Notes."
    )

    result = run_reader_agent(
        query=query,
        memory_path=memory_path,
        repo_path=repo_path,
        connected_memory_path=connected_memory_path,
    )

    # Always try to write something — even an error message is useful to
    # see in the downloaded file.
    RELEASE_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELEASE_NOTES_PATH.write_text(result["answer"], encoding="utf-8")
    result["trace"].append(f"Saved release notes to {RELEASE_NOTES_PATH}")

    return result