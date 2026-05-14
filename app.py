"""
app.py
======

Streamlit UI for the Myna-ai Reader Agent.

This module is intentionally thin: it does NOT implement any agent logic,
schema parsing, or LLM calls itself. It only:
    - Lets the user pick a memory JSON file and an optional local Git repo.
    - Lets the user type a query (or pick a quick canned one).
    - Calls `run_reader_agent(...)` or `generate_release_notes(...)`.
    - Displays the result dictionary returned by the agent verbatim.

Crucially, the UI never reads specific fields from the Writer JSON. It only
renders what the agent returns. That keeps the UI schema-agnostic too.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from reader_agent import (
    RELEASE_NOTES_PATH,
    generate_release_notes,
    run_reader_agent,
)
from reader_memory import (
    MOCK_MEMORY_PATH,
    REAL_MEMORY_PATH,
    choose_memory_path,
    load_raw_memory,
)
from repo_tools import validate_repo


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Myna-ai Reader",
    page_icon="🪺",
    layout="wide",
)

st.title("🪺 Myna-ai Reader")
st.caption(
    "An AI Git Project Manager that answers PM, QA, and developer questions "
    "from the Writer's arbitrary JSON memory — and optionally from a focused "
    "deep-dive into your local Git repo."
)

# Session state used across reruns.
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -----------------------------------------------------------------------------
# Sidebar — configuration
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    # ---- Memory file --------------------------------------------------------
    default_memory_path = str(choose_memory_path())
    memory_path_input = st.text_input(
        "Memory JSON path",
        value=default_memory_path,
        help=(
            "Defaults to the Writer's `system_memory_index.json` if it exists, "
            "otherwise the bundled mock file. You can override with any path."
        ),
    )

    # Show whether we found Writer output vs falling back to the mock.
    if Path(memory_path_input) == REAL_MEMORY_PATH:
        st.caption("Using Writer output.")
    elif Path(memory_path_input) == MOCK_MEMORY_PATH:
        st.caption("Using mock memory (Writer output not found).")
    else:
        st.caption("Using a custom memory path.")

    # Probe-load so the user gets immediate feedback.
    probe = load_raw_memory(memory_path_input)
    if probe["error"]:
        st.error(f"Could not load memory: {probe['error']}")
    else:
        st.success("Memory JSON loaded OK.")
    st.caption(f"Source: `{probe['source_path']}`")

    st.divider()

    # ---- Optional repo path -------------------------------------------------
    repo_path_input = st.text_input(
        "Local Git repo path (optional)",
        value="",
        help=(
            "If you provide a valid Git repo path, the agent may run focused "
            "`git show`, `git log`, and `git grep` commands when the JSON "
            "memory is not enough."
        ),
    )
    if repo_path_input:
        if validate_repo(repo_path_input):
            st.success("Valid Git repo detected.")
        else:
            st.warning("Path does not look like a Git repo; deep-dive disabled.")

    st.divider()

    # ---- Quick questions ----------------------------------------------------
    st.subheader("Quick questions")
    quick_question = st.selectbox(
        "Pick a starter (optional)",
        options=[
            "",
            "What should QA test?",
            "Summarize recent changes for a PM update.",
            "Why did we change the login logic and where is it implemented?",
            "What are the riskiest changes and what should we test extra carefully?",
            "Which database changes happened and what should we check before deploying?",
        ],
        index=0,
    )


# -----------------------------------------------------------------------------
# Main layout — two columns
# -----------------------------------------------------------------------------

main_col, side_col = st.columns([2, 1])


# ---------- Main column: query + answer --------------------------------------

with main_col:
    st.subheader("Ask the Reader Agent")

    query = st.text_area(
        "Your question",
        value=quick_question,
        height=120,
        placeholder=(
            "e.g. 'What should QA test?' or 'Why did we change the login "
            "logic and where is it implemented?'"
        ),
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        ask_clicked = st.button("Ask Agent", type="primary", use_container_width=True)
    with btn_col2:
        notes_clicked = st.button(
            "Generate RELEASE_NOTES.md", use_container_width=True
        )

    # Run the agent on Ask.
    if ask_clicked:
        if not query.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Running the Reader Agent..."):
                st.session_state.last_result = run_reader_agent(
                    query=query,
                    memory_path=memory_path_input or None,
                    repo_path=repo_path_input or None,
                )

    # Run the special release-notes path on the other button.
    if notes_clicked:
        with st.spinner("Generating release notes..."):
            st.session_state.last_result = generate_release_notes(
                memory_path=memory_path_input or None,
                repo_path=repo_path_input or None,
            )

    # ---- Render the result ---------------------------------------------------
    result = st.session_state.last_result
    if result is None:
        st.info(
            "Ask a question or click **Generate RELEASE_NOTES.md** to see "
            "the agent's output."
        )
    else:
        st.markdown("### Answer")
        st.markdown(result.get("answer", "_(no answer)_"))

        # If RELEASE_NOTES.md was written on disk, offer a download.
        if RELEASE_NOTES_PATH.exists():
            try:
                notes_bytes = RELEASE_NOTES_PATH.read_bytes()
                st.download_button(
                    "Download RELEASE_NOTES.md",
                    data=notes_bytes,
                    file_name="RELEASE_NOTES.md",
                    mime="text/markdown",
                )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"(Could not read release notes for download: {exc})")

        # Deep-dive evidence, if any.
        deep_dive_result = result.get("deep_dive_result")
        if deep_dive_result:
            with st.expander("Focused repo deep-dive evidence", expanded=False):
                st.json(deep_dive_result)


# ---------- Side column: trace + analysis ------------------------------------

with side_col:
    st.subheader("Agent internals")

    result = st.session_state.last_result
    if result is None:
        st.caption("Run a query to see the trace and analysis here.")
    else:
        with st.expander("Trace", expanded=True):
            for i, step in enumerate(result.get("trace", []), start=1):
                st.markdown(f"{i}. {step}")

        with st.expander("LLM memory analysis (JSON)", expanded=False):
            st.json(result.get("memory_analysis", {}))

        with st.expander("Memory metadata", expanded=False):
            st.json(result.get("memory_metadata", {}))