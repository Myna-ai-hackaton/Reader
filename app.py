"""
app.py
======

Streamlit UI for the Myna-ai Reader Agent.

UX flow:
    1.  User pastes a GitHub repo URL (and optionally a Personal Access
        Token for private repos).
    2.  We shallow-clone the repo into a local cache and look for
        `.myna/system_memory_index.json` inside it.
    3.  Once connected, the question UI unlocks. The agent uses the
        in-repo memory (or falls back to the mock if the repo doesn't
        have one yet) and can optionally deep-dive into the cloned code
        when the LLM judges memory insufficient.

This module is intentionally thin: it never implements agent or git
logic itself. All real work lives in `reader_agent.py`, `repo_tools.py`,
`reader_memory.py`, and `github_source.py`.

==========================  SECURITY NOTES  ===================================

The GitHub token is the most sensitive value this UI handles. Rules we
follow here:

1.  The token input uses `type="password"` so the value is masked in the
    browser. Browser developer tools can still inspect form values, so
    we additionally:
    - Never write the token to `st.session_state` under a long-lived key;
      we keep it only in `st.session_state["gh_token"]` which lives just
      for the user's browser session.
    - Never include the token in any displayed trace, error message,
      JSON dump, or LLM prompt. `github_source` is responsible for
      redacting it from any git error output before we ever see it here.

2.  We do NOT persist the token to disk. There is no "remember me"
    checkbox and no file write of the token, ever.

3.  When showing connection status we display only the repo slug
    (`owner/repo`), not the URL with the token in it.

4.  Cache directories on disk contain ordinary git clones. After cloning,
    `github_source.connect_repo()` rewrites the `origin` remote URL to a
    token-free public URL, so the token is NOT stored in `.git/config`.

5.  Per-user state isolation: Streamlit gives each browser session its
    own `st.session_state`. Different users hitting the same server do
    not share connected repos or tokens within memory.

==============================================================================
"""

from __future__ import annotations

import streamlit as st

from github_source import (
    ConnectedRepo,
    clear_cache,
    connect_repo,
    disconnect_repo,
    list_cached_repos,
)
from reader_agent import (
    RELEASE_NOTES_PATH,
    generate_release_notes,
    run_reader_agent,
)
from reader_memory import MOCK_MEMORY_PATH


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
    "An AI Git Project Manager. Point it at a GitHub repository — Myna's "
    "memory file is found automatically — then ask PM, QA, or developer "
    "questions. The agent can deep-dive into the cloned code when memory "
    "alone isn't enough."
)

# --- Session state initialization -------------------------------------------

# The connected repo (`ConnectedRepo` object) or None.
if "connected_repo" not in st.session_state:
    st.session_state.connected_repo = None

# The user's last query result.
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# SECURITY: the token lives only in session state and only until the
# Streamlit session ends (closing the tab). We never write it elsewhere.
if "gh_token" not in st.session_state:
    st.session_state.gh_token = ""

# Remembered repo URL across reruns (NOT the token).
if "gh_url" not in st.session_state:
    st.session_state.gh_url = ""


# -----------------------------------------------------------------------------
# Sidebar — connection + advanced
# -----------------------------------------------------------------------------

# These are populated inside the sidebar but read by the main panel.
override_memory: str = ""
override_repo: str = ""

with st.sidebar:
    st.header("Repository")

    connected: ConnectedRepo | None = st.session_state.connected_repo

    # =========================================================================
    # State 1 + 2: not connected (or connecting)
    # =========================================================================
    if connected is None:
        st.caption(
            "Paste a GitHub repo URL or `owner/repo`. The reader will clone "
            "it locally and look for `.myna/system_memory_index.json`."
        )

        url_input = st.text_input(
            "GitHub repository",
            value=st.session_state.gh_url,
            placeholder="github.com/org/repo",
            key="gh_url_input",
        )

        # SECURITY: password field masks the value in the browser. Token
        # is read into a local variable and immediately handed to
        # github_source — it is never logged or echoed.
        token_input = st.text_input(
            "GitHub token (optional, for private repos)",
            value=st.session_state.gh_token,
            type="password",
            help=(
                "A Personal Access Token with `repo` scope. The token "
                "stays on your machine — it's used only to clone, then "
                "scrubbed from the local git config. We never write it "
                "to disk and never send it to the LLM."
            ),
            key="gh_token_input",
        )

        connect_clicked = st.button(
            "Connect Repository",
            type="primary",
            use_container_width=True,
            disabled=not url_input.strip(),
        )

        if connect_clicked:
            # Remember the URL (but not the token) for next time.
            st.session_state.gh_url = url_input.strip()
            st.session_state.gh_token = token_input  # session-only

            status = st.empty()
            try:
                status.info("Resolving repository...")
                # connect_repo handles parse → clone (with token) → locate
                # `.myna/...json`. It raises on any failure with the token
                # already redacted from the message.
                repo = connect_repo(
                    raw_url=url_input.strip(),
                    token=token_input or None,
                )
                st.session_state.connected_repo = repo
                status.success(f"Connected to {repo.ref.slug}.")
                st.rerun()
            except ValueError as exc:
                status.error(f"Could not parse that URL: {exc}")
            except RuntimeError as exc:
                # github_source guarantees the token is redacted from this
                # error message before it reaches us.
                status.error(f"Clone failed: {exc}")
            except Exception as exc:  # noqa: BLE001 — last-resort safety
                status.error(f"Unexpected error: {exc}")

    # =========================================================================
    # State 3: connected
    # =========================================================================
    else:
        # SECURITY: only the safe `slug` is shown — never the URL form
        # that could ever have contained a token.
        st.success(f"✅ Connected: **{connected.ref.slug}**")

        if connected.memory_path is not None:
            rel = connected.memory_path.relative_to(connected.local_path)
            st.caption(f"📦 Memory: `{rel}`")
        else:
            st.warning(
                "📦 No Myna memory file in this repo yet "
                "(`.myna/system_memory_index.json` not found). "
                "Falling back to mock data."
            )

        if connected.ref.branch:
            st.caption(f"🔀 Branch: `{connected.ref.branch}`")
        if connected.head_commit:
            st.caption(f"⬢ HEAD: `{connected.head_commit}`")
        st.caption(f"🗂️  Cache: `{connected.local_path}`")

        col_a, col_b = st.columns(2)
        with col_a:
            refresh_clicked = st.button("🔄 Refresh", use_container_width=True)
        with col_b:
            disconnect_clicked = st.button(
                "✖ Disconnect", use_container_width=True
            )

        if refresh_clicked:
            with st.spinner("Re-fetching repository..."):
                try:
                    # Reuse cache; just fetch latest. Token may have been
                    # cleared, which is fine for public repos.
                    repo = connect_repo(
                        raw_url=f"https://github.com/{connected.ref.slug}",
                        token=st.session_state.gh_token or None,
                        force_refresh=False,
                    )
                    st.session_state.connected_repo = repo
                    st.session_state.last_result = None
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Refresh failed: {exc}")

        if disconnect_clicked:
            try:
                disconnect_repo(connected.ref)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            # SECURITY: scrub the token from session state on disconnect
            # so a subsequent user of the same browser session doesn't
            # inherit it.
            st.session_state.gh_token = ""
            st.session_state.connected_repo = None
            st.session_state.last_result = None
            st.rerun()

    # =========================================================================
    # Advanced overrides (collapsed by default)
    # =========================================================================
    with st.expander("⚙️ Advanced", expanded=False):
        st.caption(
            "Manual overrides for debugging. Leave blank to use the "
            "connected repository."
        )

        override_memory = st.text_input(
            "Override memory JSON path",
            value="",
            help=(
                "Skip the GitHub flow and load a memory JSON file directly "
                "from disk. Useful for testing without a connection."
            ),
        )
        override_repo = st.text_input(
            "Override local Git repo path",
            value="",
            help=(
                "Use a different local Git checkout for deep-dives instead "
                "of the cached clone. Must be an absolute path."
            ),
        )

        st.divider()
        st.caption("**Cache**")
        cached = list_cached_repos()
        if cached:
            for entry in cached:
                st.caption(f"• `{entry['name']}` — {entry['size_mb']} MB")
            if st.button("🗑️ Clear all cached repos"):
                n = clear_cache()
                # If the currently-connected repo got wiped, drop it.
                st.session_state.connected_repo = None
                st.session_state.last_result = None
                st.success(f"Removed {n} cached repo(s).")
                st.rerun()
        else:
            st.caption("(empty)")


# -----------------------------------------------------------------------------
# Resolve effective paths from session state + overrides
# -----------------------------------------------------------------------------

# Manual override wins. Then connected-repo memory. Then mock fallback.
effective_memory_override: str | None = override_memory.strip() or None
effective_connected_memory_path: str | None = None
effective_repo_path: str | None = override_repo.strip() or None

if effective_memory_override is None and connected is not None:
    if connected.memory_path is not None:
        effective_connected_memory_path = str(connected.memory_path)
    # If the connected repo has no memory file, both stay None and
    # `choose_memory_path` falls through to the mock.

if effective_repo_path is None and connected is not None:
    effective_repo_path = str(connected.local_path)


# -----------------------------------------------------------------------------
# Main layout — two columns
# -----------------------------------------------------------------------------

main_col, side_col = st.columns([2, 1])


# ---------- Main column: query + answer --------------------------------------

with main_col:
    st.subheader("Ask the Reader Agent")

    # Only allow questions once a repo is connected OR an override is set.
    can_ask = (
        connected is not None
        or effective_memory_override is not None
        or effective_repo_path is not None
    )

    if not can_ask:
        st.info(
            "👈 Connect a repository in the sidebar to get started — or "
            "use **Advanced → Override memory JSON path** to load a file "
            "directly."
        )
    else:
        # Show a note when we are silently falling back to mock data.
        if (
            connected is not None
            and connected.memory_path is None
            and effective_memory_override is None
        ):
            st.warning(
                f"This repository doesn't have `.myna/system_memory_index.json` "
                f"yet, so the agent is using the bundled mock memory at "
                f"`{MOCK_MEMORY_PATH}`. Once your Writer pipeline produces "
                f"that file inside the repo, the agent will pick it up "
                f"automatically."
            )

        # Quick questions selectbox — populates the textarea below.
        quick_question = st.selectbox(
            "Quick questions (optional)",
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
            ask_clicked = st.button(
                "Ask Agent", type="primary", use_container_width=True
            )
        with btn_col2:
            notes_clicked = st.button(
                "Generate RELEASE_NOTES.md", use_container_width=True
            )

        if ask_clicked:
            if not query.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner("Running the Reader Agent..."):
                    st.session_state.last_result = run_reader_agent(
                        query=query,
                        memory_path=effective_memory_override,
                        repo_path=effective_repo_path,
                        connected_memory_path=effective_connected_memory_path,
                    )

        if notes_clicked:
            with st.spinner("Generating release notes..."):
                st.session_state.last_result = generate_release_notes(
                    memory_path=effective_memory_override,
                    repo_path=effective_repo_path,
                    connected_memory_path=effective_connected_memory_path,
                )

    # ---- Render the result ---------------------------------------------------
    result = st.session_state.last_result
    if result is not None:
        st.markdown("### Answer")
        st.markdown(result.get("answer", "_(no answer)_"))

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

        deep_dive_result = result.get("deep_dive_result")
        if deep_dive_result:
            with st.expander(
                "🔍 Focused repo deep-dive evidence", expanded=False
            ):
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
                # SECURITY: nothing in the trace ever contains the token —
                # github_source guarantees this — but we still display each
                # step verbatim and trust the lower layer's redaction.
                st.markdown(f"{i}. {step}")

        with st.expander("LLM memory analysis (JSON)", expanded=False):
            st.json(result.get("memory_analysis", {}))

        with st.expander("Memory metadata", expanded=False):
            st.json(result.get("memory_metadata", {}))