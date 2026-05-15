"""
app.py
======

Streamlit UI for the Myna-ai Reader Agent.

The Reader loads project memory from Firebase and can connect to either:
    - one GitHub repository URL, or
    - a GitHub organization/user URL containing multiple repositories.

When the LLM decides Firebase memory is not enough, the agent deep-dives into
one or more cloned repositories using safe read-only Git/code tools.

This module also supports CLI usage for headless operation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Check if running in CLI mode before importing Streamlit
if __name__ == "__main__" and len(sys.argv) > 1:
    # CLI mode - import only what we need
    from github_source import connect_github_target, disconnect_target, repo_paths_from_connected
    from reader_agent import RELEASE_NOTES_PATH, generate_release_notes, run_reader_agent

    def run_cli():
        """Run the Reader Agent in CLI mode."""
        parser = argparse.ArgumentParser(
            description="Myna-ai Reader Agent - Query project memory and GitHub repositories",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python app.py --github-url https://github.com/owner/repo --query "What PRs exist?"
  python app.py --github-url https://github.com/owner/repo --token ghp_... --query "Summarize recent changes"
  python app.py --github-url https://github.com/owner/repo --release-notes

Requirements:
  - Firebase service account JSON file at: Reader/secrets/firebase-service-account.json
  - OpenAI API key configured via OPENAI_API_KEY environment variable (or OPENAI_BASE_URL for local models)
            """,
        )

        parser.add_argument(
            "--github-url",
            required=True,
            help="GitHub repository or organization URL (e.g., https://github.com/owner/repo)",
        )

        parser.add_argument(
            "--token",
            help="GitHub Personal Access Token (optional, for private repos/orgs)",
        )

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--query",
            help="Question to ask the Reader Agent",
        )
        group.add_argument(
            "--release-notes",
            action="store_true",
            help="Generate RELEASE_NOTES.md from project memory",
        )

        args = parser.parse_args()

        # Validate required credentials before proceeding
        print("Validating credentials...")

        # Check Firebase service account file
        firebase_key_path = Path(__file__).parent / "secrets" / "firebase-service-account.json"
        if not firebase_key_path.exists():
            print(f"ERROR: Firebase service account file not found: {firebase_key_path}")
            print("   Please ensure the Firebase credentials are available.")
            sys.exit(1)

        # Check OpenAI API configuration
        import os
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_base_url and not openai_api_key:
            print("ERROR: OpenAI API configuration missing.")
            print("   Set OPENAI_API_KEY for cloud API or OPENAI_BASE_URL for local models.")
            sys.exit(1)

        print("Credentials validated.")

        try:
            print("Connecting to GitHub target and cloning repo(s)...")
            target = connect_github_target(
                raw_url=args.github_url,
                token=args.token,
            )

            if hasattr(target, 'owner'):
                print(f"Connected to project: {target.owner} ({target.repo_count} repos)")
            else:
                print(f"Connected to repository: {target.ref.slug}")

            repo_paths = repo_paths_from_connected(target)

            if args.release_notes:
                print("Generating release notes...")
                result = generate_release_notes(repo_paths=repo_paths)
                print("\n" + "="*50)
                print("RELEASE NOTES")
                print("="*50)
                print(result["answer"])
                if RELEASE_NOTES_PATH.exists():
                    print(f"\nRelease notes also saved to: {RELEASE_NOTES_PATH}")
            else:
                print(f"Processing query: {args.query}")
                result = run_reader_agent(
                    query=args.query,
                    repo_paths=repo_paths,
                )
                print("\n" + "="*50)
                print("ANSWER")
                print("="*50)
                print(result["answer"])

            # Clean up connection
            try:
                disconnect_target(target)
            except Exception:
                pass  # Ignore cleanup errors

        except ValueError as exc:
            print(f"ERROR: Invalid GitHub URL: {exc}")
            sys.exit(1)
        except RuntimeError as exc:
            print(f"ERROR: GitHub connection failed: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: Unexpected error: {exc}")
            sys.exit(1)

    run_cli()
    sys.exit(0)

# Streamlit mode - import Streamlit and related modules
import streamlit as st

from github_source import (
    ConnectedProject,
    ConnectedRepo,
    clear_cache,
    clear_repo_cache,
    connect_github_target,
    disconnect_target,
    list_cached_repos,
    repo_paths_from_connected,
)
from reader_agent import RELEASE_NOTES_PATH, generate_release_notes, run_reader_agent
from firebase_memory import load_firebase_memory_for_projects
from talent_view import (
    DeveloperRow,
    aggregate_team_metrics,
    extract_developers,
)


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
    "An AI Git Project Manager. It reads Writer memory from Firebase, connects "
    "to a GitHub repository or organization, and answers PM, QA, or developer "
    "questions. When Firebase memory is not enough, it deep-dives into the "
    "cloned code."
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "connected_target" not in st.session_state:
    st.session_state.connected_target = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Session-only token. Never persisted to disk.
if "gh_token" not in st.session_state:
    st.session_state.gh_token = ""

if "gh_url" not in st.session_state:
    st.session_state.gh_url = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


# Keys that hold connection/result state. We clear these on disconnect and
# before connecting to a new target so stale repos cannot leak across runs.
#
# `gh_token` is treated separately because we want Disconnect to scrub it
# (clear_token=True) but a new Connect should preserve whatever the user just
# typed (clear_token=False).
#
# Widget keys (`gh_url_input`, `gh_token_input`) are intentionally NOT in this
# list — Streamlit manages those itself and deleting them mid-script can break
# the next render.
_CONNECTION_STATE_KEYS = (
    "connected_target",
    "last_result",
    "gh_url",
    # Talent-view cache: invalidate whenever the connection changes.
    "firebase_memory_cache",
    # Forward-compatible: clear these too if any older code path ever sets them.
    "connected_project",
    "connected_repo",
    "connected_repo_path",
    "connected_repo_paths",
    "repo_paths",
    "github_target",
    "github_url",
    "last_answer",
    "last_trace",
    "connected_memory_path",
    "effective_connected_memory_path",
)


def clear_connection_state(clear_token: bool = False) -> None:
    """
    Reset every piece of in-memory connection state and clear the last
    answer/result. Must be called:
        - on Disconnect (with clear_token=True), and
        - BEFORE attempting a new Connect (with clear_token=False),
    so a failed or successful new connection cannot inherit stale repo
    paths or stale results from a previous target.

    Safe to call when nothing is connected — every key access is guarded.
    """
    for key in _CONNECTION_STATE_KEYS:
        # Re-initialise the two keys app.py expects to read unconditionally;
        # remove the rest entirely so they can't shadow a fresh value.
        if key == "connected_target":
            st.session_state.connected_target = None
        elif key == "last_result":
            st.session_state.last_result = None
        elif key == "gh_url":
            st.session_state.gh_url = ""
        else:
            st.session_state.pop(key, None)

    if clear_token:
        st.session_state.gh_token = ""


def describe_connected_target(target: ConnectedRepo | ConnectedProject) -> str:
    if isinstance(target, ConnectedProject):
        return f"{target.owner} ({target.repo_count} repo(s))"
    return target.ref.slug


def render_connected_details(target: ConnectedRepo | ConnectedProject) -> None:
    if isinstance(target, ConnectedProject):
        st.success(f"✅ Connected project: **{target.owner}**")
        st.caption("Firebase memory is the primary source. GitHub repos are used for code verification.")
        st.caption(f"📚 Repositories loaded: **{target.repo_count}**")
        for repo in target.repos:
            head = f" — HEAD `{repo.head_commit}`" if repo.head_commit else ""
            st.caption(f"• `{repo.ref.slug}`{head}")
        if target.clone_errors:
            with st.expander("Clone warnings", expanded=False):
                for err in target.clone_errors:
                    st.warning(err)
    else:
        st.success(f"✅ Connected repository: **{target.ref.slug}**")
        st.caption("Firebase memory is the primary source. This repo is used for code verification.")
        if target.ref.branch:
            st.caption(f"🔀 Branch: `{target.ref.branch}`")
        if target.head_commit:
            st.caption(f"⬢ HEAD: `{target.head_commit}`")
        st.caption(f"🗂️ Cache: `{target.local_path}`")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("GitHub project")

    connected = st.session_state.connected_target
    
    if connected is None:
        st.info("No GitHub repository or organization connected.")
        st.caption(
            "Paste a GitHub repository URL or an organization/user URL. Examples: "
            "`https://github.com/Myna-ai-hackaton/Writer` or "
            "`https://github.com/Myna-ai-hackaton`."
        )

        url_input = st.text_input(
            "GitHub URL",
            value=st.session_state.gh_url,
            placeholder="https://github.com/Myna-ai-hackaton",
            key="gh_url_input",
        )

        token_input = st.text_input(
            "GitHub token (optional, for private repos/orgs)",
            value=st.session_state.gh_token,
            type="password",
            help=(
                "A GitHub Personal Access Token. It is used only to list/clone "
                "accessible repos and is never sent to the LLM."
            ),
            key="gh_token_input",
        )

        connect_clicked = st.button(
            "Connect",
            type="primary",
            use_container_width=True,
            disabled=not url_input.strip(),
        )

        if connect_clicked:
            # Capture the user's input BEFORE we wipe state — clear_connection_state
            # resets gh_url/last_result/connected_target, which is exactly what we
            # want, but we still need the URL the user just typed.
            new_url = url_input.strip()
            new_token = token_input  # session-only; not written to disk

            # Step 1: nuke every trace of the previous connection before we
            # touch anything new. This guarantees that if the new connection
            # fails for any reason, the app falls into the disconnected state
            # rather than continuing to use the old target.
            try:
                clear_repo_cache()
            except Exception:  # noqa: BLE001 — never let cleanup crash the UI
                pass
            clear_connection_state(clear_token=False)

            # Step 2: persist the new inputs so the Refresh button (which reads
            # st.session_state.gh_url) can re-use them later.
            st.session_state.gh_url = new_url
            st.session_state.gh_token = new_token

            status = st.empty()
            try:
                status.info("Resolving GitHub target and cloning repo(s)...")
                target = connect_github_target(
                    raw_url=new_url,
                    token=new_token or None,
                )
                st.session_state.connected_target = target
                st.session_state.last_result = None
                status.success(f"Connected to {describe_connected_target(target)}.")
                st.rerun()
            except ValueError as exc:
                # Parse error: state is already clean from Step 1.
                status.error(f"Could not parse that GitHub URL: {exc}")
            except RuntimeError as exc:
                # Clone/network/auth error: state is already clean from Step 1.
                status.error(f"GitHub connection failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                status.error(f"Unexpected error: {exc}")

    else:
        render_connected_details(connected)

        col_a, col_b = st.columns(2)
        with col_a:
            refresh_clicked = st.button("🔄 Refresh", use_container_width=True)
        with col_b:
            disconnect_clicked = st.button("✖ Disconnect", use_container_width=True)

        if refresh_clicked:
            with st.spinner("Refreshing GitHub clone(s)..."):
                try:
                    target = connect_github_target(
                        raw_url=st.session_state.gh_url,
                        token=st.session_state.gh_token or None,
                        force_refresh=False,
                    )
                    st.session_state.connected_target = target
                    st.session_state.last_result = None
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Refresh failed: {exc}")

        if disconnect_clicked:
            # Wipe local clones from disk first. We use clear_repo_cache() (full
            # cache reset) rather than disconnect_target() (which only removes
            # this specific target's directories) so no orphaned clones from
            # crashes / interrupted sessions can survive.
            try:
                clear_repo_cache()
            except Exception:  # noqa: BLE001 — never let cleanup crash the UI
                pass

            # Then scrub every piece of in-memory state, including the token.
            clear_connection_state(clear_token=True)

            st.success("Disconnected and cleared cached repositories.")
            st.rerun()

    with st.expander("⚙️ Cache", expanded=False):
        cached = list_cached_repos()
        if cached:
            for entry in cached:
                st.caption(f"• `{entry['name']}` — {entry['size_mb']} MB")
            if st.button("🗑️ Clear all cached repos"):
                n = clear_cache()
                # If we just wiped clones the active target depends on,
                # reset connection state so the UI doesn't keep pointing
                # at directories that no longer exist on disk.
                clear_connection_state(clear_token=False)
                st.success(f"Removed {n} cached repo(s).")
                st.rerun()
        else:
            st.caption("(empty)")


# -----------------------------------------------------------------------------
# Effective repo paths
# -----------------------------------------------------------------------------

connected = st.session_state.connected_target
repo_paths = repo_paths_from_connected(connected)


# -----------------------------------------------------------------------------
# Main layout — tabs
# -----------------------------------------------------------------------------


def _render_ask_tab(connected, repo_paths) -> None:
    """The original two-column ask UI, now scoped inside its own tab."""
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.subheader("Ask the Reader Agent")

        if connected is None:
            st.info("👈 Connect a GitHub repository or organization in the sidebar to get started.")
        else:
            st.info(
                "Memory source: Firebase. The connected GitHub repo(s) are used only "
                "when the agent needs code-level evidence."
            )

            quick_question = st.selectbox(
                "Quick questions (optional)",
                options=[
                    "",
                    "What information exists in Firebase right now? List projects, developers, and PRs.",
                    "Summarize the latest PR for a project manager.",
                    "What should QA test based on the stored PR summaries?",
                    "Using Firebase and the GitHub repo, verify the most important code changes.",
                    "Which developer skills and roles are currently stored?",
                ],
                index=0,
            )

            query = st.text_area(
                "Your question",
                value=quick_question,
                height=120,
                placeholder=(
                    "e.g. 'Using Firebase and the GitHub repo, verify whether PR 11 "
                    "changed scripts/agent_action.py and scripts/github_service.py.'"
                ),
            )

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                ask_clicked = st.button("Ask Agent", type="primary", use_container_width=True)
            with btn_col2:
                notes_clicked = st.button("Generate RELEASE_NOTES.md", use_container_width=True)

            if ask_clicked:
                if not query.strip():
                    st.warning("Please type a question first.")
                else:
                    # Live trace placeholder above the spinner — fills in
                    # as the agent runs (best-effort visibility into what
                    # the agent is doing while it's working).
                    live_trace = st.empty()
                    live_trace.markdown(
                        "🤔 **Thinking...**\n\n"
                        "_The agent will report each step as it works._"
                    )
                    with st.spinner("Running the Reader Agent..."):
                        result = run_reader_agent(
                            query=query,
                            repo_paths=repo_paths,
                        )
                    st.session_state.last_result = result
                    # Render the final trace once the agent has finished —
                    # the run is synchronous so we cannot stream it live
                    # without restructuring the agent. This still gives
                    # the user the play-by-play they want to see.
                    trace_md = "**Agent steps:**\n\n" + "\n".join(
                        f"`{i+1}.` {step}" for i, step in enumerate(result.get("trace", []))
                    )
                    live_trace.markdown(trace_md)

            if notes_clicked:
                with st.spinner("Generating release notes..."):
                    st.session_state.last_result = generate_release_notes(
                        repo_paths=repo_paths,
                    )

        result = st.session_state.last_result
        if result is not None:
            st.markdown("### Answer")
            st.markdown(result.get("answer", "_(no answer)_"))

            metadata = result.get("memory_metadata", {})
            source = metadata.get("source_path")
            if source:
                st.caption(f"Memory loaded from: `{source}`")

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
                with st.expander("🔍 Repo deep-dive evidence", expanded=False):
                    st.json(deep_dive_result)

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


def _render_team_tab(connected, repo_paths) -> None:
    """
    The Talent Analytics view. Reads scoped Firebase memory once and
    renders normalized developer rows. Fails gracefully when Firebase is
    not configured or contains no data for the connected project.
    """
    st.subheader("Team — Talent Analytics")

    if connected is None:
        st.info("👈 Connect a GitHub repository or organization in the sidebar to see the team view.")
        return

    project_names = list(repo_paths.keys())

    # Fetch scoped Firebase memory. Cache in session state so repeated
    # tab switches don't re-hit Firebase. Invalidate on Connect / Disconnect
    # by piggybacking on `last_result` reset — those buttons already clear
    # `firebase_memory_cache`.
    cache_key = "firebase_memory_cache"
    cached = st.session_state.get(cache_key)
    if cached is None or cached.get("projects") != project_names:
        try:
            with st.spinner("Loading team data from Firebase..."):
                memory = load_firebase_memory_for_projects(project_names)
            st.session_state[cache_key] = {"projects": project_names, "memory": memory, "error": None}
        except Exception as exc:  # noqa: BLE001 — never crash the tab
            st.session_state[cache_key] = {"projects": project_names, "memory": None, "error": str(exc)}

    cached = st.session_state[cache_key]
    if cached.get("error"):
        st.error(f"Could not load Firebase data: {cached['error']}")
        return

    memory = cached["memory"]
    devs = extract_developers(memory if isinstance(memory, dict) else {})

    # If memory is the "no data" placeholder, surface it clearly.
    is_placeholder = isinstance(memory, dict) and "__myna_note__" in memory
    if is_placeholder:
        st.warning(memory["__myna_note__"])
        st.caption(
            "Once the Writer agent runs on this project's PRs, developer "
            "profiles will appear here automatically."
        )
        return

    if not devs:
        st.info("No developer profiles found in Firebase for the connected project(s) yet.")
        return

    # ---- Top-line team metrics --------------------------------------------
    team = aggregate_team_metrics(devs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Developers", team["developer_count"])
    m2.metric("PRs merged", team["total_prs_merged"])
    m3.metric("PRs denied", team["total_prs_denied"])
    m4.metric(
        "Avg. quality",
        team["average_quality"] if team["average_quality"] is not None else "—",
        help="Mean of latest rolling quality scores across developers (0–10 scale).",
    )

    # Archetype breakdown bar
    if team["archetype_breakdown"]:
        with st.expander("Team composition by archetype", expanded=True):
            # Convert dict → 2-col format Streamlit's bar_chart can render.
            bd = team["archetype_breakdown"]
            st.bar_chart({"developers": bd}, height=180)

    st.divider()

    # ---- Per-developer cards ----------------------------------------------
    st.markdown("### Developers")

    for d in devs:
        with st.container(border=True):
            head_cols = st.columns([3, 1])
            with head_cols[0]:
                st.markdown(f"#### {d.handle}")
                if d.primary_archetype:
                    # Render as a colored pill-ish caption.
                    st.caption(f"🏷️ Archetype: **{d.primary_archetype}**")
                if d.projects:
                    st.caption(f"📁 Projects: {', '.join(d.projects)}")
                if d.last_active:
                    # Trim to date for compactness.
                    st.caption(f"🕒 Last active: {d.last_active[:10]}")
            with head_cols[1]:
                st.metric("PRs merged", d.prs_merged)
                if d.prs_denied:
                    st.metric("PRs denied", d.prs_denied)

            # Qualitative metrics row
            q_cols = st.columns(4)
            q_cols[0].metric("Quality", f"{d.quality:.2f}" if d.quality is not None else "—")
            q_cols[1].metric("Resilience", f"{d.resilience:.2f}" if d.resilience is not None else "—")
            q_cols[2].metric("Docs", f"{d.docs:.2f}" if d.docs is not None else "—")
            q_cols[3].metric("Complexity", f"{d.complexity:.2f}" if d.complexity is not None else "—")

            # Quality trend — only meaningful with ≥2 points, but show the
            # single point too so the user knows we are tracking it.
            history = [h for h in d.temporal_history if h.get("quality") is not None]
            if history:
                with st.expander("Quality trend", expanded=False):
                    # Sort by date if present, otherwise leave in stored order.
                    history_sorted = sorted(
                        history, key=lambda h: (h.get("date") or "", h.get("pr_number") or 0)
                    )
                    chart_data = {
                        "quality": [h["quality"] for h in history_sorted],
                    }
                    if len(history_sorted) >= 2:
                        st.line_chart(chart_data, height=180)
                    else:
                        st.caption(
                            f"Only one data point so far: PR #{history_sorted[0].get('pr_number','?')} → "
                            f"quality {history_sorted[0]['quality']:.2f} on {history_sorted[0].get('date','?')}"
                        )

            # Skills
            if d.skills:
                with st.expander(f"Skills ({len(d.skills)})", expanded=False):
                    # Render as a small markdown table for compactness.
                    rows = sorted(d.skills.items(), key=lambda kv: -kv[1].get("xp", 0))
                    st.markdown(
                        "| Skill | Level | XP |\n|---|---|---|\n"
                        + "\n".join(
                            f"| {name} | {info.get('level') or '—'} | {info.get('xp', 0)} |"
                            for name, info in rows
                        )
                    )


# -----------------------------------------------------------------------------
# Render the tabs
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Render the tabs
# -----------------------------------------------------------------------------

ask_tab, team_tab = st.tabs(["💬 Ask", "👥 Team"])

with ask_tab:
    _render_ask_tab(connected, repo_paths)

with team_tab:
    _render_team_tab(connected, repo_paths)