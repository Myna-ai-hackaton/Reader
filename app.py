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
        print("🔍 Validating credentials...")

        # Check Firebase service account file
        firebase_key_path = Path(__file__).parent / "secrets" / "firebase-service-account.json"
        if not firebase_key_path.exists():
            print(f"❌ Firebase service account file not found: {firebase_key_path}")
            print("   Please ensure the Firebase credentials are available.")
            sys.exit(1)

        # Check OpenAI API configuration
        import os
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_base_url and not openai_api_key:
            print("❌ OpenAI API configuration missing.")
            print("   Set OPENAI_API_KEY for cloud API or OPENAI_BASE_URL for local models.")
            sys.exit(1)

        print("✅ Credentials validated.")

        try:
            print("🔗 Connecting to GitHub target and cloning repo(s)...")
            target = connect_github_target(
                raw_url=args.github_url,
                token=args.token,
            )

            if hasattr(target, 'owner'):
                print(f"✅ Connected to project: {target.owner} ({target.repo_count} repos)")
            else:
                print(f"✅ Connected to repository: {target.ref.slug}")

            repo_paths = repo_paths_from_connected(target)

            if args.release_notes:
                print("📝 Generating release notes...")
                result = generate_release_notes(repo_paths=repo_paths)
                print("\n" + "="*50)
                print("RELEASE NOTES")
                print("="*50)
                print(result["answer"])
                if RELEASE_NOTES_PATH.exists():
                    print(f"\n📄 Release notes also saved to: {RELEASE_NOTES_PATH}")
            else:
                print(f"🤔 Processing query: {args.query}")
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
            print(f"❌ Invalid GitHub URL: {exc}")
            sys.exit(1)
        except RuntimeError as exc:
            print(f"❌ GitHub connection failed: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"❌ Unexpected error: {exc}")
            sys.exit(1)

    run_cli()
    sys.exit(0)

# Streamlit mode - import Streamlit and related modules
import streamlit as st

from github_source import (
    ConnectedProject,
    ConnectedRepo,
    clear_cache,
    connect_github_target,
    disconnect_target,
    list_cached_repos,
    repo_paths_from_connected,
)
from reader_agent import RELEASE_NOTES_PATH, generate_release_notes, run_reader_agent
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
            st.session_state.gh_url = url_input.strip()
            st.session_state.gh_token = token_input

            status = st.empty()
            try:
                status.info("Resolving GitHub target and cloning repo(s)...")
                target = connect_github_target(
                    raw_url=url_input.strip(),
                    token=token_input or None,
                )
                st.session_state.connected_target = target
                st.session_state.last_result = None
                status.success(f"Connected to {describe_connected_target(target)}.")
                st.rerun()
            except ValueError as exc:
                status.error(f"Could not parse that GitHub URL: {exc}")
            except RuntimeError as exc:
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
            try:
                disconnect_target(connected)
            except Exception:  # noqa: BLE001
                pass
            st.session_state.gh_token = ""
            st.session_state.connected_target = None
            st.session_state.last_result = None
            st.rerun()

    with st.expander("⚙️ Cache", expanded=False):
        cached = list_cached_repos()
        if cached:
            for entry in cached:
                st.caption(f"• `{entry['name']}` — {entry['size_mb']} MB")
            if st.button("🗑️ Clear all cached repos"):
                n = clear_cache()
                st.session_state.connected_target = None
                st.session_state.last_result = None
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
# Main layout
# -----------------------------------------------------------------------------

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
                with st.spinner("Running the Reader Agent..."):
                    st.session_state.last_result = run_reader_agent(
                        query=query,
                        repo_paths=repo_paths,
                    )

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


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Check if running with CLI arguments
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Run Streamlit app
        pass  # Streamlit handles the rest automatically
