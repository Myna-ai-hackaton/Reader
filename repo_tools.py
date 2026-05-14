"""
repo_tools.py
=============

Controlled Git read tools used by the Reader Agent for *focused* deep-dives.

These tools are only invoked when:
    1. The LLM has decided that the JSON memory alone is insufficient, AND
    2. The user has provided a valid local Git repo path.

Design rules:
    - Read-only. No commits, no fetches, no checkouts, no writes of any kind.
    - The LLM picks WHICH files/commits/keywords to inspect; Python only runs
      the commands. We do not let the LLM choose arbitrary shell strings.
    - Output is always truncated to keep prompt sizes sane.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Hard cap on how much text any single command can contribute to the prompt.
MAX_COMMAND_OUTPUT = 12_000


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def run_git(repo_path: str, args: list[str]) -> str:
    """
    Run `git <args>` inside `repo_path` and return its stdout.

    - Raises FileNotFoundError if the path doesn't exist.
    - Raises RuntimeError (with stderr) on a nonzero exit code.
    - Captures both streams as text with safe error replacement so a stray
      non-UTF-8 byte in a diff cannot crash us.
    """
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo}")

    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def _truncate(text: str, max_chars: int) -> str:
    """Trim long command output, leaving a clear marker so the LLM knows."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[OUTPUT TRUNCATED]"


# -----------------------------------------------------------------------------
# Repo validation
# -----------------------------------------------------------------------------

def validate_repo(repo_path: str | None) -> bool:
    """
    Return True iff `repo_path` points to the root (or inside) a Git work tree.

    Designed to NEVER raise — the UI calls this on every keystroke-style
    update, and any exception becomes a silent "not a repo".
    """
    if not repo_path:
        return False
    try:
        out = run_git(repo_path, ["rev-parse", "--is-inside-work-tree"])
        return out.strip().lower() == "true"
    except Exception:  # noqa: BLE001 — best-effort probe
        return False


# -----------------------------------------------------------------------------
# Individual focused commands
# -----------------------------------------------------------------------------

def git_show_commit(
    repo_path: str,
    sha: str,
    max_chars: int = MAX_COMMAND_OUTPUT,
) -> str:
    """
    Return a truncated `git show` for a single commit.

    Uses --stat + --patch so the LLM sees both the file summary and the
    actual diff hunks (with 3 lines of context).
    """
    out = run_git(
        repo_path,
        ["show", "--stat", "--patch", "--unified=3", sha],
    )
    return _truncate(out, max_chars)


def git_log_for_file(
    repo_path: str,
    file_path: str,
    max_count: int = 8,
) -> str:
    """
    Return the most recent commits that touched a given file.

    Format is one commit per line: `shorthash | YYYY-MM-DD | author | subject`.
    Easy for both humans and the LLM to read.
    """
    return run_git(
        repo_path,
        [
            "log",
            f"--max-count={max_count}",
            "--pretty=format:%h | %ad | %an | %s",
            "--date=short",
            "--",
            file_path,
        ],
    )


def git_grep_term(
    repo_path: str,
    term: str,
    file_paths: list[str] | None = None,
    max_lines: int = 30,
) -> list[dict[str, Any]]:
    """
    Case-insensitive `git grep` for a single term, optionally scoped to a
    pathspec list.

    Returns a list of structured hits:
        {"file": str, "line": int | None, "content": str, "term": str}

    Returns an empty list rather than raising on "no matches" or any grep
    failure — grep returns nonzero when there are no hits, and we treat
    that as a normal "nothing found" result.
    """
    args = ["grep", "-n", "-i", term]
    if file_paths:
        args.append("--")
        args.extend(file_paths)

    try:
        out = run_git(repo_path, args)
    except Exception:  # noqa: BLE001 — "no matches" is a nonzero exit
        return []

    if not out:
        return []

    results: list[dict[str, Any]] = []
    for raw_line in out.splitlines()[:max_lines]:
        # git grep -n output: "<file>:<lineno>:<content>"
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_name, line_str, content = parts
        try:
            line_num: int | None = int(line_str)
        except ValueError:
            line_num = None
        results.append(
            {
                "file": file_name,
                "line": line_num,
                "content": content.strip(),
                "term": term,
            }
        )

    return results


# -----------------------------------------------------------------------------
# Orchestrated deep-dive
# -----------------------------------------------------------------------------

def focused_repo_deep_dive(
    repo_path: str,
    files: list[str],
    commits: list[str],
    keywords: list[str],
) -> dict[str, Any]:
    """
    Run a bounded set of read-only Git commands driven by LLM hints.

    The LLM provides candidate `files`, `commits`, and `keywords`. We:
        - Cap each list to a sensible budget.
        - For each commit:   run `git show` and collect the patch excerpt.
        - For each file:     run `git log` and collect recent history.
        - For each keyword:  run `git grep`, scoped to the candidate files
                             if any were given.
        - If keyword search comes up empty AND files were used to scope it,
          fall back to a repo-wide grep for the first few keywords so we
          don't return nothing just because the LLM guessed wrong files.

    Every failure is captured as a warning string rather than raised, so the
    Agent always gets a usable structured result.
    """
    # Bound the work the LLM can request.
    files = files[:20]
    commits = commits[:5]
    keywords = keywords[:10]

    commit_patches: list[dict[str, Any]] = []
    file_logs: list[dict[str, Any]] = []
    grep_results: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Commits: show patches.
    for sha in commits:
        try:
            patch = git_show_commit(repo_path, sha)
            commit_patches.append({"commit": sha, "patch_excerpt": patch})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"git show failed for {sha}: {exc}")

    # Files: recent history. Cap at 10 to keep prompt size predictable.
    for file_path in files[:10]:
        try:
            history = git_log_for_file(repo_path, file_path)
            file_logs.append({"file": file_path, "recent_history": history})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"git log failed for {file_path}: {exc}")

    # Keywords: grep, scoped to candidate files when we have them.
    scoped_paths = files if files else None
    for term in keywords:
        try:
            hits = git_grep_term(repo_path, term, file_paths=scoped_paths)
            grep_results.extend(hits)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"git grep failed for '{term}': {exc}")

    # Fallback: if scoped grep found nothing, broaden to the whole repo.
    if not grep_results and keywords and scoped_paths is not None:
        for term in keywords[:5]:
            try:
                hits = git_grep_term(repo_path, term, file_paths=None)
                grep_results.extend(hits)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"broad git grep failed for '{term}': {exc}")

    return {
        "files_considered": files,
        "commits_considered": commits,
        "keywords_considered": keywords,
        "commit_patches": commit_patches,
        "file_logs": file_logs,
        "grep_results": grep_results[:100],
        "warnings": warnings,
    }


# -----------------------------------------------------------------------------
# Prompt formatting
# -----------------------------------------------------------------------------

def format_deep_dive_for_prompt(deep_dive: dict[str, Any]) -> str:
    """
    Render a deep-dive result as a single human-readable text block suitable
    for embedding into the final-answer prompt.
    """
    lines: list[str] = []

    files = deep_dive.get("files_considered", [])
    commits = deep_dive.get("commits_considered", [])
    keywords = deep_dive.get("keywords_considered", [])

    lines.append("=== Focused Repo Deep-Dive Evidence ===")
    lines.append(f"Files considered:    {files or '(none)'}")
    lines.append(f"Commits considered:  {commits or '(none)'}")
    lines.append(f"Keywords considered: {keywords or '(none)'}")
    lines.append("")

    # Commit patches.
    commit_patches = deep_dive.get("commit_patches", [])
    if commit_patches:
        lines.append("--- Commit patch excerpts ---")
        for cp in commit_patches:
            lines.append(f"\n[commit {cp.get('commit', '?')}]")
            lines.append(cp.get("patch_excerpt", "(empty)"))
        lines.append("")

    # File histories.
    file_logs = deep_dive.get("file_logs", [])
    if file_logs:
        lines.append("--- Recent file histories ---")
        for fl in file_logs:
            lines.append(f"\n[{fl.get('file', '?')}]")
            lines.append(fl.get("recent_history", "(no history)"))
        lines.append("")

    # Grep hits.
    grep_results = deep_dive.get("grep_results", [])
    if grep_results:
        lines.append("--- Grep results ---")
        for hit in grep_results:
            file_ = hit.get("file", "?")
            line = hit.get("line", "?")
            term = hit.get("term", "?")
            content = hit.get("content", "")
            lines.append(f"{file_}:{line} [{term}] {content}")
        lines.append("")

    # Warnings always shown at the bottom so the LLM knows what failed.
    warnings = deep_dive.get("warnings", [])
    if warnings:
        lines.append("--- Warnings ---")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines).strip()