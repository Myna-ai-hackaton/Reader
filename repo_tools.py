"""
repo_tools.py
=============

Controlled, read-only repo tools used by the Reader Agent for focused
codebase investigations.

The important design choice is that the LLM never receives arbitrary shell
access. It may request *what* to inspect (files, commits, symbols, keywords),
but Python decides *how* to inspect it through a small whitelist of safe tools.

Capabilities:
    - Validate that a path is a Git work tree.
    - Summarise repo metadata and tracked file tree.
    - Read full text files or bounded line ranges.
    - Extract lightweight code outlines from Python / JS / TS style files.
    - Search with git grep and read local context around grep hits.
    - Inspect commit patches and file histories.

Design rules:
    - Read-only. No commits, no fetches, no checkouts, no writes.
    - Path-safe. User/LLM-provided paths are resolved inside the repo only.
    - Bounded. Every output has a size budget to protect prompt length.
    - Best-effort. Individual failures become warnings, not fatal errors.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Any

# Hard caps. These make the repo reader safe to call from LLM-driven plans.
MAX_COMMAND_OUTPUT = 12_000
MAX_FILE_CHARS = 20_000
MAX_FILE_RANGE_LINES = 240
MAX_TREE_FILES = 500
MAX_GREP_HITS = 100
MAX_CONTEXT_BLOCKS = 20
MAX_PROJECT_INFO_FILES = 8

# Extensions that are usually worth reading as text/code.
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".scala", ".r", ".sql", ".sh",
    ".bash", ".zsh", ".ps1", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".env", ".md", ".txt", ".rst", ".html",
    ".css", ".scss", ".xml", ".dockerfile",
}

TEXT_FILENAMES = {
    "dockerfile", "makefile", "readme", "readme.md", "requirements.txt",
    "pyproject.toml", "package.json", "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "poetry.lock", "pipfile", "pipfile.lock", "setup.py",
    "setup.cfg", "tox.ini", ".env.example", ".gitignore",
}

IMPORTANT_FILENAMES = {
    "readme", "readme.md", "requirements.txt", "pyproject.toml", "package.json",
    "setup.py", "setup.cfg", "dockerfile", "docker-compose.yml", "compose.yml",
    "app.py", "main.py", "server.py", "index.js", "index.ts", "vite.config.ts",
    "next.config.js", "next.config.ts", "streamlit_app.py",
}

SKIP_DIR_PARTS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__",
    "node_modules", "dist", "build", ".next", ".venv", "venv", "env",
    ".tox", ".idea", ".vscode",
}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """Trim long output and leave an explicit marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[OUTPUT TRUNCATED]"


def _repo_root(repo_path: str) -> Path:
    """Return the Git work-tree root for repo_path, or raise."""
    out = run_git(repo_path, ["rev-parse", "--show-toplevel"])
    return Path(out.strip()).expanduser().resolve()


def _normalise_repo_relative_path(repo_root: Path, file_path: str) -> str:
    """
    Normalise an LLM/user-provided path and ensure it stays inside repo_root.

    Returns a POSIX-style path suitable for Git pathspecs and display.
    """
    if not file_path or "\x00" in file_path:
        raise ValueError("empty or invalid file path")

    raw = file_path.strip().replace("\\", "/")
    # Common LLM output shape: "path/to/file.py:123". Keep only the path.
    if re.search(r":\d+(:\d+)?$", raw):
        raw = raw.rsplit(":", 1)[0]

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()

    try:
        rel = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {file_path}") from exc

    return rel.as_posix()


def _looks_binary_bytes(data: bytes) -> bool:
    """Small binary detector: NUL bytes almost always mean non-text."""
    return b"\x00" in data[:4096]


def _is_probably_text_path(path: str) -> bool:
    p = Path(path)
    name = p.name.lower()
    suffix = p.suffix.lower()
    return name in TEXT_FILENAMES or suffix in TEXT_EXTENSIONS


def _should_skip_tracked_file(path: str) -> bool:
    parts = set(Path(path).parts)
    if parts & SKIP_DIR_PARTS:
        return True
    return False


def _line_numbered(text: str, start_line: int = 1) -> str:
    """Prefix text with stable line numbers for LLM citation/navigation."""
    lines = text.splitlines()
    width = len(str(start_line + max(len(lines) - 1, 0)))
    return "\n".join(
        f"{i:>{width}} | {line}"
        for i, line in enumerate(lines, start=start_line)
    )


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def run_git(repo_path: str, args: list[str]) -> str:
    """
    Run `git <args>` inside `repo_path` and return stdout.

    Raises FileNotFoundError if the path doesn't exist and RuntimeError on a
    nonzero Git exit. Individual public helpers usually catch those errors and
    convert them into warnings.
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


# -----------------------------------------------------------------------------
# Repo validation and overview
# -----------------------------------------------------------------------------

def validate_repo(repo_path: str | None) -> bool:
    """
    Return True iff repo_path points to a Git work tree.

    Designed to never raise because the UI can call this frequently.
    """
    if not repo_path:
        return False
    try:
        out = run_git(repo_path, ["rev-parse", "--is-inside-work-tree"])
        return out.strip().lower() == "true"
    except Exception:  # noqa: BLE001
        return False


def repo_overview(repo_path: str, max_files: int = MAX_TREE_FILES) -> dict[str, Any]:
    """
    Return lightweight repo metadata and a bounded tracked-file tree.

    This is the agent's high-level orientation tool. It lets the model see the
    repo shape before it reasons about exact files.
    """
    warnings: list[str] = []

    try:
        root = _repo_root(repo_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not resolve repo root: {exc}", "warnings": []}

    def _git_or_empty(args: list[str], warning_label: str) -> str:
        try:
            return run_git(str(root), args)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{warning_label}: {exc}")
            return ""

    head = _git_or_empty(["rev-parse", "--short", "HEAD"], "could not read HEAD")
    branch = _git_or_empty(
        ["rev-parse", "--abbrev-ref", "HEAD"], "could not read branch"
    )
    remote = _git_or_empty(
        ["config", "--get", "remote.origin.url"], "could not read origin URL"
    )

    tracked_raw = _git_or_empty(["ls-files"], "could not list tracked files")
    all_files = [p for p in tracked_raw.splitlines() if p and not _should_skip_tracked_file(p)]
    text_files = [p for p in all_files if _is_probably_text_path(p)]
    important_files = [
        p for p in text_files
        if Path(p).name.lower() in IMPORTANT_FILENAMES or Path(p).suffix.lower() in {".md", ".toml", ".json", ".yml", ".yaml"}
    ][:80]

    top_level_dirs = sorted({Path(p).parts[0] for p in all_files if len(Path(p).parts) > 1})
    top_level_files = sorted({p for p in all_files if len(Path(p).parts) == 1})

    return {
        "repo_root": str(root),
        "head": head,
        "branch": branch,
        "origin": remote,
        "total_tracked_files": len(all_files),
        "total_probably_text_files": len(text_files),
        "top_level_dirs": top_level_dirs[:80],
        "top_level_files": top_level_files[:80],
        "important_files": important_files,
        "tracked_files_sample": text_files[:max_files],
        "tracked_files_sample_truncated": len(text_files) > max_files,
        "warnings": warnings,
    }


def list_repo_files(
    repo_path: str,
    *,
    max_files: int = MAX_TREE_FILES,
    include_non_text: bool = False,
    glob_hint: str | None = None,
) -> dict[str, Any]:
    """
    List tracked repo files, optionally filtered by a simple substring/glob hint.

    This is read-only and based on `git ls-files`, so it avoids generated or
    untracked local junk unless Git tracks it.
    """
    root = _repo_root(repo_path)
    raw = run_git(str(root), ["ls-files"])
    files = [p for p in raw.splitlines() if p and not _should_skip_tracked_file(p)]
    if not include_non_text:
        files = [p for p in files if _is_probably_text_path(p)]
    if glob_hint:
        hint = glob_hint.lower().strip("*")
        files = [p for p in files if hint in p.lower()]

    return {
        "files": files[:max_files],
        "total_matches": len(files),
        "truncated": len(files) > max_files,
    }


# -----------------------------------------------------------------------------
# File reading and code outlines
# -----------------------------------------------------------------------------

def read_repo_file(
    repo_path: str,
    file_path: str,
    *,
    max_chars: int = MAX_FILE_CHARS,
    line_numbers: bool = True,
) -> dict[str, Any]:
    """
    Read a tracked text file from the working tree with safety checks.

    Returns structured metadata plus content. It never writes. It rejects paths
    outside the repo and likely-binary files.
    """
    root = _repo_root(repo_path)
    rel = _normalise_repo_relative_path(root, file_path)
    abs_path = root / rel

    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError(f"file does not exist in working tree: {rel}")

    raw = abs_path.read_bytes()
    if _looks_binary_bytes(raw):
        raise ValueError(f"refusing to read likely-binary file: {rel}")

    text = raw.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    shown = _truncate(text, max_chars)

    return {
        "file": rel,
        "size_bytes": len(raw),
        "line_count": text.count("\n") + (0 if text.endswith("\n") else 1),
        "content": _line_numbered(shown) if line_numbers else shown,
        "truncated": truncated,
    }


def read_repo_file_range(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    max_lines: int = MAX_FILE_RANGE_LINES,
) -> dict[str, Any]:
    """
    Read a bounded, line-numbered slice of a text file.

    Useful after grep finds `file.py:137`; the agent can read 120-170 instead
    of wasting prompt budget on the full file.
    """
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    requested_end_line = end_line
    if end_line - start_line + 1 > max_lines:
        end_line = start_line + max_lines - 1

    root = _repo_root(repo_path)
    rel = _normalise_repo_relative_path(root, file_path)
    abs_path = root / rel
    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError(f"file does not exist in working tree: {rel}")

    raw = abs_path.read_bytes()
    if _looks_binary_bytes(raw):
        raise ValueError(f"refusing to read likely-binary file: {rel}")

    lines = raw.decode("utf-8", errors="replace").splitlines()
    actual_start = min(start_line, len(lines) or 1)
    actual_end = min(end_line, len(lines))
    selected = lines[actual_start - 1:actual_end]

    return {
        "file": rel,
        "start_line": actual_start,
        "end_line": actual_end,
        "line_count": len(lines),
        "content": _line_numbered("\n".join(selected), start_line=actual_start),
        "truncated_by_line_budget": end_line < requested_end_line,
    }


def extract_code_outline(repo_path: str, file_path: str, *, max_symbols: int = 80) -> dict[str, Any]:
    """
    Extract a lightweight symbol outline from a code/text file.

    Python gets AST-level class/function names. JS/TS/etc. get conservative
    regex-based functions/classes/exports. This is not a full language server;
    it is a cheap orientation layer for the LLM.
    """
    root = _repo_root(repo_path)
    rel = _normalise_repo_relative_path(root, file_path)
    abs_path = root / rel
    text = abs_path.read_text(encoding="utf-8", errors="replace")
    suffix = Path(rel).suffix.lower()
    symbols: list[dict[str, Any]] = []
    warnings: list[str] = []

    if suffix in {".py", ".pyi"}:
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = (
                        "class" if isinstance(node, ast.ClassDef)
                        else "async function" if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    )
                    symbols.append({"kind": kind, "name": node.name, "line": node.lineno})
        except SyntaxError as exc:
            warnings.append(f"Python AST parse failed: {exc}")

    if not symbols:
        patterns = [
            ("class", r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)"),
            ("function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
            ("function", r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
            ("method", r"^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*{"),
            ("route", r"^\s*app\.(get|post|put|patch|delete)\s*\("),
        ]
        for i, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in patterns:
                m = re.search(pattern, line)
                if m:
                    name = m.group(1)
                    symbols.append({"kind": kind, "name": name, "line": i})
                    break
            if len(symbols) >= max_symbols:
                break

    symbols = sorted(symbols, key=lambda s: int(s.get("line", 0)))[:max_symbols]
    return {
        "file": rel,
        "symbols": symbols,
        "symbols_truncated": len(symbols) >= max_symbols,
        "warnings": warnings,
    }


# -----------------------------------------------------------------------------
# Git history/search commands
# -----------------------------------------------------------------------------

def git_show_commit(repo_path: str, sha: str, max_chars: int = MAX_COMMAND_OUTPUT) -> str:
    """Return a truncated `git show --stat --patch` for a single commit."""
    out = run_git(repo_path, ["show", "--stat", "--patch", "--unified=3", sha])
    return _truncate(out, max_chars)


def git_log_for_file(repo_path: str, file_path: str, max_count: int = 8) -> str:
    """Return recent commits that touched a given file."""
    root = _repo_root(repo_path)
    rel = _normalise_repo_relative_path(root, file_path)
    return run_git(
        str(root),
        [
            "log",
            f"--max-count={max_count}",
            "--pretty=format:%h | %ad | %an | %s",
            "--date=short",
            "--",
            rel,
        ],
    )


def git_grep_term(
    repo_path: str,
    term: str,
    file_paths: list[str] | None = None,
    max_lines: int = 30,
) -> list[dict[str, Any]]:
    """
    Case-insensitive `git grep` for a single term, optionally scoped to files.

    Returns structured hits. No-match/failure returns an empty list because Git
    grep exits nonzero when there are no matches.
    """
    if not term or "\x00" in term:
        return []

    root = _repo_root(repo_path)
    args = ["grep", "-n", "-i", "--", term]

    normalised_paths: list[str] = []
    if file_paths:
        for p in file_paths[:50]:
            try:
                normalised_paths.append(_normalise_repo_relative_path(root, p))
            except Exception:
                continue
    if normalised_paths:
        args.extend(normalised_paths)

    try:
        out = run_git(str(root), args)
    except Exception:  # noqa: BLE001
        return []

    if not out:
        return []

    results: list[dict[str, Any]] = []
    for raw_line in out.splitlines()[:max_lines]:
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


def read_context_around_grep_hits(
    repo_path: str,
    grep_results: list[dict[str, Any]],
    *,
    radius: int = 20,
    max_blocks: int = MAX_CONTEXT_BLOCKS,
) -> list[dict[str, Any]]:
    """Read line-range context around top grep hits."""
    context_blocks: list[dict[str, Any]] = []
    used_ranges: set[tuple[str, int, int]] = set()

    for hit in grep_results:
        if len(context_blocks) >= max_blocks:
            break
        file_path = hit.get("file")
        line = hit.get("line")
        if not file_path or not isinstance(line, int):
            continue
        start = max(1, line - radius)
        end = line + radius
        key = (str(file_path), start, end)
        if key in used_ranges:
            continue
        try:
            block = read_repo_file_range(repo_path, str(file_path), start, end)
            block["matched_term"] = hit.get("term")
            block["matched_line"] = line
            context_blocks.append(block)
            used_ranges.add(key)
        except Exception:
            continue

    return context_blocks


# -----------------------------------------------------------------------------
# Orchestrated deep-dive
# -----------------------------------------------------------------------------

def focused_repo_deep_dive(
    repo_path: str,
    files: list[str],
    commits: list[str],
    keywords: list[str],
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run a bounded set of read-only repo inspections driven by LLM hints.

    What this does now:
        1. Creates a high-level repo overview and tracked-file sample.
        2. Reads likely files directly, with line numbers.
        3. Extracts symbol outlines from likely files.
        4. Shows commit patches for likely commits.
        5. Shows recent history for likely files.
        6. Greps likely keywords, then reads context around top hits.
        7. If the LLM guessed wrong files, falls back to repo-wide grep.

    It still does NOT persist anything. The returned dict is intended for the
    current answer/UI only; Firebase can store it later.
    """
    files = _unique_preserve_order(files)[:25]
    commits = _unique_preserve_order(commits)[:5]
    keywords = _unique_preserve_order(keywords)[:12]
    symbols = _unique_preserve_order(symbols or [])[:12]

    result: dict[str, Any] = {
        "repo_overview": {},
        "files_considered": files,
        "commits_considered": commits,
        "keywords_considered": keywords,
        "symbols_considered": symbols,
        "project_info_reads": [],
        "file_reads": [],
        "file_outlines": [],
        "commit_patches": [],
        "file_logs": [],
        "grep_results": [],
        "grep_context": [],
        "warnings": [],
    }
    warnings: list[str] = result["warnings"]

    # 1. High-level orientation.
    try:
        result["repo_overview"] = repo_overview(repo_path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"repo overview failed: {exc}")

    # 2. Read high-level project info files (README/config/requirements/etc.).
    # These are not persisted; they are only passed into the current LLM prompt.
    try:
        important_files = (result.get("repo_overview") or {}).get("important_files", [])
        for info_file in important_files[:MAX_PROJECT_INFO_FILES]:
            try:
                result["project_info_reads"].append(
                    read_repo_file(repo_path, info_file, max_chars=6_000)
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"project info read failed for {info_file}: {exc}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"project info reads failed: {exc}")

    # 3. Direct file reads and symbol outlines.
    normalised_files: list[str] = []
    try:
        root = _repo_root(repo_path)
        for file_path in files:
            try:
                normalised_files.append(_normalise_repo_relative_path(root, file_path))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"invalid candidate file {file_path!r}: {exc}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"could not normalise candidate files: {exc}")

    for file_path in normalised_files[:12]:
        try:
            result["file_reads"].append(read_repo_file(repo_path, file_path))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"read file failed for {file_path}: {exc}")
        try:
            result["file_outlines"].append(extract_code_outline(repo_path, file_path))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"outline failed for {file_path}: {exc}")

    # 4. Commit patches.
    for sha in commits:
        try:
            patch = git_show_commit(repo_path, sha)
            result["commit_patches"].append({"commit": sha, "patch_excerpt": patch})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"git show failed for {sha}: {exc}")

    # 5. File histories.
    for file_path in normalised_files[:12]:
        try:
            history = git_log_for_file(repo_path, file_path)
            result["file_logs"].append({"file": file_path, "recent_history": history})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"git log failed for {file_path}: {exc}")

    # 6. Keyword/symbol grep. Scope to files first, then broaden if needed.
    search_terms = _unique_preserve_order(keywords + symbols)
    scoped_paths = normalised_files if normalised_files else None
    grep_results: list[dict[str, Any]] = []
    for term in search_terms:
        hits = git_grep_term(repo_path, term, file_paths=scoped_paths)
        grep_results.extend(hits)

    if not grep_results and search_terms and scoped_paths is not None:
        for term in search_terms[:8]:
            hits = git_grep_term(repo_path, term, file_paths=None)
            grep_results.extend(hits)

    result["grep_results"] = grep_results[:MAX_GREP_HITS]

    # 7. If grep found files not already read, read local context around hits.
    try:
        result["grep_context"] = read_context_around_grep_hits(
            repo_path, result["grep_results"], radius=20, max_blocks=MAX_CONTEXT_BLOCKS
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"grep context read failed: {exc}")

    return result


# -----------------------------------------------------------------------------
# Prompt formatting
# -----------------------------------------------------------------------------

def format_deep_dive_for_prompt(deep_dive: dict[str, Any]) -> str:
    """Render a deep-dive result as a readable block for the final LLM prompt."""
    lines: list[str] = []

    files = deep_dive.get("files_considered", [])
    commits = deep_dive.get("commits_considered", [])
    keywords = deep_dive.get("keywords_considered", [])
    symbols = deep_dive.get("symbols_considered", [])

    lines.append("=== Focused Repo Deep-Dive Evidence ===")
    lines.append(f"Files considered:    {files or '(none)'}")
    lines.append(f"Commits considered:  {commits or '(none)'}")
    lines.append(f"Keywords considered: {keywords or '(none)'}")
    lines.append(f"Symbols considered:  {symbols or '(none)'}")
    lines.append("")

    overview = deep_dive.get("repo_overview") or {}
    if overview:
        lines.append("--- Repo overview ---")
        lines.append(f"Root: {overview.get('repo_root', '(unknown)')}")
        lines.append(f"HEAD: {overview.get('head', '(unknown)')}")
        lines.append(f"Branch: {overview.get('branch', '(unknown)')}")
        lines.append(f"Tracked files: {overview.get('total_tracked_files', '(unknown)')}")
        lines.append(f"Top-level dirs: {overview.get('top_level_dirs', [])}")
        important = overview.get("important_files", [])[:40]
        lines.append(f"Important/readable files: {important}")
        sample = overview.get("tracked_files_sample", [])[:120]
        lines.append(f"Tracked readable files sample: {sample}")
        lines.append("")

    project_info_reads = deep_dive.get("project_info_reads", [])
    if project_info_reads:
        lines.append("--- High-level project info files ---")
        for fr in project_info_reads:
            lines.append(f"\n[{fr.get('file', '?')}] lines={fr.get('line_count', '?')} truncated={fr.get('truncated', False)}")
            lines.append(fr.get("content", "(empty)"))
        lines.append("")

    file_outlines = deep_dive.get("file_outlines", [])
    if file_outlines:
        lines.append("--- Code outlines for candidate files ---")
        for outline in file_outlines:
            lines.append(f"\n[{outline.get('file', '?')}] symbols:")
            symbols_list = outline.get("symbols", [])
            if symbols_list:
                for sym in symbols_list:
                    lines.append(
                        f"- line {sym.get('line', '?')}: {sym.get('kind', '?')} {sym.get('name', '?')}"
                    )
            else:
                lines.append("- (no symbols detected)")
        lines.append("")

    file_reads = deep_dive.get("file_reads", [])
    if file_reads:
        lines.append("--- Candidate file contents ---")
        for fr in file_reads:
            lines.append(f"\n[{fr.get('file', '?')}] lines={fr.get('line_count', '?')} truncated={fr.get('truncated', False)}")
            lines.append(fr.get("content", "(empty)"))
        lines.append("")

    commit_patches = deep_dive.get("commit_patches", [])
    if commit_patches:
        lines.append("--- Commit patch excerpts ---")
        for cp in commit_patches:
            lines.append(f"\n[commit {cp.get('commit', '?')}]")
            lines.append(cp.get("patch_excerpt", "(empty)"))
        lines.append("")

    file_logs = deep_dive.get("file_logs", [])
    if file_logs:
        lines.append("--- Recent file histories ---")
        for fl in file_logs:
            lines.append(f"\n[{fl.get('file', '?')}]")
            lines.append(fl.get("recent_history", "(no history)"))
        lines.append("")

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

    grep_context = deep_dive.get("grep_context", [])
    if grep_context:
        lines.append("--- Local context around grep hits ---")
        for block in grep_context:
            lines.append(
                f"\n[{block.get('file', '?')}:{block.get('start_line', '?')}-{block.get('end_line', '?')}; "
                f"matched {block.get('matched_term', '?')} at line {block.get('matched_line', '?')} ]"
            )
            lines.append(block.get("content", "(empty)"))
        lines.append("")

    warnings = deep_dive.get("warnings", [])
    overview_warnings = (deep_dive.get("repo_overview") or {}).get("warnings", [])
    all_warnings = warnings + overview_warnings
    if all_warnings:
        lines.append("--- Warnings ---")
        for w in all_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines).strip()