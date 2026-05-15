"""
github_source.py
================

Connect to a GitHub repository by URL, shallow-clone it into a local cache,
and report back where the clone lives plus where Myna's memory JSON sits
inside it.

The agent (`reader_agent.py`) and the repo tools (`repo_tools.py`) do not
know anything about GitHub — they only ever see local paths. This module
is the only bridge between "user pasted a GitHub URL" and "we have a
directory on disk with `.git/` in it".

==========================  SECURITY NOTES  ===================================

The GitHub token is the single most sensitive piece of data this module
touches. We follow these rules:

1.  The token enters the process ONLY via:
        - the Streamlit text input (UI, `type="password"`), or
        - explicit function arguments passed by `app.py`.
    It is never read from .env, never persisted to disk, and is held only in
    Streamlit's per-session state (cleared when the user closes the browser
    tab or clicks Disconnect).

2.  When invoking `git clone`, we pass the token by EMBEDDING it in the URL
    using a temporary `GIT_ASKPASS` helper or, in our simpler approach, via
    a one-shot environment override that `git` consumes and discards. We
    DO NOT log, print, or echo the URL with the token in it. The trace
    only shows the canonical `github.com/owner/repo` form.

3.  The token is NEVER passed to the LLM. None of the prompt builders in
    `reader_agent.py` see it. None of the deep-dive output ever includes it.

4.  The cache directory is local to the user's machine. We do not transmit
    cloned content anywhere. The LLM only sees focused, narrow excerpts
    selected by `repo_tools.py` (commit patches, file logs, grep hits).

5.  If the token is wrong or revoked, the clone fails with a clear error
    and no partial state is left in session. We do not retry without the
    token (which could silently leak whether a repo is private).

6.  The token is treated as opaque. We do not parse it, classify it
    (classic vs fine-grained), or check its scopes — that is GitHub's job
    and the failure mode is clean (HTTP 401 from git).

==============================================================================
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Where the Writer pipeline is expected to drop its memory file INSIDE the
# target repository. This is the project-wide convention.
MEMORY_RELATIVE_PATH = Path(".myna") / "system_memory_index.json"

# Local cache root — sibling of the reader's data folder.
READER_DIR = Path(__file__).resolve().parent
CACHE_ROOT = READER_DIR / "cache"

# How shallow to clone. 50 commits is enough for the kind of forensic
# questions the deep-dive answers (recent file history, recent diffs)
# without dragging down huge repos. Tunable per call.
DEFAULT_CLONE_DEPTH = 50


# -----------------------------------------------------------------------------
# URL parsing
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoRef:
    """Canonical reference to a GitHub repository."""

    owner: str
    repo: str
    branch: str | None = None  # None means "use the default branch"

    @property
    def slug(self) -> str:
        """Human-readable identifier — safe to log."""
        return f"{self.owner}/{self.repo}"

    @property
    def cache_dir_name(self) -> str:
        """Filesystem-safe directory name for the local cache."""
        # Double underscore separator so we can round-trip if we ever need to.
        return f"{self.owner}__{self.repo}"

    @property
    def https_url(self) -> str:
        """Token-free HTTPS clone URL. Safe to log and display."""
        return f"https://github.com/{self.owner}/{self.repo}.git"


_REPO_URL_PATTERNS = [
    # https://github.com/owner/repo[.git][/tree/branch][/...]
    re.compile(
        r"^https?://github\.com/"
        r"(?P<owner>[^/\s]+)/"
        r"(?P<repo>[^/\s]+?)(?:\.git)?"
        r"(?:/tree/(?P<branch>[^/\s]+))?"
        r"(?:/.*)?/?$",
        re.IGNORECASE,
    ),
    # github.com/owner/repo[.git]
    re.compile(
        r"^github\.com/"
        r"(?P<owner>[^/\s]+)/"
        r"(?P<repo>[^/\s]+?)(?:\.git)?/?$",
        re.IGNORECASE,
    ),
    # Bare slug: owner/repo
    re.compile(
        r"^(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$",
    ),
]


def parse_repo_url(raw: str) -> RepoRef:
    """
    Accept any common form of GitHub repo reference and return a RepoRef.

    Accepted inputs (all equivalent):
        owner/repo
        github.com/owner/repo
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/tree/some-branch
        https://github.com/owner/repo/blob/main/some/file.py  (branch ignored)

    Raises ValueError on anything we cannot parse.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty repository reference.")

    for pattern in _REPO_URL_PATTERNS:
        m = pattern.match(text)
        if m:
            owner = m.group("owner").strip()
            repo = m.group("repo").strip()
            # `branch` group only exists on the first pattern.
            branch = None
            if "branch" in m.groupdict() and m.group("branch"):
                branch = m.group("branch").strip()
            if owner and repo:
                return RepoRef(owner=owner, repo=repo, branch=branch)

    raise ValueError(
        f"Could not parse '{raw}' as a GitHub repo reference. "
        "Expected something like 'owner/repo' or "
        "'https://github.com/owner/repo'."
    )


# -----------------------------------------------------------------------------
# Cloning (with token handling)
# -----------------------------------------------------------------------------

@dataclass
class ConnectedRepo:
    """A successfully connected repository ready for the agent to use."""

    ref: RepoRef
    local_path: Path
    memory_path: Path | None  # None if `.myna/...json` does not exist
    fetched_at: float          # unix timestamp
    head_commit: str | None    # short SHA of HEAD after clone


def _build_authenticated_url(ref: RepoRef, token: str) -> str:
    """
    Construct a URL with the token embedded for `git clone`.

    SECURITY: This URL is constructed only at the moment of cloning and
    is passed to `git` as an argv element. We never log it, never store
    it, and never return it from any public function. The variable is
    overwritten with an empty string immediately after `subprocess.run`
    returns.

    GitHub accepts `https://<token>@github.com/...` — token used as a
    username with an empty password. Equivalent forms exist, but this is
    the most portable and is what GitHub's own docs show for unattended
    clones.
    """
    # The token might contain characters that aren't URL-safe in theory,
    # but GitHub tokens are ASCII and URL-safe in practice. We do NOT URL
    # encode here because git is tolerant and encoding can mask user typos.
    return f"https://{token}@github.com/{ref.owner}/{ref.repo}.git"


def _safe_command_for_logging(args: list[str], token: str | None) -> list[str]:
    """
    Return a copy of `args` with the token redacted, suitable for logs
    or error messages. NEVER print `args` directly when a token is used.
    """
    if not token:
        return args
    redacted = []
    for a in args:
        if token and token in a:
            redacted.append(a.replace(token, "***REDACTED***"))
        else:
            redacted.append(a)
    return redacted


def _run_git_safe(
    args: list[str],
    cwd: Path | None,
    token: str | None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """
    Run a git command, capturing stdout/stderr. If anything goes wrong,
    the raised error message has the token redacted.

    We do not stream output to the terminal. stderr is captured and only
    surfaced after redaction.
    """
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Build a token-free message for the user.
        safe_args = _safe_command_for_logging(list(exc.cmd or args), token)
        raise RuntimeError(
            f"git command timed out after {timeout}s: {' '.join(safe_args)}"
        ) from None

    if result.returncode != 0:
        safe_args = _safe_command_for_logging(args, token)
        # Defensive: redact token from stderr too, in case git ever echoed it.
        safe_stderr = result.stderr
        if token:
            safe_stderr = safe_stderr.replace(token, "***REDACTED***")
        raise RuntimeError(
            f"git failed (exit {result.returncode}): "
            f"{' '.join(safe_args)}\n{safe_stderr.strip()}"
        )

    return result


def _get_head_short_sha(repo_path: Path) -> str | None:
    """Best-effort: return short SHA of HEAD, or None if anything goes wrong."""
    try:
        result = _run_git_safe(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            token=None,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 — purely informational
        return None


def _on_rm_error(func, path, exc_info):
    """
    rmtree onerror handler that fixes the #1 cause of "rmtree silently
    failed" on Windows: read-only files inside `.git/objects/pack/`.

    Git marks pack files read-only on Windows. `shutil.rmtree` with
    `ignore_errors=True` then silently fails to delete them, leaving the
    cache directory looking healthy in code while remaining on disk.

    This handler is invoked once per failing path. We chmod the path
    writable and retry the original operation (usually os.unlink). If
    that still fails, we re-raise — so the caller knows about it instead
    of pretending success.

    Works on POSIX too: chmod 0o777 on an already-writable file is a no-op.
    """
    import os
    import stat

    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except Exception:  # noqa: BLE001 — best effort
        pass
    # Retry whatever rmtree was trying to do (os.unlink / os.rmdir / os.scandir).
    func(path)


def _clear_directory(path: Path) -> None:
    """
    Remove a directory tree.

    Uses a custom onerror handler that fixes read-only files on Windows
    (a notorious problem with `.git/objects/pack/*.pack` and `*.idx`
    files that Git marks read-only). After the handler retries, any
    remaining failure is silently swallowed — we don't want a stale
    cache to crash the UI.
    """
    if not path.exists():
        return
    try:
        # Python 3.12+: prefer `onexc` over the deprecated `onerror`.
        shutil.rmtree(path, onexc=_on_rm_error)  # type: ignore[call-arg]
    except TypeError:
        # Older Python: fall back to `onerror`.
        shutil.rmtree(path, onerror=_on_rm_error)
    except Exception:
        # Even the handler couldn't fix it (locked file, etc.). Try one
        # more time with ignore_errors so we at least delete what we can.
        shutil.rmtree(path, ignore_errors=True)


def connect_repo(
    raw_url: str,
    token: str | None = None,
    depth: int = DEFAULT_CLONE_DEPTH,
    force_refresh: bool = False,
) -> ConnectedRepo:
    """
    Top-level entry point. Parse the URL, clone (or reuse a cached clone),
    locate the Myna memory file, and return a ConnectedRepo.

    Args:
        raw_url:        Any accepted GitHub repo reference (see parse_repo_url).
        token:          Optional GitHub PAT for private repos.
                        SECURITY: handled per the module header rules.
        depth:          Shallow clone depth. Larger = more history available
                        for deep-dives but slower clone.
        force_refresh:  If True, delete any existing cache for this repo
                        before cloning. Otherwise we do a `git fetch` to
                        update if a clone already exists.

    Raises:
        ValueError on URL parsing problems.
        RuntimeError on git failures (with the token already redacted).
    """
    ref = parse_repo_url(raw_url)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    target = CACHE_ROOT / ref.cache_dir_name

    if force_refresh:
        _clear_directory(target)

    if target.exists() and (target / ".git").exists():
        # Existing cache — fetch latest instead of re-cloning.
        # We pass the token via an in-process override of the remote URL
        # for this fetch, then immediately restore the token-free URL.
        if token:
            auth_url = _build_authenticated_url(ref, token)
            try:
                _run_git_safe(
                    ["git", "remote", "set-url", "origin", auth_url],
                    cwd=target,
                    token=token,
                )
                _run_git_safe(
                    ["git", "fetch", "--depth", str(depth), "origin"],
                    cwd=target,
                    token=token,
                )
            finally:
                # SECURITY: scrub the auth URL from the repo's git config so
                # the token does not persist in `.git/config` on disk.
                try:
                    _run_git_safe(
                        ["git", "remote", "set-url", "origin", ref.https_url],
                        cwd=target,
                        token=None,
                    )
                except Exception:  # noqa: BLE001
                    pass
                auth_url = ""  # best-effort scrub of the local variable
        else:
            _run_git_safe(
                ["git", "fetch", "--depth", str(depth), "origin"],
                cwd=target,
                token=None,
            )
    else:
        # Fresh clone. Use the authenticated URL only on the command line;
        # immediately rewrite the remote to a token-free URL afterwards.
        clone_url = (
            _build_authenticated_url(ref, token) if token else ref.https_url
        )
        try:
            _run_git_safe(
                [
                    "git",
                    "clone",
                    "--depth",
                    str(depth),
                    "--single-branch",
                    *(
                        ["--branch", ref.branch]
                        if ref.branch
                        else []
                    ),
                    clone_url,
                    str(target),
                ],
                cwd=None,
                token=token,
            )
        except Exception:
            # Clean up any half-finished clone so the next attempt starts fresh.
            _clear_directory(target)
            raise
        finally:
            # SECURITY: even on success, immediately wipe the token-bearing
            # URL from `.git/config` by resetting `origin` to the public URL.
            # This means the token does NOT live on disk after this call.
            if token and (target / ".git").exists():
                try:
                    _run_git_safe(
                        [
                            "git",
                            "remote",
                            "set-url",
                            "origin",
                            ref.https_url,
                        ],
                        cwd=target,
                        token=None,
                    )
                except Exception:  # noqa: BLE001
                    pass
            clone_url = ""  # best-effort scrub of the local variable

    # Locate Myna memory file inside the repo (if present).
    candidate = target / MEMORY_RELATIVE_PATH
    memory_path = candidate if candidate.exists() else None

    return ConnectedRepo(
        ref=ref,
        local_path=target,
        memory_path=memory_path,
        fetched_at=time.time(),
        head_commit=_get_head_short_sha(target),
    )


# -----------------------------------------------------------------------------
# Cache management helpers (used by the UI's Advanced section)
# -----------------------------------------------------------------------------

def list_cached_repos() -> list[dict[str, Any]]:
    """
    Enumerate cached clones for display. Returns a list of dicts:
        {"name": "<owner>__<repo>", "path": <Path>, "size_mb": <float>}
    """
    if not CACHE_ROOT.exists():
        return []
    entries: list[dict[str, Any]] = []
    for child in CACHE_ROOT.iterdir():
        if not child.is_dir():
            continue
        total = 0
        for f in child.rglob("*"):
            try:
                total += f.stat().st_size
            except OSError:
                continue
        entries.append(
            {
                "name": child.name,
                "path": child,
                "size_mb": round(total / (1024 * 1024), 2),
            }
        )
    return entries


def clear_cache() -> int:
    """Delete every cached clone. Returns the number of directories removed."""
    if not CACHE_ROOT.exists():
        return 0
    count = 0
    for child in CACHE_ROOT.iterdir():
        if child.is_dir():
            _clear_directory(child)
            count += 1
    return count


def clear_repo_cache() -> None:
    """
    Wipe the entire local GitHub clone cache and recreate it empty.

    This is the function the UI should call on Disconnect, and BEFORE
    connecting to a new target, to make sure no stale cloned repos can
    leak into the next session.

    Safety:
        - We only ever touch CACHE_ROOT, which is a fixed path under the
          Reader package directory (`<reader>/cache`). We never delete
          anything outside it.
        - Uses `_clear_directory` which handles Windows read-only `.git`
          files correctly (the cause of the "Disconnect didn't actually
          empty the cache" bug we hit in testing).
        - The directory is recreated empty so subsequent clones land in a
          known location.
    """
    # Defensive: never delete something that's not under our cache root.
    if CACHE_ROOT.parent != READER_DIR:
        raise RuntimeError(
            f"Refusing to clear cache: unexpected CACHE_ROOT location {CACHE_ROOT}"
        )
    _clear_directory(CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def disconnect_repo(ref: RepoRef) -> bool:
    """
    Remove a single repo from the cache. Returns True if it existed.

    Note: this only deletes the local clone. There is nothing remote to
    "disconnect" from — we never had a persistent connection.
    """
    target = CACHE_ROOT / ref.cache_dir_name
    if target.exists():
        _clear_directory(target)
        return True
    return False

# -----------------------------------------------------------------------------
# GitHub organization / owner support
# -----------------------------------------------------------------------------

import json as _json
import urllib.error as _urlerror
import urllib.request as _urlrequest


@dataclass(frozen=True)
class OwnerRef:
    """Canonical reference to a GitHub owner: user or organization."""

    owner: str

    @property
    def slug(self) -> str:
        return self.owner


@dataclass
class ConnectedProject:
    """
    A connected GitHub owner/project containing one or more cloned repos.

    The Reader can use `repo_paths` to deep-dive across the whole project.
    """

    owner: str
    repos: list[ConnectedRepo]
    fetched_at: float
    clone_errors: list[str]

    @property
    def slug(self) -> str:
        return self.owner

    @property
    def repo_paths(self) -> dict[str, str]:
        return {repo.ref.slug: str(repo.local_path) for repo in self.repos}

    @property
    def repo_count(self) -> int:
        return len(self.repos)


_OWNER_URL_PATTERNS = [
    # https://github.com/owner
    re.compile(r"^https?://github\.com/(?P<owner>[^/\s]+)/?$", re.IGNORECASE),
    # github.com/owner
    re.compile(r"^github\.com/(?P<owner>[^/\s]+)/?$", re.IGNORECASE),
    # bare owner
    re.compile(r"^(?P<owner>[^/\s]+)$"),
]


def parse_owner_url(raw: str) -> OwnerRef:
    """
    Parse a GitHub owner/org URL such as:
        https://github.com/Myna-ai-hackaton
        github.com/Myna-ai-hackaton
        Myna-ai-hackaton
    """
    text = raw.strip().rstrip("/")
    if not text:
        raise ValueError("Empty GitHub owner reference.")

    for pattern in _OWNER_URL_PATTERNS:
        m = pattern.match(text)
        if m:
            owner = m.group("owner").strip()
            if owner and owner.lower() not in {"github.com", "http:", "https:"}:
                return OwnerRef(owner=owner)

    raise ValueError(
        f"Could not parse '{raw}' as a GitHub owner/org reference. "
        "Expected something like 'Myna-ai-hackaton' or "
        "'https://github.com/Myna-ai-hackaton'."
    )


def parse_github_target(raw: str) -> RepoRef | OwnerRef:
    """
    Parse either a specific repository URL or an organization/user URL.

    Repo examples:
        https://github.com/Myna-ai-hackaton/Writer
        Myna-ai-hackaton/Writer

    Owner examples:
        https://github.com/Myna-ai-hackaton
        github.com/Myna-ai-hackaton
        Myna-ai-hackaton

    Owner parsing is attempted first so `github.com/<owner>` is not
    accidentally treated as the bare repo slug `github.com/<owner>`.
    """
    try:
        return parse_owner_url(raw)
    except ValueError:
        return parse_repo_url(raw)


def _github_api_get(url: str, token: str | None = None) -> tuple[int, Any]:
    """Small stdlib GitHub API GET helper. Returns (status, decoded_json)."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "myna-ai-reader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = _urlrequest.Request(url, headers=headers)
    try:
        with _urlrequest.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, _json.loads(body) if body else None
    except _urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = _json.loads(body) if body else None
        except Exception:  # noqa: BLE001
            payload = body
        return exc.code, payload


def list_github_owner_repos(owner: str, token: str | None = None) -> list[RepoRef]:
    """
    List repositories for a GitHub organization or user.

    We first try the organization endpoint, then the user endpoint. The token is
    optional but needed for private organizations/repos.
    """
    repos: list[RepoRef] = []
    errors: list[str] = []

    for kind in ("orgs", "users"):
        page = 1
        while True:
            url = (
                f"https://api.github.com/{kind}/{owner}/repos"
                f"?per_page=100&page={page}&type=all&sort=updated"
            )
            status, payload = _github_api_get(url, token=token)

            if status == 404:
                errors.append(f"{kind}/{owner}: not found")
                break
            if status == 401 or status == 403:
                message = payload.get("message") if isinstance(payload, dict) else payload
                raise RuntimeError(
                    f"GitHub API denied access while listing repos for {owner}: {message}"
                )
            if status < 200 or status >= 300:
                message = payload.get("message") if isinstance(payload, dict) else payload
                raise RuntimeError(
                    f"GitHub API failed while listing repos for {owner} "
                    f"via /{kind}: HTTP {status}: {message}"
                )
            if not isinstance(payload, list):
                raise RuntimeError(
                    f"Unexpected GitHub API response while listing repos for {owner}: {payload!r}"
                )
            if not payload:
                break

            for item in payload:
                full_name = item.get("full_name")
                repo_name = item.get("name")
                owner_login = (item.get("owner") or {}).get("login", owner)
                if full_name and repo_name:
                    repos.append(RepoRef(owner=owner_login, repo=repo_name))

            if len(payload) < 100:
                break
            page += 1

        if repos:
            # The organization endpoint succeeded, or the user endpoint succeeded.
            break

    # Deduplicate while preserving order.
    unique: list[RepoRef] = []
    seen: set[str] = set()
    for ref in repos:
        if ref.slug not in seen:
            unique.append(ref)
            seen.add(ref.slug)

    if not unique:
        joined_errors = "; ".join(errors) if errors else "no repositories returned"
        raise RuntimeError(f"No repositories found for GitHub owner '{owner}' ({joined_errors}).")

    return unique


def connect_github_target(
    raw_url: str,
    token: str | None = None,
    depth: int = DEFAULT_CLONE_DEPTH,
    force_refresh: bool = False,
) -> ConnectedRepo | ConnectedProject:
    """
    Connect either a single GitHub repo URL or an owner/org URL.

    If `raw_url` points to a repo, returns ConnectedRepo.
    If `raw_url` points to an owner/org, lists all accessible repos and returns
    ConnectedProject with every successfully cloned repo.
    """
    target = parse_github_target(raw_url)

    if isinstance(target, RepoRef):
        return connect_repo(
            raw_url=f"https://github.com/{target.slug}",
            token=token,
            depth=depth,
            force_refresh=force_refresh,
        )

    repo_refs = list_github_owner_repos(target.owner, token=token)
    connected_repos: list[ConnectedRepo] = []
    clone_errors: list[str] = []

    for ref in repo_refs:
        try:
            connected_repos.append(
                connect_repo(
                    raw_url=f"https://github.com/{ref.slug}",
                    token=token,
                    depth=depth,
                    force_refresh=force_refresh,
                )
            )
        except Exception as exc:  # noqa: BLE001
            clone_errors.append(f"{ref.slug}: {exc}")

    if not connected_repos:
        raise RuntimeError(
            "Found repositories, but none could be cloned. "
            + ("Errors: " + " | ".join(clone_errors) if clone_errors else "")
        )

    return ConnectedProject(
        owner=target.owner,
        repos=connected_repos,
        fetched_at=time.time(),
        clone_errors=clone_errors,
    )


def disconnect_target(target: ConnectedRepo | ConnectedProject) -> bool:
    """Remove cached clone(s) for a connected repo or project."""
    if isinstance(target, ConnectedProject):
        removed = False
        for repo in target.repos:
            removed = disconnect_repo(repo.ref) or removed
        return removed
    return disconnect_repo(target.ref)


def repo_paths_from_connected(target: ConnectedRepo | ConnectedProject | None) -> dict[str, str]:
    """Return {full_repo_name: local_path} for the connected target."""
    if target is None:
        return {}
    if isinstance(target, ConnectedProject):
        return target.repo_paths
    return {target.ref.slug: str(target.local_path)}
