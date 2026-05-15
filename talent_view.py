"""
talent_view.py
==============

Schema-tolerant reader for the developer/talent data that the Writer agent
stores in Firebase. The Writer's documents have evolved across PRs and
mix several field-name conventions, so this module is intentionally
defensive: it looks at whatever keys are present, gracefully ignores
anything it doesn't recognize, and produces a normalized list of
developer rows that the UI can render with no extra logic.

It does NOT assume a fixed schema. It does NOT call Firebase directly —
it reads the same dict the Reader Agent already loaded via
`load_firebase_memory_for_projects`, so there is no extra DB roundtrip
and nothing to break if the Writer pipeline reshapes the data.

Public API:
    extract_developers(memory: dict) -> list[DeveloperRow]
    aggregate_team_metrics(devs: list[DeveloperRow]) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeveloperRow:
    """One normalized developer record collapsed across all projects."""

    handle: str
    last_active: str | None = None

    # Aggregated counters
    prs_merged: int = 0
    prs_denied: int = 0
    prs_analyzed: int = 0

    # Latest known qualitative scores (0–10 scale in Writer's data)
    quality: float | None = None
    resilience: float | None = None
    docs: float | None = None
    complexity: float | None = None

    # Behavioural profile
    primary_archetype: str | None = None
    archetype_distribution: dict[str, int] = field(default_factory=dict)
    focus_distribution: dict[str, int] = field(default_factory=dict)

    # Skills: {skill_name: {"xp": int, "level": str}}
    skills: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Quality history points: [{"date": "...", "quality": x, "pr_number": n}, ...]
    temporal_history: list[dict[str, Any]] = field(default_factory=list)

    # Projects this developer is known to contribute to (display only).
    projects: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_float(value: Any) -> float | None:
    """Return value as float or None — never raises."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int = 0) -> int:
    """Return value as int, defaulting to `default` — never raises."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _walk_for_developer_dicts(node: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict]]:
    """
    Walk the nested memory dict looking for sub-dicts under a key called
    'developers'. Returns a list of (path_so_far, {developer_handle: data}).
    Tolerates any surrounding structure.
    """
    found: list[tuple[tuple[str, ...], dict]] = []
    if not isinstance(node, dict):
        return found

    for key, value in node.items():
        if key == "developers" and isinstance(value, dict):
            found.append((path, value))
        elif isinstance(value, dict):
            found.extend(_walk_for_developer_dicts(value, path + (str(key),)))
    return found


def _merge_into_row(row: DeveloperRow, data: dict, project_hint: str | None) -> None:
    """
    Pull whatever fields exist out of a single developer's Firebase doc
    into the row. Multiple calls aggregate counters and overwrite latest
    scalar fields. Missing keys are silently skipped.
    """
    if not isinstance(data, dict):
        return

    # github_handle field overrides any path-derived handle
    handle_in_doc = data.get("github_handle")
    if handle_in_doc and isinstance(handle_in_doc, str):
        row.handle = handle_in_doc

    last_active = data.get("last_active")
    if isinstance(last_active, str) and (row.last_active is None or last_active > row.last_active):
        row.last_active = last_active

    # Counters live under python_managed_state.activity_counters (the
    # Brain output) and/or overall_metrics (the legacy aggregate).
    pms = data.get("python_managed_state") or {}
    counters = pms.get("activity_counters") if isinstance(pms, dict) else None
    if isinstance(counters, dict):
        row.prs_merged += _coerce_int(counters.get("merged"))
        row.prs_denied += _coerce_int(counters.get("denied"))

    overall = data.get("overall_metrics") or {}
    if isinstance(overall, dict):
        row.prs_merged = max(row.prs_merged, _coerce_int(overall.get("total_prs_merged")))
        row.prs_denied = max(row.prs_denied, _coerce_int(overall.get("total_prs_denied")))

    # Qualitative metrics: prefer rolling (most recent) over overall.
    rolling = pms.get("rolling_metrics") if isinstance(pms, dict) else None
    if isinstance(rolling, dict):
        row.quality    = _coerce_float(rolling.get("quality"))     or row.quality
        row.resilience = _coerce_float(rolling.get("resilience"))  or row.resilience
        row.docs       = _coerce_float(rolling.get("docs"))        or row.docs
        row.complexity = _coerce_float(rolling.get("complexity"))  or row.complexity
    if isinstance(overall, dict):
        if row.quality is None:
            row.quality = _coerce_float(overall.get("initial_quality_score"))
        if row.resilience is None:
            row.resilience = _coerce_float(overall.get("review_resilience_score"))
        if row.docs is None:
            row.docs = _coerce_float(overall.get("documentation_habit_score"))
        if row.complexity is None:
            row.complexity = _coerce_float(overall.get("average_complexity_score"))

    # Archetype / focus distributions
    arch = data.get("archetype_distribution") or (
        pms.get("archetype_distribution") if isinstance(pms, dict) else None
    )
    if isinstance(arch, dict):
        for k, v in arch.items():
            row.archetype_distribution[str(k)] = row.archetype_distribution.get(str(k), 0) + _coerce_int(v)

    focus = pms.get("focus_distribution") if isinstance(pms, dict) else None
    if isinstance(focus, dict):
        for k, v in focus.items():
            row.focus_distribution[str(k)] = row.focus_distribution.get(str(k), 0) + _coerce_int(v)

    # Primary archetype/focus: pick whichever bucket has the highest count.
    if row.archetype_distribution and not row.primary_archetype:
        row.primary_archetype = max(row.archetype_distribution.items(), key=lambda kv: kv[1])[0]
    if not row.primary_archetype:
        # Fall back to per-project primary_archetype / primary_focus if any.
        projects = data.get("projects") or {}
        if isinstance(projects, dict):
            for proj_data in projects.values():
                if isinstance(proj_data, dict):
                    pa = proj_data.get("primary_archetype") or proj_data.get("primary_focus")
                    if isinstance(pa, str):
                        row.primary_archetype = pa
                        break

    # Skills: merge by max XP per skill name.
    skills_doc = data.get("skills") or (
        pms.get("skills_matrix") if isinstance(pms, dict) else None
    )
    if isinstance(skills_doc, dict):
        for skill_name, payload in skills_doc.items():
            if not isinstance(payload, dict):
                continue
            xp = _coerce_int(payload.get("xp"))
            level = payload.get("level") if isinstance(payload.get("level"), str) else None
            existing = row.skills.get(str(skill_name), {"xp": 0, "level": None})
            if xp > existing.get("xp", 0):
                existing["xp"] = xp
            if level:
                existing["level"] = level
            row.skills[str(skill_name)] = existing

    # Temporal history (quality over time)
    th = pms.get("temporal_history") if isinstance(pms, dict) else None
    if isinstance(th, list):
        for entry in th:
            if isinstance(entry, dict):
                row.temporal_history.append({
                    "date": entry.get("date"),
                    "quality": _coerce_float(entry.get("quality")),
                    "pr_number": entry.get("pr_number"),
                })

    # Project hints
    projects = data.get("projects") or {}
    if isinstance(projects, dict):
        for proj_name in projects.keys():
            if isinstance(proj_name, str) and proj_name not in row.projects:
                row.projects.append(proj_name)
    if project_hint and project_hint not in row.projects:
        row.projects.append(project_hint)


# ---------------------------------------------------------------------------
# Public extraction
# ---------------------------------------------------------------------------

def extract_developers(memory: dict[str, Any]) -> list[DeveloperRow]:
    """
    Walk a (possibly scoped) Firebase memory dict and return one row per
    distinct developer. Same developer appearing under multiple projects
    is merged.

    Returns an empty list if the memory contains no `developers` keys
    anywhere — including the case where memory is the "no Firebase data
    for this project" placeholder.
    """
    if not isinstance(memory, dict):
        return []

    rows: dict[str, DeveloperRow] = {}

    for path, dev_dict in _walk_for_developer_dicts(memory):
        # The closest non-`developers` parent in the path is usually the
        # project document ID — useful as a project hint per developer.
        project_hint = path[-1] if path else None

        for handle, dev_data in dev_dict.items():
            if not isinstance(dev_data, dict):
                continue
            row = rows.get(handle) or DeveloperRow(handle=handle)
            _merge_into_row(row, dev_data, project_hint)
            rows[handle] = row

    # Stable ordering: most recently active first, then alphabetical.
    def _sort_key(r: DeveloperRow):
        return (r.last_active or "", r.handle)

    return sorted(rows.values(), key=_sort_key, reverse=True)


def aggregate_team_metrics(devs: list[DeveloperRow]) -> dict[str, Any]:
    """
    Roll up a developer list into top-line team metrics for the dashboard.
    Returns zeros / empty when the input is empty.
    """
    if not devs:
        return {
            "developer_count": 0,
            "total_prs_merged": 0,
            "total_prs_denied": 0,
            "average_quality": None,
            "archetype_breakdown": {},
        }

    total_merged = sum(d.prs_merged for d in devs)
    total_denied = sum(d.prs_denied for d in devs)
    qualities = [d.quality for d in devs if d.quality is not None]
    avg_quality = round(sum(qualities) / len(qualities), 2) if qualities else None

    archetype_breakdown: dict[str, int] = {}
    for d in devs:
        if d.primary_archetype:
            archetype_breakdown[d.primary_archetype] = archetype_breakdown.get(d.primary_archetype, 0) + 1

    return {
        "developer_count": len(devs),
        "total_prs_merged": total_merged,
        "total_prs_denied": total_denied,
        "average_quality": avg_quality,
        "archetype_breakdown": archetype_breakdown,
    }
