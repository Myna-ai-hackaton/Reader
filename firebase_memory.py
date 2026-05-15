from __future__ import annotations

from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


FIREBASE_COLLECTION = "summaries"


def get_firestore_client():
    """
    Initialize Firebase using the local service-account JSON file.

    Expected local file:
        Reader/secrets/firebase-service-account.json

    Do NOT commit that file.
    """

    if firebase_admin._apps:
        return firestore.client()

    base_dir = Path(__file__).resolve().parent
    key_path = base_dir / "secrets" / "firebase-service-account.json"

    if not key_path.exists():
        raise FileNotFoundError(
            f"Firebase service-account file not found at: {key_path}"
        )

    cred = credentials.Certificate(str(key_path))
    firebase_admin.initialize_app(cred)

    return firestore.client()


import json
from typing import Any


def make_json_safe(obj: Any) -> Any:
    """
    Convert Firestore/Python objects into JSON-serializable values.

    This does NOT add inspector metadata.
    It only converts values that JSON cannot serialize directly,
    such as Firestore timestamps.
    """

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, set):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    # Firestore timestamp / Python datetime
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # Firestore document reference
    if hasattr(obj, "path"):
        try:
            return obj.path
        except Exception:
            pass

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def dump_collection_raw(collection_ref) -> dict[str, Any]:
    """
    Dump a Firestore collection in clean nested JSON shape.

    Output shape:

    {
      "<document_id>": {
        ...document fields...,
        "<subcollection_id>": {
          "<subdocument_id>": {
            ...subdocument fields...
          }
        }
      }
    }

    Important:
    We use list_documents(), not stream(), so we also see documents that
    have no fields but do have subcollections.
    """

    result: dict[str, Any] = {}

    for doc_ref in collection_ref.list_documents():
        snapshot = doc_ref.get()

        if snapshot.exists:
            doc_data = make_json_safe(snapshot.to_dict() or {})
        else:
            doc_data = {}

        for subcollection_ref in doc_ref.collections():
            doc_data[subcollection_ref.id] = dump_collection_raw(subcollection_ref)

        result[doc_ref.id] = doc_data

    return result


def load_full_firebase_memory() -> dict[str, Any]:
    """
    Load the whole Firebase DB in clean nested JSON shape.

    Example output:

    {
      "myna_ai_info": {
        "Writer": {
          "developers": {
            "AshRider1": {...}
          },
          "prs": {
            "pr_11": {...}
          }
        }
      }
    }

    This is what the Reader Agent should receive as memory.
    """

    db = get_firestore_client()

    memory: dict[str, Any] = {}

    for collection_ref in db.collections():
        memory[collection_ref.id] = dump_collection_raw(collection_ref)

    if not memory:
        raise FileNotFoundError("Firebase database is empty or no collections were found.")

    return memory


def _candidate_project_keys(repo_full_name: str) -> set[str]:
    """
    Given a repo identifier like 'Myna-ai-hackaton/Writer', return the set
    of strings that *could* match a Firestore document ID for that project.

    We try several common conventions so this works even if Writer renames
    its document IDs later:
        - the full slug:              'Myna-ai-hackaton/Writer'
        - just the repo name:         'Writer'
        - lowercase variants:         'writer', 'myna-ai-hackaton/writer'
        - underscore-flattened slug:  'Myna-ai-hackaton__Writer'

    Comparison against actual document IDs is case-insensitive in
    `load_firebase_memory_for_projects`, so callers don't need to think
    about casing.
    """
    candidates: set[str] = set()
    candidates.add(repo_full_name)
    if "/" in repo_full_name:
        owner, repo = repo_full_name.split("/", 1)
        candidates.add(repo)
        candidates.add(f"{owner}__{repo}")
        candidates.add(f"{owner}_{repo}")
    return {c.strip() for c in candidates if c and c.strip()}


def load_firebase_memory_for_projects(
    project_names: list[str],
) -> dict[str, Any]:
    """
    Load Firebase memory scoped to a specific set of projects.

    Why this exists
    ---------------
    The original `load_full_firebase_memory()` dumps every collection and
    every document. When the Reader is connected to repo X, but Firebase
    also holds data for unrelated projects A, B, and C, the LLM sees ALL
    of them and may answer questions about the wrong project. This loader
    filters to only the documents matching the currently connected repos.

    Matching strategy
    -----------------
    For each top-level collection (e.g. `myna_ai_info`):
        - Look at every direct child document ID.
        - Keep documents whose ID matches any candidate key derived from
          the connected repo names (see `_candidate_project_keys`), using
          case-insensitive comparison.
        - Recurse with the matched documents only.

    If no documents match any connected project, returns:
        {
          "__myna_note__": "No Firebase data exists for the connected project(s). ...",
          "__connected_projects__": [...],
          "__firebase_top_level_keys__": [...]  # so the LLM can suggest hints
        }
    This is intentional: better to say "we have nothing on this project"
    than to silently fall back to data about a different project.

    Args:
        project_names: list of identifiers like 'owner/repo' or just 'repo'.
                       Typically the keys of `repo_paths_from_connected(...)`.
                       Empty list → returns the same "no scope" placeholder
                       rather than dumping everything (intentional safety).

    Raises:
        Same as `load_full_firebase_memory` when the DB itself is empty.
    """
    if not project_names:
        return {
            "__myna_note__": (
                "No GitHub project is connected, so Firebase memory was "
                "not loaded. Connect a repository or organization first."
            ),
            "__connected_projects__": [],
        }

    # Build the case-insensitive candidate set across all connected projects.
    candidate_keys: set[str] = set()
    for name in project_names:
        candidate_keys.update(_candidate_project_keys(name))
    lowered_candidates = {c.lower() for c in candidate_keys}

    db = get_firestore_client()

    scoped_memory: dict[str, Any] = {}
    top_level_collection_ids: list[str] = []

    for collection_ref in db.collections():
        top_level_collection_ids.append(collection_ref.id)
        matched_in_collection: dict[str, Any] = {}

        for doc_ref in collection_ref.list_documents():
            if doc_ref.id.lower() not in lowered_candidates:
                continue

            snapshot = doc_ref.get()
            if snapshot.exists:
                doc_data = make_json_safe(snapshot.to_dict() or {})
            else:
                doc_data = {}

            for sub_ref in doc_ref.collections():
                doc_data[sub_ref.id] = dump_collection_raw(sub_ref)

            matched_in_collection[doc_ref.id] = doc_data

        if matched_in_collection:
            scoped_memory[collection_ref.id] = matched_in_collection

    if not scoped_memory:
        # IMPORTANT: do NOT silently fall back to load_full_firebase_memory()
        # here. Returning everything is exactly the bug that caused the
        # Reader to answer about Writer while connected to an unrelated repo.
        return {
            "__myna_note__": (
                "Firebase has no documents matching the connected "
                "project(s). The Reader can still answer questions using "
                "the local repository code via deep-dive, but no "
                "stored PR summaries or developer profiles exist for this "
                "project yet."
            ),
            "__connected_projects__": list(project_names),
            "__firebase_top_level_keys__": top_level_collection_ids,
        }

    scoped_memory["__connected_projects__"] = list(project_names)
    return scoped_memory



