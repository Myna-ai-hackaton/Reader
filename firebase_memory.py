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

def load_firebase_memory_for_projects(project_names: list[str]) -> dict[str, Any]:
    """
    Adapted for the 'Writer -> prs / developers' schema.
    This specifically looks inside the PRs and Developers subcollections 
    to find data matching the requested GitHub repo.
    """
    if not project_names:
        return {
            "__myna_note__": "No GitHub project is connected.",
            "__connected_projects__": []
        }

    # Normalize project names so "osu-crypto/libOTe" matches "osu-crypto__libOTe" and "libOTe"
    target_projects = []
    for p in project_names:
        target_projects.append(p.replace("/", "__").lower())
        if "/" in p:
            target_projects.append(p.split("/")[-1].lower())

    db = get_firestore_client()
    
    # Structure we will return to the LLM
    scoped_memory = {"Writer": {"prs": {}, "developers": {}}}
    found_data = False

    # Support both root collections based on your code and screenshots
    root_collections = ["myna_ai_info", "myna_ai_info2"]

    for root in root_collections:
        writer_doc = db.collection(root).document("Writer")

        # 1. Grab matching PRs
        prs_ref = writer_doc.collection("prs")
        for doc in prs_ref.list_documents():
            # doc.id looks like "osu-crypto__libOTe_pr_124"
            if any(t in doc.id.lower() for t in target_projects):
                snap = doc.get()
                if snap.exists:
                    scoped_memory["Writer"]["prs"][doc.id] = make_json_safe(snap.to_dict())
                    found_data = True

        # 2. Grab matching Developers
        devs_ref = writer_doc.collection("developers")
        for doc in devs_ref.list_documents():
            snap = doc.get()
            if snap.exists:
                data = snap.to_dict() or {}
                projects = data.get("projects", {})
                
                # Check if the developer contributed to the requested repo
                is_match = False
                for p_name in projects.keys():
                    if any(t in p_name.lower() for t in target_projects):
                        is_match = True
                        break
                        
                if is_match:
                    scoped_memory["Writer"]["developers"][doc.id] = make_json_safe(data)
                    found_data = True

    # If we found nothing, let the LLM know clearly
    if not found_data:
        return {
            "__myna_note__": (
                "Firebase has no documents matching the connected project(s). "
                "The Reader will answer using ONLY the local Git code."
            ),
            "__connected_projects__": list(project_names)
        }

    # Wrap it in the expected root key so the schema guide matches
    return {"myna_ai_info": scoped_memory}
