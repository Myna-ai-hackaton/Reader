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



