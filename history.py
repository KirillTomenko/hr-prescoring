"""
History Manager — persists candidate assessment results to JSON.
Provides load / save / clear / export helpers.
"""
import json
import os
from datetime import datetime
from typing import Any

HISTORY_FILE = "history.json"


def _load_raw() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(records: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def append_result(row: dict[str, Any]) -> None:
    """Append one result row (dict) to history."""
    records = _load_raw()
    row["_saved_at"] = datetime.now().isoformat()
    records.append(row)
    _save_raw(records)


def load_history() -> list[dict]:
    """Return all saved history rows."""
    return _load_raw()


def clear_history() -> None:
    """Delete all history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def export_json() -> str:
    """Return history as formatted JSON string for download."""
    return json.dumps(_load_raw(), ensure_ascii=False, indent=2)
