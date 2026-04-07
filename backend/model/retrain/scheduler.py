"""
Retraining cooldown scheduler.

Prevents repeated retraining by enforcing a minimum gap (default: 7 days)
between successive retrains.  State is persisted in a small JSON file so
it survives process restarts.

State file: retrain/last_retrain.json
  { "last_retrain": "2025-06-01T14:32:00" }
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

STATE_PATH   = Path(__file__).parent / "last_retrain.json"
COOLDOWN_DAYS = 7


def can_retrain(cooldown_days: int = COOLDOWN_DAYS) -> bool:
    """
    Return True if enough time has passed since the last retrain.
    Always returns True if no prior retrain has been recorded.
    """
    if not STATE_PATH.exists():
        return True
    state     = json.loads(STATE_PATH.read_text())
    last_dt   = datetime.fromisoformat(state["last_retrain"])
    elapsed   = datetime.now() - last_dt
    return elapsed >= timedelta(days=cooldown_days)


def record_retrain() -> None:
    """Persist the current timestamp as the last retrain time."""
    STATE_PATH.write_text(
        json.dumps({"last_retrain": datetime.now().isoformat()}, indent=2)
    )


def time_since_last() -> str | None:
    """Return human-readable elapsed time since last retrain, or None."""
    if not STATE_PATH.exists():
        return None
    state   = json.loads(STATE_PATH.read_text())
    last_dt = datetime.fromisoformat(state["last_retrain"])
    delta   = datetime.now() - last_dt
    return f"{delta.days}d {delta.seconds // 3600}h ago"
