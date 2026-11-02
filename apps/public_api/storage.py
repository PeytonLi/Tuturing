"""Durable request cache, response staging and session records (PRD 8.2, 8.3, 23.1).

One SQLite file holds Person A's public request metadata, staged responses and the committed
transcript. It is not the authoritative learning state; B's event log is. Raw learner text lives
only in the transcript table (access controlled storage), never in logs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.response.hashing import canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  caller TEXT NOT NULL,
  request_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  session_id TEXT,
  fingerprint TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  state TEXT NOT NULL,
  status INTEGER,
  body TEXT,
  plan TEXT,
  plan_kind TEXT,
  staged_response TEXT,
  content_hash TEXT,
  commit_attempted INTEGER NOT NULL DEFAULT 0,
  committed TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (caller, request_id)
);
CREATE INDEX IF NOT EXISTS requests_by_session ON requests (caller, session_id, state);
CREATE TABLE IF NOT EXISTS sessions (
  caller TEXT NOT NULL,
  session_id TEXT NOT NULL,
  public_state TEXT NOT NULL,
  last_response_id TEXT,
  last_plan TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (caller, session_id)
);
CREATE TABLE IF NOT EXISTS transcript (
  caller TEXT NOT NULL,
  session_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  request_id TEXT NOT NULL,
  role TEXT NOT NULL,
  entry TEXT NOT NULL,
  PRIMARY KEY (caller, session_id, request_id, role)
);
CREATE INDEX IF NOT EXISTS transcript_order ON transcript (caller, session_id, seq);
"""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fingerprint(operation: str, path_session_id: str | None, body: Any) -> str:
    """Stable digest of a request's content: operation, path session and canonical JSON body."""
    payload = canonical_json({"operation": operation, "session": path_session_id, "body": body})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


@dataclass
class RequestRecord:
    caller: str
    request_id: str
    operation: str
    session_id: str | None
    fingerprint: str
    trace_id: str
    state: str
    status: int | None
    body: Any
    plan: dict | None
    plan_kind: str | None
    staged_response: dict | None
    content_hash: str | None
    commit_attempted: bool
    committed: dict | None

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RequestRecord:
        return cls(
            caller=row["caller"],
            request_id=row["request_id"],
            operation=row["operation"],
            session_id=row["session_id"],
            fingerprint=row["fingerprint"],
            trace_id=row["trace_id"],
            state=row["state"],
            status=row["status"],
            body=_loads(row["body"]),
            plan=_loads(row["plan"]),
            plan_kind=row["plan_kind"],
            staged_response=_loads(row["staged_response"]),
            content_hash=row["content_hash"],
            commit_attempted=bool(row["commit_attempted"]),
            committed=_loads(row["committed"]),
        )


@dataclass
class SessionRecord:
    caller: str
    session_id: str
    public_state: dict
    last_response_id: str | None
    last_plan: dict | None

    @property
    def state_version(self) -> int:
        return int(self.public_state["state_version"])

    @property
    def ended(self) -> bool:
        return self.public_state["status"] == "ended"


class RequestStore:
    """SQLite-backed durable store. Safe for use from several threads of one process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            if str(self.path) != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # Requests ---------------------------------------------------------------------------------

    def begin_request(
        self,
        caller: str,
        request_id: str,
        operation: str,
        session_id: str | None,
        request_fingerprint: str,
        trace_id: str,
    ) -> RequestRecord:
        """Insert the request if it is new; always return the stored record."""
        now = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO requests (caller, request_id, operation, session_id,"
                " fingerprint, trace_id, state, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'in_progress', ?, ?)",
                (
                    caller,
                    request_id,
                    operation,
                    session_id,
                    request_fingerprint,
                    trace_id,
                    now,
                    now,
                ),
            )
            return self.get_request(caller, request_id)  # type: ignore[return-value]

    def get_request(self, caller: str, request_id: str) -> RequestRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM requests WHERE caller = ? AND request_id = ?", (caller, request_id)
            ).fetchone()
        return None if row is None else RequestRecord.from_row(row)

    def _update(self, caller: str, request_id: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock:
            self._connection.execute(
                f"UPDATE requests SET {assignments} WHERE caller = ? AND request_id = ?",
                (*fields.values(), caller, request_id),
            )

    def stage_plan(self, caller: str, request_id: str, plan: dict, kind: str) -> None:
        self._update(caller, request_id, plan=canonical_json(plan), plan_kind=kind)

    def stage_response(self, caller: str, request_id: str, response: dict) -> None:
        """Durably stage the exact generated response and its content hash (PRD 8.2 step 5)."""
        self._update(
            caller,
            request_id,
            staged_response=canonical_json(response),
            content_hash=response["content_hash"],
        )

    def mark_commit_attempt(self, caller: str, request_id: str) -> None:
        self._update(caller, request_id, commit_attempted=1)

    def stage_commit(self, caller: str, request_id: str, committed: dict) -> None:
        self._update(caller, request_id, committed=canonical_json(committed))

    def complete_request(self, caller: str, request_id: str, status: int, body: Any) -> None:
        self._update(caller, request_id, state="complete", status=status, body=canonical_json(body))

    def pending_turns(self, caller: str, session_id: str) -> list[RequestRecord]:
        """Turn requests of a session that reserved a plan and have not completed."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM requests WHERE caller = ? AND session_id = ? AND state ="
                " 'in_progress' AND plan IS NOT NULL AND plan_kind = 'plan'",
                (caller, session_id),
            ).fetchall()
        return [RequestRecord.from_row(row) for row in rows]

    # Sessions ---------------------------------------------------------------------------------

    def create_session(self, caller: str, session_id: str, public_state: dict) -> None:
        now = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions (caller, session_id, public_state, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?)",
                (caller, session_id, canonical_json(public_state), now, now),
            )

    def get_session(self, caller: str, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE caller = ? AND session_id = ?", (caller, session_id)
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            caller=row["caller"],
            session_id=row["session_id"],
            public_state=json.loads(row["public_state"]),
            last_response_id=row["last_response_id"],
            last_plan=_loads(row["last_plan"]),
        )

    def update_session(
        self,
        caller: str,
        session_id: str,
        public_state: dict,
        *,
        last_response_id: str | None = None,
        last_plan: dict | None = None,
        entries: list[tuple[str, dict]] = (),  # type: ignore[assignment]
    ) -> None:
        """Atomically advance the stored public state and append transcript entries.

        Idempotent: a repeated call for the same committed turn (recovery after a crash between
        staging the commit and completing the request) neither regresses the state version nor
        duplicates transcript entries.
        """
        version = int(public_state["state_version"])
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if last_response_id is None:
                    self._connection.execute(
                        "UPDATE sessions SET public_state = ?, updated_at = ? WHERE caller = ?"
                        " AND session_id = ? AND json_extract(public_state, '$.state_version')"
                        " <= ?",
                        (canonical_json(public_state), utc_now(), caller, session_id, version),
                    )
                else:
                    self._connection.execute(
                        "UPDATE sessions SET public_state = ?, last_response_id = ?, last_plan"
                        " = ?, updated_at = ? WHERE caller = ? AND session_id = ? AND"
                        " json_extract(public_state, '$.state_version') <= ?",
                        (
                            canonical_json(public_state),
                            last_response_id,
                            canonical_json(last_plan),
                            utc_now(),
                            caller,
                            session_id,
                            version,
                        ),
                    )
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS seq FROM transcript WHERE caller = ? AND"
                    " session_id = ?",
                    (caller, session_id),
                ).fetchone()
                seq = int(row["seq"])
                for request_id, entry in entries:
                    seq += 1
                    self._connection.execute(
                        "INSERT OR IGNORE INTO transcript (caller, session_id, seq, request_id,"
                        " role, entry) VALUES (?, ?, ?, ?, ?, ?)",
                        (caller, session_id, seq, request_id, entry["role"], canonical_json(entry)),
                    )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def transcript(self, caller: str, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT entry FROM transcript WHERE caller = ? AND session_id = ? ORDER BY seq",
                (caller, session_id),
            ).fetchall()
        return [json.loads(row["entry"]) for row in rows]
