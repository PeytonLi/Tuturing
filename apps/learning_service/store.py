"""Durable learning transactions. A connection is never used outside its lock."""

import json
import sqlite3
import threading
from contextlib import contextmanager


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, db_path):
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, state_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plans (
                plan_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                status TEXT NOT NULL, plan_json TEXT NOT NULL, state_json TEXT NOT NULL,
                item_json TEXT, reserved_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                content_hash TEXT, response_id TEXT, commit_body_hash TEXT,
                commit_response_json TEXT, canceled_reason TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending ON plans(session_id)
                WHERE status = 'pending';
            CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
                payload_json TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY(session_id, seq));
            CREATE TABLE IF NOT EXISTS idempotency (
                endpoint TEXT NOT NULL, key TEXT NOT NULL, body_hash TEXT NOT NULL,
                response_json TEXT NOT NULL, PRIMARY KEY(endpoint, key));
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except Exception:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()

    def close(self):
        with self.lock:
            self.connection.close()
