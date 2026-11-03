"""
Experiment storage using SQLite from the Python standard library.

Why SQLite and not MongoDB for this prototype? It needs no server,
no extra installation, and very little RAM - which matters on an
8 GB laptop - while still storing config, runs and results properly.
The API in main.py does not depend on SQLite details, so the storage
can be changed later without changing the endpoints.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            scenario TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    return conn


def save_experiment(result):
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO experiments (created_at, scenario, result_json) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), result["scenario"],
             json.dumps(result)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_experiments():
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, created_at, scenario, result_json FROM experiments ORDER BY id DESC"
        ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "scenario": r[2],
             "result": json.loads(r[3])}
            for r in rows
        ]
    finally:
        conn.close()
