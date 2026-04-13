"""SQLite persistence for benchmark results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("results/benchmark.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    model_path      TEXT,
    hf_repo         TEXT,
    model_label     TEXT,
    quantization    TEXT,
    param_count     TEXT,
    ctx_size        INTEGER,
    n_gpu_layers    INTEGER,
    threads         INTEGER,
    batch_size      INTEGER,
    flash_attn      INTEGER,
    extra_args      TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT REFERENCES runs(run_id),
    category                TEXT,
    prompt_name             TEXT,
    prompt_text             TEXT,
    response_text           TEXT,
    prompt_tokens           INTEGER,
    completion_tokens       INTEGER,
    generation_tps          REAL,
    prompt_eval_tps         REAL,
    time_to_first_token_ms  REAL,
    total_time_sec          REAL,
    peak_rss_mb             REAL,
    quality_score           REAL,
    quality_notes           TEXT,
    iteration               INTEGER DEFAULT 1,
    finish_reason           TEXT,
    truncated               INTEGER DEFAULT 0
);
"""


MIGRATIONS = [
    ("finish_reason", "ALTER TABLE results ADD COLUMN finish_reason TEXT"),
    ("truncated", "ALTER TABLE results ADD COLUMN truncated INTEGER DEFAULT 0"),
    ("judge_score", "ALTER TABLE results ADD COLUMN judge_score REAL"),
    ("judge_reason", "ALTER TABLE results ADD COLUMN judge_reason TEXT"),
]


def get_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Run migrations for existing DBs
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(results)").fetchall()}
    for col_name, sql in MIGRATIONS:
        if col_name not in existing_cols:
            conn.execute(sql)
    conn.commit()
    return conn


def save_run(conn: sqlite3.Connection, run_id: str, timestamp: str, config) -> None:
    conn.execute(
        """INSERT INTO runs
           (run_id, timestamp, model_path, hf_repo, model_label, quantization, param_count,
            ctx_size, n_gpu_layers, threads, batch_size, flash_attn, extra_args)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            timestamp,
            config.model_path,
            config.hf_repo,
            config.label,
            config.quantization,
            config.param_count,
            config.ctx_size,
            config.n_gpu_layers,
            config.threads,
            config.batch_size,
            int(config.flash_attn),
            json.dumps(config.extra_args),
        ),
    )
    conn.commit()


def save_result(conn: sqlite3.Connection, result: dict) -> None:
    cols = [
        "run_id", "category", "prompt_name", "prompt_text", "response_text",
        "prompt_tokens", "completion_tokens", "generation_tps", "prompt_eval_tps",
        "time_to_first_token_ms", "total_time_sec", "peak_rss_mb",
        "quality_score", "quality_notes", "iteration",
        "finish_reason", "truncated",
    ]
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    values = tuple(result.get(c) for c in cols)
    conn.execute(f"INSERT INTO results ({col_names}) VALUES ({placeholders})", values)
    conn.commit()


def get_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
    return [dict(r) for r in rows]


def get_results(
    conn: sqlite3.Connection,
    run_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    query = "SELECT * FROM results WHERE 1=1"
    params: list = []
    if run_ids:
        placeholders = ", ".join("?" for _ in run_ids)
        query += f" AND run_id IN ({placeholders})"
        params.extend(run_ids)
    if categories:
        placeholders = ", ".join("?" for _ in categories)
        query += f" AND category IN ({placeholders})"
        params.extend(categories)
    query += " ORDER BY id"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_quality(conn: sqlite3.Connection, result_id: int, score: float, notes: str = "") -> None:
    conn.execute(
        "UPDATE results SET quality_score = ?, quality_notes = ? WHERE id = ?",
        (score, notes, result_id),
    )
    conn.commit()
