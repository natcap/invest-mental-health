"""
pipeline/db.py — SQLite progress tracking, source cache, and extracted-records store.

Provides resume capability: the pipeline checks which cities are already complete
and skips them.  Failed cities can be retried up to MAX_RETRIES times.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DB_PATH, log


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PipelineDB:
    """Thin wrapper around an SQLite database for pipeline state."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ── Schema ────────────────────────────────────────────

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pipeline_progress (
                city        TEXT    NOT NULL,
                state       TEXT    NOT NULL,
                stage       TEXT    NOT NULL DEFAULT 'pending',
                status      TEXT    NOT NULL DEFAULT 'pending',
                attempts    INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT,
                error_message TEXT,
                PRIMARY KEY (city, state)
            );

            CREATE TABLE IF NOT EXISTS source_cache (
                url             TEXT PRIMARY KEY,
                city            TEXT,
                state           TEXT,
                content_type    TEXT,
                text_content    TEXT,
                title           TEXT,
                fetch_timestamp TEXT,
                http_status     INTEGER,
                is_official     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS discovered_urls (
                city            TEXT NOT NULL,
                state           TEXT NOT NULL,
                url             TEXT NOT NULL,
                url_type        TEXT NOT NULL DEFAULT 'official',
                added_at        TEXT,
                PRIMARY KEY (city, state, url)
            );

            CREATE TABLE IF NOT EXISTS extracted_records (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                city                TEXT    NOT NULL,
                state               TEXT    NOT NULL,
                source_url          TEXT,
                record_json         TEXT,
                score               REAL,
                verification_status TEXT,
                created_at          TEXT
            );

            CREATE TABLE IF NOT EXISTS city_results (
                city        TEXT NOT NULL,
                state       TEXT NOT NULL,
                result_json TEXT,
                updated_at  TEXT,
                PRIMARY KEY (city, state)
            );

            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id      TEXT PRIMARY KEY,
                started_at  TEXT,
                finished_at TEXT,
                total_cities INTEGER,
                config_json TEXT
            );
            """
        )
        self.conn.commit()

    # ── Progress ──────────────────────────────────────────

    def init_city(self, city: str, state: str):
        """Ensure a row exists for this city."""
        self.conn.execute(
            """INSERT OR IGNORE INTO pipeline_progress
               (city, state, stage, status, attempts, last_updated)
               VALUES (?, ?, 'pending', 'pending', 0, ?)""",
            (city, state, _now()),
        )
        self.conn.commit()

    def get_city_status(self, city: str, state: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM pipeline_progress WHERE city=? AND state=?",
            (city, state),
        ).fetchone()
        return dict(row) if row else None

    def update_city_stage(
        self, city: str, state: str, stage: str, status: str, error: str = ""
    ):
        self.conn.execute(
            """UPDATE pipeline_progress
               SET stage=?, status=?, last_updated=?, error_message=?,
                   attempts = CASE WHEN ? = 'failed' THEN attempts + 1 ELSE attempts END
               WHERE city=? AND state=?""",
            (stage, status, _now(), error, status, city, state),
        )
        self.conn.commit()

    def get_pending_cities(self, max_attempts: int = 5) -> List[dict]:
        rows = self.conn.execute(
            """SELECT * FROM pipeline_progress
               WHERE status != 'complete'
                 AND attempts < ?
               ORDER BY city""",
            (max_attempts,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_complete_cities(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM pipeline_progress WHERE status = 'complete'"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_progress(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM pipeline_progress ORDER BY city"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Source Cache ──────────────────────────────────────

    def cache_source(
        self,
        url: str,
        city: str,
        state: str,
        content_type: str,
        text_content: str,
        title: str = "",
        http_status: int = 200,
        is_official: bool = False,
    ):
        self.conn.execute(
            """INSERT OR REPLACE INTO source_cache
               (url, city, state, content_type, text_content, title,
                fetch_timestamp, http_status, is_official)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                url, city, state, content_type, text_content, title,
                _now(), http_status, int(is_official),
            ),
        )
        self.conn.commit()

    def get_cached_source(self, url: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM source_cache WHERE url=?", (url,)
        ).fetchone()
        return dict(row) if row else None

    # ── Discovered URLs ───────────────────────────────────

    def save_discovered_urls(
        self, city: str, state: str, urls: List[str], url_type: str = "official"
    ):
        now = _now()
        self.conn.executemany(
            """INSERT OR IGNORE INTO discovered_urls
               (city, state, url, url_type, added_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(city, state, u, url_type, now) for u in urls],
        )
        self.conn.commit()

    def get_discovered_urls(
        self, city: str, state: str, url_type: str | None = None
    ) -> List[str]:
        if url_type:
            rows = self.conn.execute(
                """SELECT url FROM discovered_urls
                   WHERE city=? AND state=? AND url_type=?""",
                (city, state, url_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT url FROM discovered_urls WHERE city=? AND state=?",
                (city, state),
            ).fetchall()
        return [r["url"] for r in rows]

    # ── Extracted Records ─────────────────────────────────

    def save_extracted_record(
        self,
        city: str,
        state: str,
        source_url: str,
        record: dict,
        score: float,
        verification_status: str = "unverified",
    ):
        self.conn.execute(
            """INSERT INTO extracted_records
               (city, state, source_url, record_json, score,
                verification_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (city, state, source_url, json.dumps(record), score,
             verification_status, _now()),
        )
        self.conn.commit()

    def get_extracted_records(self, city: str, state: str) -> List[dict]:
        rows = self.conn.execute(
            """SELECT * FROM extracted_records
               WHERE city=? AND state=?
               ORDER BY score DESC""",
            (city, state),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── City Results ──────────────────────────────────────

    def save_city_result(self, city: str, state: str, result: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO city_results
               (city, state, result_json, updated_at)
               VALUES (?, ?, ?, ?)""",
            (city, state, json.dumps(result), _now()),
        )
        self.conn.commit()

    def get_city_result(self, city: str, state: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT result_json FROM city_results WHERE city=? AND state=?",
            (city, state),
        ).fetchone()
        if row:
            return json.loads(row["result_json"])
        return None

    def get_all_results(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT result_json FROM city_results ORDER BY city"
        ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]

    # ── Run Metadata ──────────────────────────────────────

    def start_run(self, run_id: str, total_cities: int, config: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO run_metadata
               (run_id, started_at, total_cities, config_json)
               VALUES (?, ?, ?, ?)""",
            (run_id, _now(), total_cities, json.dumps(config)),
        )
        self.conn.commit()

    def finish_run(self, run_id: str):
        self.conn.execute(
            "UPDATE run_metadata SET finished_at=? WHERE run_id=?",
            (_now(), run_id),
        )
        self.conn.commit()

    # ── Stats ─────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        total = self.conn.execute(
            "SELECT COUNT(*) as n FROM pipeline_progress"
        ).fetchone()["n"]
        complete = self.conn.execute(
            "SELECT COUNT(*) as n FROM pipeline_progress WHERE status='complete'"
        ).fetchone()["n"]
        failed = self.conn.execute(
            "SELECT COUNT(*) as n FROM pipeline_progress WHERE status='failed'"
        ).fetchone()["n"]
        pending = total - complete - failed
        return {
            "total": total,
            "complete": complete,
            "failed": failed,
            "pending": pending,
        }

    def close(self):
        self.conn.close()
