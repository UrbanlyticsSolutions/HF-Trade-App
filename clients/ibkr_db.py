"""
ibkr_db.py — SQLite persistence for IBKR market data snapshots.

Stores historical bars, quotes, option chains, and account summaries
fetched via the ibkr_client.py standalone script.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class IBKRDatabase:
    """Thread-safe SQLite storage for IBKR data."""

    def __init__(self, db_path: str = "ibkr_data.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    # Each thread gets its own connection
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bars (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                sec_type    TEXT NOT NULL DEFAULT 'STK',
                date        TEXT NOT NULL,
                open        REAL, high REAL, low REAL, close REAL,
                volume      REAL, wap REAL, bar_count INTEGER,
                bar_size    TEXT, duration TEXT, what_to_show TEXT,
                expiry      TEXT, strike REAL, right TEXT,
                fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS quotes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                sec_type    TEXT NOT NULL DEFAULT 'STK',
                tick_json   TEXT,
                expiry      TEXT, strike REAL, right TEXT,
                fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS option_chains (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                chain_json  TEXT,
                fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS account_summary (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                account     TEXT,
                summary_json TEXT,
                fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                positions_json TEXT,
                fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

    # ------------------------------------------------------------------
    def save_bars(self, symbol: str, sec_type: str, bars: list,
                  bar_size: str = "", duration: str = "",
                  what_to_show: str = "", expiry: str = "",
                  strike: float = 0.0, right: str = "") -> None:
        if not bars:
            return
        conn = self._conn()
        conn.executemany(
            """INSERT INTO bars
               (symbol, sec_type, date, open, high, low, close,
                volume, wap, bar_count, bar_size, duration,
                what_to_show, expiry, strike, right)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(symbol, sec_type, b.get("date"), b.get("open"),
              b.get("high"), b.get("low"), b.get("close"),
              b.get("volume"), b.get("wap"), b.get("barCount"),
              bar_size, duration, what_to_show, expiry, strike, right)
             for b in bars],
        )
        conn.commit()

    def save_quote(self, symbol: str, sec_type: str, tick: dict,
                   expiry: str = "", strike: float = 0.0,
                   right: str = "") -> None:
        conn = self._conn()
        conn.execute(
            """INSERT INTO quotes (symbol, sec_type, tick_json, expiry, strike, right)
               VALUES (?,?,?,?,?,?)""",
            (symbol, sec_type, json.dumps(tick), expiry, strike, right),
        )
        conn.commit()

    def save_option_chain(self, symbol: str, chain: list) -> None:
        conn = self._conn()
        # Convert sets to lists for JSON
        serializable = []
        for entry in chain:
            e = dict(entry)
            if isinstance(e.get("expirations"), set):
                e["expirations"] = sorted(e["expirations"])
            if isinstance(e.get("strikes"), set):
                e["strikes"] = sorted(float(s) for s in e["strikes"])
            serializable.append(e)
        conn.execute(
            "INSERT INTO option_chains (symbol, chain_json) VALUES (?,?)",
            (symbol, json.dumps(serializable)),
        )
        conn.commit()

    def save_account_summary(self, summary: dict,
                             account: str = "") -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO account_summary (account, summary_json) VALUES (?,?)",
            (account, json.dumps(dict(summary))),
        )
        conn.commit()

    def save_positions(self, positions: dict) -> None:
        conn = self._conn()
        serializable = {}
        for sym, info in positions.items():
            entry = dict(info)
            entry.pop("contract", None)  # Contract objects aren't JSON
            serializable[sym] = entry
        conn.execute(
            "INSERT INTO positions (positions_json) VALUES (?)",
            (json.dumps(serializable),),
        )
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
