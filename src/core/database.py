"""Signal & Trade History — SQLite audit trail.

Stores every decision with full reasoning chain for:
- Post-trade analysis (was the thesis correct?)
- Evolution fitness tracking (which params worked?)
- Human review (why did the system do this?)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger("database")


class TradingDatabase:
    """SQLite persistence for signals, trades, and evolution history."""

    def __init__(self, db_path: str = "data/trading_bot.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    strength TEXT,
                    confidence REAL,
                    regime TEXT,
                    sentiment_score REAL,
                    alpha_score REAL,
                    reasoning TEXT,
                    active_signals INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    amount REAL,
                    pnl REAL,
                    fees REAL,
                    reason TEXT,
                    hold_candles INTEGER,
                    thesis_correct INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    generation INTEGER,
                    genome_id TEXT,
                    fitness REAL,
                    params TEXT,
                    risk_params TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")

    def store_signal(self, ticker: str, action: str, strength: str,
                     confidence: float, regime: str, reasoning: str,
                     sentiment: float = 0, alpha_score: float = 0,
                     active_signals: int = 0, metadata: dict = None) -> str:
        sig_id = str(uuid.uuid4())[:12]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sig_id, datetime.now(timezone.utc).isoformat(), ticker,
                 action, strength, confidence, regime, sentiment,
                 alpha_score, reasoning, active_signals,
                 json.dumps(metadata or {}))
            )
        return sig_id

    def store_trade(self, ticker: str, side: str, entry_price: float,
                    exit_price: float = None, amount: float = 0,
                    pnl: float = 0, fees: float = 0, reason: str = "",
                    hold_candles: int = 0, thesis_correct: bool = None,
                    signal_id: str = None) -> str:
        trade_id = str(uuid.uuid4())[:12]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trade_id, signal_id, datetime.now(timezone.utc).isoformat(),
                 ticker, side, entry_price, exit_price, amount, pnl, fees,
                 reason, hold_candles,
                 int(thesis_correct) if thesis_correct is not None else None)
            )
        return trade_id

    def store_evolution(self, generation: int, genome_id: str,
                        fitness: float, params: dict, risk_params: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO evolution VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4())[:12], datetime.now(timezone.utc).isoformat(),
                 generation, genome_id, fitness,
                 json.dumps(params), json.dumps(risk_params))
            )

    def get_recent_signals(self, ticker: str = None, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM signals WHERE ticker=? ORDER BY timestamp DESC LIMIT ?",
                    (ticker, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_trades(self, ticker: str = None, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE ticker=? ORDER BY timestamp DESC LIMIT ?",
                    (ticker, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self, ticker: str = None) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            where = "WHERE ticker=?" if ticker else ""
            params = (ticker,) if ticker else ()
            total = conn.execute(f"SELECT COUNT(*) FROM trades {where}", params).fetchone()[0]
            if total == 0:
                return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}
            wins = conn.execute(f"SELECT COUNT(*) FROM trades {where} AND pnl > 0" if ticker else "SELECT COUNT(*) FROM trades WHERE pnl > 0", params if ticker else ()).fetchone()[0]
            total_pnl = conn.execute(f"SELECT COALESCE(SUM(pnl), 0) FROM trades {where}", params).fetchone()[0]
            return {
                "total": total, "wins": wins, "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0,
                "total_pnl": total_pnl,
            }
