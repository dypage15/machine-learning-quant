"""
SQLite database layer — schema, insert helpers, query helpers.
"""
import sqlite3, json, logging, os
from datetime import datetime
from contextlib import contextmanager
from .config import DB_PATH, FEATURE_COLS

log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────
# NOTE: WAL pragma is set separately in init_db() — not inside executescript()
# because WAL fails on Windows network/mounted filesystems inside transactions.
SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    ts          INTEGER NOT NULL,           -- Unix timestamp (bar open)
    dt          TEXT    NOT NULL,           -- ISO datetime string
    open        REAL, high REAL, low REAL, close REAL, volume REAL,
    atr14       REAL,
    log_ret     REAL,
    UNIQUE(symbol, ts)
);

CREATE TABLE IF NOT EXISTS features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL UNIQUE,    -- MNQ bar timestamp
    pca_zscore  REAL,
    atr_ratio   REAL,
    body_ratio  REAL,
    upper_wick  REAL,
    lower_wick  REAL,
    log_ret_mnq REAL,
    log_ret_es  REAL,
    log_ret_ym  REAL,
    log_ret_zn  REAL,
    rsi14       REAL,
    bb_pos      REAL,
    vol_ratio   REAL,
    hour_sin    REAL,
    hour_cos    REAL,
    dow_sin     REAL,
    dow_cos     REAL,
    beta_es     REAL,
    trend_align REAL,
    candle_grade TEXT,   -- A B C D E F
    grade_score  REAL
);

CREATE TABLE IF NOT EXISTS trade_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_ts        INTEGER NOT NULL,
    exit_ts         INTEGER,
    direction       INTEGER,    -- 1=long  -1=short
    entry_price     REAL,
    exit_price      REAL,
    sl_price        REAL,
    tp_price        REAL,
    exit_reason     TEXT,       -- 'TP' | 'SL' | 'MH' | 'EOD'
    pnl             REAL,
    bars_held       INTEGER,
    pca_zscore_entry REAL,
    candle_grade    TEXT,
    lorentzian_vote REAL,       -- -1..1 at entry
    lstm_conf       REAL,       -- 0..1 at entry
    actual_outcome  INTEGER,    -- 1=win  0=loss  (filled after exit)
    session_date    TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    target_ts   INTEGER,        -- bar this prediction is for
    lorentzian  REAL,           -- vote fraction  -1..1
    lstm_dir    INTEGER,        -- -1 / 0 / 1
    lstm_conf   REAL,           -- 0..1
    ensemble    REAL,           -- weighted combination
    signal_text TEXT,           -- 'LONG' | 'SHORT' | 'NEUTRAL'
    grade       TEXT,           -- expected candle grade
    fired       INTEGER DEFAULT 0  -- 1 once bar is known
);

CREATE TABLE IF NOT EXISTS model_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at  TEXT    NOT NULL,
    model_type  TEXT,           -- 'lorentzian' | 'lstm'
    n_samples   INTEGER,
    val_accuracy REAL,
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_candles_ts     ON candles(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_features_ts    ON features(ts);
CREATE INDEX IF NOT EXISTS idx_trades_entry   ON trade_results(entry_ts);
CREATE INDEX IF NOT EXISTS idx_preds_ts       ON predictions(target_ts);
"""

# ── Connection ─────────────────────────────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    # Ensure the directory exists before SQLite tries to create the file
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Create tables using executescript (no WAL here — unreliable on Windows mounts)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

    # Try WAL mode separately for better concurrent read performance.
    # Falls back silently to DELETE mode on filesystems that don't support it.
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.close()
    except Exception:
        pass  # DELETE mode is fine for single-user use

    # Migrations: add columns that may be missing from older DB versions
    _migrate(DB_PATH)

    log.info(f"Database initialised at {DB_PATH}")


def _migrate(db_path: str):
    """Apply any schema migrations needed for older DB versions."""
    migrations = [
        "ALTER TABLE predictions ADD COLUMN signal_text TEXT",
    ]
    conn = sqlite3.connect(db_path)
    try:
        for sql in migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore
        conn.commit()
    finally:
        conn.close()


# ── Candle helpers ─────────────────────────────────────────────────────────
def upsert_candles(rows: list[dict]):
    """Insert or ignore candle rows. Each dict: symbol,ts,dt,open,high,low,close,volume,atr14,log_ret"""
    sql = """
        INSERT OR IGNORE INTO candles
            (symbol,ts,dt,open,high,low,close,volume,atr14,log_ret)
        VALUES
            (:symbol,:ts,:dt,:open,:high,:low,:close,:volume,:atr14,:log_ret)
    """
    with get_conn() as conn:
        conn.executemany(sql, rows)
    log.info(f"Upserted {len(rows)} candle rows")


def get_candles(symbol: str, limit: int = 2000) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM candles WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ── Feature helpers ────────────────────────────────────────────────────────
def upsert_features(rows: list[dict]):
    cols = ["ts"] + FEATURE_COLS + ["candle_grade", "grade_score"]
    placeholders = ", ".join(f":{c}" for c in cols)
    col_str = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO features ({col_str}) VALUES ({placeholders})"
    with get_conn() as conn:
        conn.executemany(sql, rows)
    log.info(f"Upserted {len(rows)} feature rows")


def get_features(limit: int = 5000) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM features ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ── Trade helpers ──────────────────────────────────────────────────────────
def insert_trade(trade: dict) -> int:
    cols = list(trade.keys())
    sql  = f"INSERT INTO trade_results ({','.join(cols)}) VALUES ({','.join(':'+c for c in cols)})"
    with get_conn() as conn:
        cur = conn.execute(sql, trade)
        return cur.lastrowid


def close_trade(trade_id: int, exit_ts: int, exit_price: float,
                exit_reason: str, pnl: float, bars_held: int):
    outcome = 1 if pnl > 0 else 0
    with get_conn() as conn:
        conn.execute("""
            UPDATE trade_results
               SET exit_ts=?, exit_price=?, exit_reason=?,
                   pnl=?, bars_held=?, actual_outcome=?
             WHERE id=?
        """, (exit_ts, exit_price, exit_reason, pnl, bars_held, outcome, trade_id))


def get_trades(limit: int = 1000) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trade_results ORDER BY entry_ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Prediction helpers ─────────────────────────────────────────────────────
def insert_prediction(pred: dict):
    pred["created_at"] = datetime.utcnow().isoformat()
    cols = list(pred.keys())
    sql  = f"INSERT INTO predictions ({','.join(cols)}) VALUES ({','.join(':'+c for c in cols)})"
    with get_conn() as conn:
        conn.execute(sql, pred)


def get_latest_predictions(n: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Model log ──────────────────────────────────────────────────────────────
def log_model_run(model_type: str, n_samples: int,
                  val_accuracy: float = None, notes: str = ""):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO model_log (trained_at, model_type, n_samples, val_accuracy, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), model_type, n_samples, val_accuracy, notes))


# ── Stats ──────────────────────────────────────────────────────────────────
def get_stats() -> dict:
    with get_conn() as conn:
        n_candles   = conn.execute("SELECT COUNT(*) FROM candles WHERE symbol='MNQ'").fetchone()[0]
        n_features  = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        n_trades    = conn.execute("SELECT COUNT(*) FROM trade_results WHERE actual_outcome IS NOT NULL").fetchone()[0]
        win_rate    = conn.execute(
            "SELECT AVG(actual_outcome) FROM trade_results WHERE actual_outcome IS NOT NULL"
        ).fetchone()[0] or 0.0
        total_pnl   = conn.execute(
            "SELECT SUM(pnl) FROM trade_results WHERE pnl IS NOT NULL"
        ).fetchone()[0] or 0.0
        last_model  = conn.execute(
            "SELECT trained_at, model_type, val_accuracy FROM model_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(
        n_candles=n_candles, n_features=n_features, n_trades=n_trades,
        win_rate=win_rate, total_pnl=total_pnl,
        last_model=dict(last_model) if last_model else None
    )
