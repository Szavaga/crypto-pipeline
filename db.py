"""
Database module for crypto-pipeline.
Handles PostgreSQL connection and CRUD operations.

Connection string is read from .env:
  DATABASE_URL=postgresql://cryptobot:password@<ORACLE_IP>:5432/crypto_pipeline

Falls back silently if DB is not configured — all writes are optional,
CSV/JSON files remain the primary storage.
"""

import os
import contextlib

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    coin        VARCHAR(10) NOT NULL,
    price       NUMERIC(18,6),
    signal      VARCHAR(50),
    prob_up     NUMERIC(6,2),
    prob_down   NUMERIC(6,2),
    kelly_pct   NUMERIC(6,2),
    confidence  VARCHAR(30),
    mtf_score   INTEGER,
    fear_greed  INTEGER,
    fg_label    VARCHAR(30),
    funding     NUMERIC(10,6),
    btc_dom     NUMERIC(6,2),
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, coin)
);

CREATE TABLE IF NOT EXISTS trades (
    id                  SERIAL PRIMARY KEY,
    date                TIMESTAMP NOT NULL,
    account_type        VARCHAR(10) NOT NULL,
    coin                VARCHAR(10) NOT NULL,
    action              VARCHAR(20) NOT NULL,
    price               NUMERIC(18,6),
    quantity            NUMERIC(18,8),
    value_usdt          NUMERIC(18,4),
    commission          NUMERIC(18,6),
    pnl                 NUMERIC(18,4),
    pnl_pct             NUMERIC(8,4),
    balance_after       NUMERIC(18,4),
    signal_confidence   NUMERIC(6,2),
    kelly_pct           NUMERIC(6,2),
    reason              TEXT,
    order_id            VARCHAR(50),
    oco_order_list_id   INTEGER DEFAULT -1
);

CREATE TABLE IF NOT EXISTS positions (
    id                  SERIAL PRIMARY KEY,
    account_type        VARCHAR(10) NOT NULL,
    coin                VARCHAR(10) NOT NULL,
    in_position         BOOLEAN DEFAULT FALSE,
    entry_price         NUMERIC(18,6) DEFAULT 0,
    quantity            NUMERIC(18,8) DEFAULT 0,
    value_usdt          NUMERIC(18,4) DEFAULT 0,
    balance             NUMERIC(18,4) DEFAULT 0,
    sl_price            NUMERIC(18,6) DEFAULT 0,
    tp_price            NUMERIC(18,6) DEFAULT 0,
    confidence          NUMERIC(6,2) DEFAULT 0,
    kelly_pct           NUMERIC(6,2) DEFAULT 0,
    entry_date          DATE,
    entry_order_id      VARCHAR(50),
    oco_order_list_id   INTEGER DEFAULT -1,
    updated_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(account_type, coin)
);

CREATE TABLE IF NOT EXISTS candles (
    id      SERIAL PRIMARY KEY,
    date    DATE NOT NULL,
    coin    VARCHAR(10) NOT NULL,
    open    NUMERIC(18,6),
    high    NUMERIC(18,6),
    low     NUMERIC(18,6),
    close   NUMERIC(18,6),
    volume  NUMERIC(24,2),
    UNIQUE(date, coin)
);
"""

# ── Connection ────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _get_conn():
    """Context manager: yields a psycopg2 connection or raises if not configured."""
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 not installed")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_available() -> bool:
    """Return True if DB is configured and reachable."""
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def init_schema():
    """Create all tables if they don't exist. Safe to call multiple times."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
    print("  ✓  DB schema ready")


# ── Write helpers ─────────────────────────────────────────────────────────────

def upsert_signal(s: dict):
    """
    Insert or update a signal row.
    s must have: date, coin — all other fields optional.
    """
    sql = """
    INSERT INTO signals
        (date, coin, price, signal, prob_up, prob_down, kelly_pct,
         confidence, mtf_score, fear_greed, fg_label, funding, btc_dom)
    VALUES
        (%(date)s, %(coin)s, %(price)s, %(signal)s, %(prob_up)s, %(prob_down)s,
         %(kelly_pct)s, %(confidence)s, %(mtf_score)s, %(fear_greed)s,
         %(fg_label)s, %(funding)s, %(btc_dom)s)
    ON CONFLICT (date, coin) DO UPDATE SET
        price       = EXCLUDED.price,
        signal      = EXCLUDED.signal,
        prob_up     = EXCLUDED.prob_up,
        prob_down   = EXCLUDED.prob_down,
        kelly_pct   = EXCLUDED.kelly_pct,
        confidence  = EXCLUDED.confidence,
        mtf_score   = EXCLUDED.mtf_score,
        fear_greed  = EXCLUDED.fear_greed,
        fg_label    = EXCLUDED.fg_label,
        funding     = EXCLUDED.funding,
        btc_dom     = EXCLUDED.btc_dom
    """
    row = {
        "date":       s.get("date"),
        "coin":       s.get("coin"),
        "price":      s.get("price") or None,
        "signal":     s.get("signal") or None,
        "prob_up":    s.get("prob_up") or None,
        "prob_down":  s.get("prob_down") or None,
        "kelly_pct":  s.get("kelly_pct") or None,
        "confidence": s.get("confidence") or None,
        "mtf_score":  s.get("mtf_score") or None,
        "fear_greed": s.get("fear_greed") or None,
        "fg_label":   s.get("fg_label") or None,
        "funding":    s.get("funding") or None,
        "btc_dom":    s.get("btc_dom") or None,
    }
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)


def insert_trade(account_type: str, t: dict):
    """
    Insert a trade row. account_type = 'paper' | 'testnet'.
    t must match the ledger CSV column structure.
    """
    sql = """
    INSERT INTO trades
        (date, account_type, coin, action, price, quantity, value_usdt,
         commission, pnl, pnl_pct, balance_after, signal_confidence,
         kelly_pct, reason, order_id, oco_order_list_id)
    VALUES
        (%(date)s, %(account_type)s, %(coin)s, %(action)s, %(price)s,
         %(quantity)s, %(value_usdt)s, %(commission)s, %(pnl)s, %(pnl_pct)s,
         %(balance_after)s, %(signal_confidence)s, %(kelly_pct)s,
         %(reason)s, %(order_id)s, %(oco_order_list_id)s)
    """
    row = {
        "date":               t.get("date"),
        "account_type":       account_type,
        "coin":               t.get("coin"),
        "action":             t.get("action"),
        "price":              t.get("price") or None,
        "quantity":           t.get("quantity") or None,
        "value_usdt":         t.get("value_usdt") or t.get("value") or None,
        "commission":         t.get("commission") or None,
        "pnl":                t.get("pnl") or None,
        "pnl_pct":            t.get("pnl_pct") or None,
        "balance_after":      t.get("balance_after") or None,
        "signal_confidence":  t.get("signal_confidence") or None,
        "kelly_pct":          t.get("kelly_pct") or None,
        "reason":             t.get("reason") or None,
        "order_id":           str(t.get("order_id") or ""),
        "oco_order_list_id":  int(t.get("oco_order_list_id") or -1),
    }
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)


def upsert_position(account_type: str, coin: str, p: dict):
    """
    Insert or update a position row.
    account_type = 'paper' | 'testnet'.
    """
    sql = """
    INSERT INTO positions
        (account_type, coin, in_position, entry_price, quantity, value_usdt,
         balance, sl_price, tp_price, confidence, kelly_pct, entry_date,
         entry_order_id, oco_order_list_id, updated_at)
    VALUES
        (%(account_type)s, %(coin)s, %(in_position)s, %(entry_price)s,
         %(quantity)s, %(value_usdt)s, %(balance)s, %(sl_price)s, %(tp_price)s,
         %(confidence)s, %(kelly_pct)s, %(entry_date)s, %(entry_order_id)s,
         %(oco_order_list_id)s, NOW())
    ON CONFLICT (account_type, coin) DO UPDATE SET
        in_position       = EXCLUDED.in_position,
        entry_price       = EXCLUDED.entry_price,
        quantity          = EXCLUDED.quantity,
        value_usdt        = EXCLUDED.value_usdt,
        balance           = EXCLUDED.balance,
        sl_price          = EXCLUDED.sl_price,
        tp_price          = EXCLUDED.tp_price,
        confidence        = EXCLUDED.confidence,
        kelly_pct         = EXCLUDED.kelly_pct,
        entry_date        = EXCLUDED.entry_date,
        entry_order_id    = EXCLUDED.entry_order_id,
        oco_order_list_id = EXCLUDED.oco_order_list_id,
        updated_at        = NOW()
    """
    row = {
        "account_type":       account_type,
        "coin":               coin,
        "in_position":        bool(p.get("in_position", False)),
        "entry_price":        p.get("entry_price") or 0,
        "quantity":           p.get("quantity") or 0,
        "value_usdt":         p.get("value_usdt") or p.get("value") or 0,
        "balance":            p.get("balance") or 0,
        "sl_price":           p.get("sl_price") or 0,
        "tp_price":           p.get("tp_price") or 0,
        "confidence":         p.get("confidence") or 0,
        "kelly_pct":          p.get("kelly_pct") or 0,
        "entry_date":         p.get("entry_date") or None,
        "entry_order_id":     str(p.get("entry_order_id") or ""),
        "oco_order_list_id":  int(p.get("oco_order_list_id") or -1),
    }
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)


def upsert_candles(coin: str, candles: list):
    """
    Bulk upsert OHLCV candles.
    candles = list of dicts with keys: date/t, open/o, high/h, low/l, close/c, volume/v
    """
    sql = """
    INSERT INTO candles (date, coin, open, high, low, close, volume)
    VALUES %s
    ON CONFLICT (date, coin) DO UPDATE SET
        open   = EXCLUDED.open,
        high   = EXCLUDED.high,
        low    = EXCLUDED.low,
        close  = EXCLUDED.close,
        volume = EXCLUDED.volume
    """
    import datetime
    rows = []
    for c in candles:
        # Support both dashboard format {t,o,h,l,c,v} and plain {date,open,...}
        if "t" in c:
            date = datetime.datetime.utcfromtimestamp(c["t"] / 1000).date()
        else:
            date = c.get("date")
        rows.append((
            date, coin,
            c.get("o") or c.get("open"),
            c.get("h") or c.get("high"),
            c.get("l") or c.get("low"),
            c.get("c") or c.get("close"),
            c.get("v") or c.get("volume"),
        ))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows)


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_signals(coin: str = None, limit: int = 100) -> list:
    """Return recent signals as list of dicts."""
    sql = "SELECT * FROM signals"
    params = []
    if coin:
        sql += " WHERE coin = %s"
        params.append(coin)
    sql += " ORDER BY date DESC, created_at DESC LIMIT %s"
    params.append(limit)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def get_trades(account_type: str = None, coin: str = None, limit: int = 100) -> list:
    """Return recent trades as list of dicts."""
    where = []
    params = []
    if account_type:
        where.append("account_type = %s"); params.append(account_type)
    if coin:
        where.append("coin = %s"); params.append(coin)
    sql = "SELECT * FROM trades"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date DESC LIMIT %s"
    params.append(limit)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def get_positions(account_type: str = None) -> list:
    """Return current positions as list of dicts."""
    sql = "SELECT * FROM positions"
    params = []
    if account_type:
        sql += " WHERE account_type = %s"
        params.append(account_type)
    sql += " ORDER BY account_type, coin"
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def get_candles(coin: str, limit: int = 120) -> list:
    """Return recent candles as list of dicts."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM candles WHERE coin = %s ORDER BY date DESC LIMIT %s",
                (coin, limit)
            )
            return [dict(r) for r in cur.fetchall()]


# ── Convenience wrapper ───────────────────────────────────────────────────────

def try_write(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) silently if DB is not available.
    Use this to wrap DB writes so missing DB never breaks the pipeline.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"  ⚠  DB write skipped: {e}")


# ── CLI: init schema ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing database schema...")
    init_schema()
    print("Done.")
