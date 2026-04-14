"""
Crypto Dashboard Web Server
Replaces `python -m http.server 8080` with a proper server that:
  - Serves the dashboard and report pages securely (no .env, .py exposure)
  - Exposes API endpoints so the dashboard can trigger pipeline runs
  - Streams pipeline log output back to the browser in real-time

Usage: python server.py
Then open: http://localhost:8080

Requires: pip install flask
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone

try:
    from flask import Flask, Response, jsonify, send_file, abort
except ImportError:
    print("Flask is not installed. Run: pip install flask")
    sys.exit(1)

app    = Flask(__name__, static_folder=None)
PYTHON = sys.executable
PORT   = 8080

# Force UTF-8 output from all subprocesses (fixes cp1250 UnicodeEncodeError on Windows)
_UTF8_ENV = os.environ.copy()
_UTF8_ENV["PYTHONIOENCODING"] = "utf-8"
_UTF8_ENV["PYTHONUTF8"]       = "1"

# Files in data/ that the browser is allowed to fetch.
# .env, .py, models/*.pkl etc. are never exposed.
_SAFE_DATA = {
    "dashboard_data.json",
    "signal_log.csv",
    "paper_summary.json",
    "paper_positions.json",
    "paper_ledger.csv",
    "testnet_summary.json",
    "testnet_positions.json",
    "testnet_ledger.csv",
    "strategy_comparison.json",
    "simple_paper_state.json",
    "simple_paper_ledger.csv",
}

# ── Pipeline run state (shared between request threads) ──────────────────────

_state = {
    "running":   False,
    "log":       [],        # list of log lines (capped at 500)
    "last_run":  None,      # ISO timestamp of last run start
    "exit_code": None,      # 0 = success, non-zero = failed, None = never run
}
_lock = threading.Lock()


# ── Static pages ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("dashboard.html")


@app.route("/report")
def report():
    if os.path.exists("report.html"):
        return send_file("report.html")
    return ("<h1 style='font-family:sans-serif;color:#ef4444'>No report yet</h1>"
            "<p style='font-family:sans-serif;color:#94a3b8;margin-top:8px'>"
            "Run <code>python report.py</code> first.</p>"), 404


# ── Safe data file serving ────────────────────────────────────────────────────

@app.route("/data/<path:filename>")
def serve_data(filename):
    """Serve only whitelisted files from the data/ directory."""
    if filename not in _SAFE_DATA:
        abort(403)
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


# ── API: status ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Current portfolio status + whether the pipeline is running."""
    result = {"paper": {}, "testnet": {}}

    for key, fname in [("paper", "paper_summary.json"),
                       ("testnet", "testnet_summary.json")]:
        path = os.path.join("data", fname)
        if os.path.exists(path):
            with open(path) as f:
                result[key] = json.load(f)

    with _lock:
        result["pipeline"] = {
            "running":   _state["running"],
            "last_run":  _state["last_run"],
            "exit_code": _state["exit_code"],
        }

    return jsonify(result)


# ── API: log ──────────────────────────────────────────────────────────────────

@app.route("/api/log")
def api_log():
    """Return the last N lines of the most recent pipeline run."""
    with _lock:
        return jsonify({
            "running": _state["running"],
            "log":     _state["log"][-200:],
        })


# ── API: refresh dashboard data ───────────────────────────────────────────────

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Re-run dashboard.py to pull fresh prices/signals. Takes ~10 seconds."""
    try:
        proc = subprocess.run(
            [PYTHON, "dashboard.py"],
            capture_output=True, text=True, timeout=90,
            env=_UTF8_ENV,
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return jsonify({
            "ok":      proc.returncode == 0,
            "updated": now,
            "output":  (proc.stdout or "")[-400:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "dashboard.py timed out"}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── API: run full daily pipeline ──────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def api_run():
    """
    Start the daily pipeline (predict → paper trade → testnet → dashboard → report)
    in a background thread. Pipeline output is streamed into _state["log"] so
    the browser can poll /api/log to watch progress.
    """
    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Pipeline already running"}), 409
        _state["running"]   = True
        _state["log"]       = ["[server] Pipeline started..."]
        _state["exit_code"] = None
        _state["last_run"]  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _run():
        try:
            proc = subprocess.Popen(
                [PYTHON, "run-all.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_UTF8_ENV,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                with _lock:
                    _state["log"].append(stripped)
                    if len(_state["log"]) > 500:
                        _state["log"] = _state["log"][-500:]
            proc.wait()
            with _lock:
                _state["exit_code"] = proc.returncode
                msg = "[server] Pipeline finished OK" if proc.returncode == 0 \
                      else f"[server] Pipeline exited with code {proc.returncode}"
                _state["log"].append(msg)
        except Exception as exc:
            with _lock:
                _state["log"].append(f"[server] Error: {exc}")
                _state["exit_code"] = -1
        finally:
            with _lock:
                _state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Pipeline started"})


# ── Server-Sent Events: live log stream ───────────────────────────────────────

@app.route("/api/log/stream")
def api_log_stream():
    """
    SSE endpoint: streams new log lines to the browser as they arrive.
    The dashboard connects with EventSource('/api/log/stream').
    """
    def generate():
        sent = 0
        while True:
            with _lock:
                lines   = _state["log"]
                running = _state["running"]
            if len(lines) > sent:
                for line in lines[sent:]:
                    yield f"data: {json.dumps(line)}\n\n"
                sent = len(lines)
            if not running and sent > 0:
                yield "data: __done__\n\n"
                return
            import time
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── API: trading performance ──────────────────────────────────────────────────

@app.route("/api/trading")
def api_trading():
    """Return paper and testnet trading performance (metrics + recent trades)."""
    import csv as _csv

    def load_metrics(ledger_path):
        if not os.path.exists(ledger_path):
            return {"recent_trades": [], "metrics": {}}
        rows = []
        try:
            with open(ledger_path, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            return {"recent_trades": [], "metrics": {}}

        sells = [r for r in rows if (r.get("action") or "").startswith("SELL")
                 or r.get("action") == "SELL"]
        pnls  = []
        for r in sells:
            try:
                pnls.append(float(r.get("pnl") or 0))
            except (ValueError, TypeError):
                pass

        n            = len(sells)
        wins         = sum(1 for p in pnls if p > 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss   = abs(sum(p for p in pnls if p < 0))

        return {
            "recent_trades": rows[-10:],
            "metrics": {
                "n_trades":      n,
                "wins":          wins,
                "win_rate":      round(wins / n * 100, 1) if n else 0,
                "total_pnl":     round(sum(pnls), 2),
                "profit_factor": round(gross_profit / gross_loss, 2)
                                 if gross_loss > 0 else (9.99 if gross_profit > 0 else 0),
            },
        }

    result = {"paper": {}, "testnet": {}}
    for key, sfile, lfile in [
        ("paper",   "paper_summary.json",   "paper_ledger.csv"),
        ("testnet", "testnet_summary.json", "testnet_ledger.csv"),
    ]:
        spath = os.path.join("data", sfile)
        if os.path.exists(spath):
            with open(spath) as f:
                result[key]["summary"] = json.load(f)
        result[key].update(load_metrics(os.path.join("data", lfile)))

    return jsonify(result)


# ── API: database read endpoints ─────────────────────────────────────────────

@app.route("/api/db/signals")
def api_db_signals():
    """Return recent signals from PostgreSQL (falls back to empty if DB unavailable)."""
    try:
        import db
        coin  = __import__("flask").request.args.get("coin")
        limit = int(__import__("flask").request.args.get("limit", 100))
        rows  = db.get_signals(coin=coin, limit=limit)
        return jsonify({"ok": True, "data": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "data": []}), 200


@app.route("/api/db/trades")
def api_db_trades():
    """Return recent trades from PostgreSQL."""
    try:
        import db
        from flask import request
        account = request.args.get("account_type")
        coin    = request.args.get("coin")
        limit   = int(request.args.get("limit", 100))
        rows    = db.get_trades(account_type=account, coin=coin, limit=limit)
        return jsonify({"ok": True, "data": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "data": []}), 200


@app.route("/api/db/positions")
def api_db_positions():
    """Return current positions from PostgreSQL."""
    try:
        import db
        from flask import request
        account = request.args.get("account_type")
        rows    = db.get_positions(account_type=account)
        return jsonify({"ok": True, "data": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "data": []}), 200


@app.route("/api/db/candles")
def api_db_candles():
    """Return recent OHLCV candles from PostgreSQL."""
    try:
        import db
        from flask import request
        coin  = request.args.get("coin", "BTC")
        limit = int(request.args.get("limit", 120))
        rows  = db.get_candles(coin=coin, limit=limit)
        return jsonify({"ok": True, "data": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "data": []}), 200


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  Crypto Dashboard Server")
    print(f"  Dashboard : http://localhost:{PORT}")
    print(f"  Report    : http://localhost:{PORT}/report")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*52}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
