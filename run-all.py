"""
run_all.py — Master Orchestrator
Runs the full pipeline automatically. The scheduler calls this every morning.

Modes:
  python run_all.py           → daily: signals + paper trade + dashboard + report
  python run_all.py --full    → weekly: re-fetch data + retrain + daily routine
  python run_all.py --report  → just generate today's report
  python run_all.py --weekly-report → weekly performance report
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime, timezone

PYTHON  = sys.executable
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run(script: str, label: str, args: list = None) -> bool:
    cmd = [PYTHON, script] + (args or [])
    print(f"\n  ▶  {label}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ✗  {label} FAILED (exit {result.returncode})")
        return False
    print(f"  ✓  {label} done")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",          action="store_true", help="Full rebuild + daily")
    parser.add_argument("--report",        action="store_true", help="Report only")
    parser.add_argument("--weekly-report", action="store_true", help="Weekly report only")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    now_time = datetime.now(timezone.utc)

    print(f"\n{'='*52}")
    if args.full:
        mode = "FULL REBUILD"
    elif args.report:
        mode = "REPORT ONLY"
    elif args.weekly_report:
        mode = "WEEKLY REPORT"
    else:
        mode = "DAILY RUN"
    print(f"  {mode} — {now}")
    print(f"{'='*52}")

    # Report only modes
    if args.report:
        run("report.py", "Daily report")
        return
    if args.weekly_report:
        run("report.py", "Weekly report", ["--weekly"])
        return

    # Full rebuild — run once a week (Sundays)
    if args.full:
        steps = [
            ("crypto-pipeline.py",     "Fetching market data",      []),
            ("feature-engineering.py", "Feature engineering (MTF)", []),
            ("train-model.py",         "Training ensemble models",  []),
        ]
        for script, label, script_args in steps:
            if os.path.exists(script):
                ok = run(script, label, script_args)
                if not ok:
                    print(f"  ⚠  Continuing despite error...")

    # Daily routine — runs every morning
    daily_steps = [
        ("predict-today.py",   "Generating signals",           []),
        ("paper-trader.py",    "Executing paper trades",       []),
        ("testnet-trader.py",  "Executing testnet trades",     []),
        ("dashboard.py",       "Updating dashboard",           []),
        ("report.py",          "Generating daily report",      []),
    ]

    # On Mondays also generate weekly report
    if now_time.weekday() == 0:
        daily_steps.append(
            ("report.py", "Generating weekly report", ["--weekly"])
        )

    success = True
    for script, label, script_args in daily_steps:
        if not os.path.exists(script):
            print(f"  ⚠  {script} not found — skipping")
            continue
        ok = run(script, label, script_args)
        if not ok:
            success = False

    # Final status
    print(f"\n{'='*52}")
    status = "✓ COMPLETED" if success else "⚠ COMPLETED WITH ERRORS"
    print(f"  {status} — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"\n  Dashboard: {os.path.abspath('dashboard.html')}")
    print(f"  Report:    {os.path.abspath('report.html')}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()