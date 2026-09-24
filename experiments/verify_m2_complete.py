"""Hard completion check for Study 3 M2 (does NOT trust the queue's exit marker).

A run counts as complete only if ALL of:
  - experiments/study3/<run>/final.pt exists
  - all 4 raw eval files exist with exactly eligible*families*4 lines
  - all 4 metrics parquets exist, load, and have exactly eligible*families*3 rows

M2 is complete only when every run in configs/study3/queue_order.txt (24 runs)
passes. Exit 0 = complete, 1 = incomplete (per-run report on stdout).

Usage: python experiments/verify_m2_complete.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]


def expected_counts(split):
    eligible = len(json.load(open(
        ROOT / f"data_processed/study3/eligible_sets_{split}.json")))
    fams = 1 if split == "test_unseen_word" else 2
    return eligible * fams * 4, eligible * fams * 3  # raw lines, parquet rows


def check_run(run, exp):
    problems = []
    if not (ROOT / "experiments/study3" / run / "final.pt").exists():
        problems.append("final.pt missing")
    for split in SPLITS:
        n_raw, n_parquet = exp[split]
        raw = ROOT / f"results/study3/raw/{run}/{split}.jsonl"
        if not raw.exists():
            problems.append(f"{split}: raw missing")
        else:
            got = sum(1 for _ in open(raw))
            if got != n_raw:
                problems.append(f"{split}: raw {got}/{n_raw} lines")
        pq = ROOT / f"results/study3/metrics_{run}_{split}.parquet"
        if not pq.exists():
            problems.append(f"{split}: parquet missing")
        else:
            try:
                rows = len(pd.read_parquet(pq))
                if rows != n_parquet:
                    problems.append(f"{split}: parquet {rows}/{n_parquet} rows")
            except Exception as e:  # noqa: BLE001 - any unreadable parquet is a failure
                problems.append(f"{split}: parquet unreadable ({e})")
    return problems


def main():
    exp = {s: expected_counts(s) for s in SPLITS}
    runs = []
    for line in open(ROOT / "configs/study3/queue_order.txt"):
        cfg = yaml.safe_load(open(ROOT / "configs/study3" / line.strip()))
        runs.append(cfg["run_name"])
    incomplete = 0
    for run in runs:
        problems = check_run(run, exp)
        if problems:
            incomplete += 1
            print(f"INCOMPLETE {run}: " + "; ".join(problems))
        else:
            print(f"OK {run}")
    print(f"M2_VERIFY: {len(runs) - incomplete}/{len(runs)} runs complete")
    print("M2_VERIFY_PASS" if incomplete == 0 else "M2_VERIFY_FAIL")
    sys.exit(0 if incomplete == 0 else 1)


if __name__ == "__main__":
    main()
