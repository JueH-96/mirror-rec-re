"""Ingest a real phone catalog for Study 4d (second domain; study4_plan.md §2).

Source: cleaned public mirror of the GSMArena Phone Dataset (arwinneil,
Kaggle; scraped from GSMArena.com). The mirror is a widely-redistributed
course-repo copy with columns weight_g / internal_memory_GB / RAM_GB /
battery (mAh) / approx_price_EUR / CPU. Raw CSV cached in data_raw/;
processed registry-unit rows go to data_processed/real_phones.csv.
Provenance and license status are recorded in phone_catalog_meta.json and
experiment_manifest.yaml (user ruling b).

Qualification gate (M0, all must pass or the script exits non-zero):
  1. >= 300 complete rows within registry valid ranges;
  2. >= 500 pairs with gap >= min_pair_gap for each target attribute;
  3. numeric dtype + registry range assertions.
"""
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

URLS = [
    "https://raw.githubusercontent.com/shaiful019/MachineLearningWithPython/master/Sessions/Day11/LR/regression_mobile_price.csv",
    "https://raw.githubusercontent.com/shaiful019/MLOB1/master/Sessions/Day15/regression_mobile_price.csv",
]
EUR_USD = 1.08  # same conversion used for the laptop catalog


def main():
    raw_path = ROOT / "data_raw" / "phone_specs_raw.csv"
    df = None
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        for url in URLS:
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                raw_path.write_bytes(data)
                df = pd.read_csv(raw_path)
                break
            except Exception as e:  # noqa: BLE001 - try next mirror
                print(f"WARN: {url} failed: {e}", file=sys.stderr)
    if df is None:
        sys.exit("ERROR: no phone catalog available; Study 4d blocked (report to user).")

    registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry_phones.yaml"))
    out = pd.DataFrame()
    out["price"] = (pd.to_numeric(df["approx_price_EUR"], errors="coerce") * EUR_USD).round(0)
    out["weight"] = (pd.to_numeric(df["weight_g"], errors="coerce") / 1000).round(3)
    out["battery_life"] = pd.to_numeric(df["battery"], errors="coerce")
    out["storage"] = pd.to_numeric(df["internal_memory_GB"], errors="coerce")
    out["cpu"] = df["CPU"].fillna("n/a")
    out["ram"] = pd.to_numeric(df["RAM_GB"], errors="coerce").fillna(2).astype(int)

    n_raw = len(out)
    out = out.dropna(subset=["price", "weight", "battery_life", "storage"])
    for a in ("price", "weight", "battery_life", "storage"):
        lo, hi = registry[a]["valid_range"]
        out = out[(out[a] >= lo) & (out[a] <= hi)]
    out = out.reset_index(drop=True)

    gate = {"rows_raw": n_raw, "rows_qualified": len(out), "eligible_pairs": {}}
    failures = []
    if len(out) < 300:
        failures.append(f"rows_qualified {len(out)} < 300")
    for a in ("price", "weight", "battery_life", "storage"):
        vals = sorted(out[a].tolist())
        gap = registry[a]["min_pair_gap"]
        # pairs with value difference >= gap, via bisect over the sorted values
        from bisect import bisect_left
        n_pairs = sum(len(vals) - bisect_left(vals, v + gap) for v in vals)
        gate["eligible_pairs"][a] = n_pairs
        if n_pairs < 500:
            failures.append(f"{a}: eligible pairs {n_pairs} < 500")
        assert pd.api.types.is_numeric_dtype(out[a]), a

    dest = ROOT / "data_processed" / "real_phones.csv"
    out.to_csv(dest, index=False)
    meta = {
        "rows": len(out),
        "source_mirror": URLS[0],
        "upstream": "GSMArena Phone Dataset (arwinneil, kaggle.com/datasets/arwinneil/gsmarena-phone-dataset); data scraped from GSMArena.com",
        "license": ("no explicit dataset license on the mirror; upstream scraper repo "
                    "(github.com/arwinneil/phone-dataset) is MIT; underlying specs data "
                    "(c) GSMArena.com - research/evaluation use only"),
        "access_date": str(date.today()),
        "unit_conversions": {"price": f"EUR*{EUR_USD}->USD", "weight": "g/1000->kg",
                             "battery_life": "mAh (capacity proxy)", "storage": "GB"},
        "qualification_gate": gate,
        "qualification_passed": not failures,
        "failures": failures,
    }
    (ROOT / "data_processed" / "phone_catalog_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
