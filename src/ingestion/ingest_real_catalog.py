"""Ingest a real laptop catalog for type-B matched pairs (§7.4B).

Tries public mirrors of the widely-used 1300-laptop specs dataset. The raw CSV
is cached in data_raw/; processed rows (attribute-normalized, registry units)
go to data_processed/real_laptops.csv. Battery life is absent from this
dataset, so type-B sets cover price/weight/storage only — recorded in the
manifest and surfaced in the pilot report (Go condition 8 scope note).
"""
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

URLS = [
    "https://raw.githubusercontent.com/Raghavagr/Laptop_Price_Prediction/main/laptop_data.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset4/main/laptop_price.csv",
]


def parse_storage_gb(memory_str):
    """'256GB SSD' / '1TB HDD' / '128GB SSD + 1TB HDD' -> total GB."""
    total = 0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB)", str(memory_str), re.I):
        total += float(num) * (1024 if unit.upper() == "TB" else 1)
    return total or None


def main():
    raw_path = ROOT / "data_raw" / "laptop_price_raw.csv"
    df = None
    if raw_path.exists():
        df = pd.read_csv(raw_path, encoding="latin-1")
    else:
        for url in URLS:
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                raw_path.write_bytes(data)
                df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
                break
            except Exception as e:  # noqa: BLE001 - try next mirror
                print(f"WARN: {url} failed: {e}", file=sys.stderr)
    if df is None:
        print("ERROR: no real catalog available; type-B sets will fall back to synthetic.")
        sys.exit(2)

    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    price_col = cols.get("price_euros") or cols.get("price")
    price = pd.to_numeric(df[price_col], errors="coerce")
    if "euros" in price_col.lower():
        price = price * 1.08  # EUR -> USD
    elif price.median() > 20000:
        price = price * 0.012  # INR -> USD (mirror stores INR)
    out["price"] = price
    out["weight"] = pd.to_numeric(df[cols["weight"]].astype(str).str.replace("kg", "", regex=False),
                                  errors="coerce") if "weight" in cols else None
    out["storage"] = df[cols["memory"]].map(parse_storage_gb) if "memory" in cols else None
    out["battery_life"] = None  # not present in this dataset
    out["cpu"] = df[cols["cpu"]] if "cpu" in cols else "n/a"
    out["ram"] = pd.to_numeric(df[cols["ram"]].astype(str).str.replace("GB", "", regex=False),
                               errors="coerce") if "ram" in cols else 16
    out = out.dropna(subset=["price", "weight", "storage"])
    out["price"] = out["price"].round(0)
    dest = ROOT / "data_processed" / "real_laptops.csv"
    out.to_csv(dest, index=False)
    meta = {"rows": len(out), "source": "laptop_price.csv public mirror",
            "attributes_available": ["price", "weight", "storage"],
            "attributes_missing": ["battery_life"]}
    (ROOT / "data_processed" / "real_catalog_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
