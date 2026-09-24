"""Aggregate the parser-noise axis (study3_plan §0-P2) into
results/study3/parser_noise_report.json.

DVR/RR (chain-level, test_main, true chain structure) as a function of the
parser corruption rate for:
  - S1  (projection: parse-dependent chain assembly)  [computed]
  - S8  (mirror: g_a(l) conditions on parsed (a,l))   [computed, 3 seeds]
  - S7  (base mode: consumes NO parsed inputs -> outputs identical under
         parser noise by construction; rate-0 values carried, flagged
         structural_identity)
S0 rate-0 values included as the untrained reference. Raw numbers only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RATES = [0, 5, 10, 20]
SEEDS = [1042, 1043, 1044]


def chain_stats(run):
    p = ROOT / f"results/study3/metrics_{run}_test_main.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    g = df.groupby("chain_id")
    return {"DVR": round(float(g["violation"].mean().mean()), 4),
            "RR": round(float(g["responsive"].mean().mean()), 4)}


def main():
    report = {"split": "test_main", "rates_percent": RATES,
              "corruption_model": "per-request; 50% strength off-by-one "
                                  "(±1 clamped), 50% attribute confusion "
                                  "(uniform over other 3); tables shared "
                                  "across systems (seed 20260731+rate)",
              "corruptions": {}, "systems": {}}
    for r in [5, 10, 20]:
        t = json.loads((ROOT / "results/study3/parser_noise" /
                        f"corruptions_rate{r}.json").read_text())
        kinds = {}
        for v in t.values():
            kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
        report["corruptions"][str(r)] = {"n_requests": len(t), **kinds}

    s1 = {"0": chain_stats("S1_llama")}
    for r in [5, 10, 20]:
        s1[str(r)] = chain_stats(f"S1N{r}_llama")
    report["systems"]["S1_llama"] = {
        "mechanism": "parse-dependent chain assembly + PAVA", "by_rate": s1}

    s8 = {}
    for r in RATES:
        per_seed = {}
        for seed in SEEDS:
            run = (f"S8_llama_seed{seed}" if r == 0
                   else f"S8N{r}_llama_seed{seed}")
            st = chain_stats(run)
            if st:
                per_seed[str(seed)] = st
        if per_seed:
            s8[str(r)] = {
                "per_seed": per_seed,
                "DVR_mean": round(float(np.mean(
                    [v["DVR"] for v in per_seed.values()])), 4),
                "RR_mean": round(float(np.mean(
                    [v["RR"] for v in per_seed.values()])), 4)}
    report["systems"]["S8_llama"] = {
        "mechanism": "g_a(l) conditions on parsed (attribute, level)",
        "by_rate": s8}

    s7_clean = {str(s): chain_stats(f"S7_llama_seed{s}") for s in SEEDS}
    report["systems"]["S7_llama"] = {
        "mechanism": "base mode: no parsed inputs consumed; end-to-end text",
        "structural_identity": "outputs are byte-identical under parser "
                               "noise (the corrupted representation is never "
                               "an input); rate-0 values apply at all rates, "
                               "NOT re-run",
        "all_rates": {"per_seed": s7_clean,
                      "DVR_mean": round(float(np.mean(
                          [v["DVR"] for v in s7_clean.values()])), 4),
                      "RR_mean": round(float(np.mean(
                          [v["RR"] for v in s7_clean.values()])), 4)}}

    report["systems"]["S0_llama_reference"] = chain_stats("S0_llama")

    out = ROOT / "results/study3/parser_noise_report.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
