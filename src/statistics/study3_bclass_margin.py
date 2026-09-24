"""M2-review supplementary checks (user ruling 2026-07-31, item 4):

A. Per-system DVR decomposition by candidate-set type on test_main, with the
   B class (real-item pairs, eval-only per plan) reported separately —
   guards against "trained zero-DVR is an artifact of synthetic sets".
B. Per-system score-margin distribution on test_main: transition steps
   (delta_{k+1} - delta_k; violation iff step < -eps) and per-chain minimum
   step — guards against "zero DVR sits just above the violation boundary".

Raw numbers only. Output: results/study3/bclass_margin_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study3_main_matrix import SYSTEMS, runs_for  # noqa: E402

EPS = yaml.safe_load(open(ROOT / "configs/pilot.yaml"))["epsilon"]
SPLIT = "test_main"


def set_types():
    types = {}
    for line in open(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl"):
        cs = json.loads(line)
        types[cs["set_id"]] = cs["set_type"]
    return types


def load_run(run):
    p = ROOT / f"results/study3/metrics_{run}_{SPLIT}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    df["run"] = run  # keep seeds separate: the chain unit is (run, chain_id)
    df["step"] = df["delta_k1"] - df["delta_k"]
    return df


def pctl(x, qs=(5, 25, 50, 75, 95)):
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def main():
    types = set_types()
    report = {"split": SPLIT, "epsilon": EPS,
              "definitions": {
                  "step": "delta_{k+1} - delta_k on the focal pair; a "
                          "transition is a violation iff step < -epsilon",
                  "min_step_per_chain": "worst transition margin in a chain; "
                          "chain violates iff min_step < -epsilon",
                  "near_boundary": "|step| <= epsilon (within epsilon of the "
                          "violation boundary)",
                  "norm_min_step": "min_step / max(|delta_k|) over the chain "
                          "(scale-free; scores are in system-specific units)"}}

    out = {}
    for family in ["llama", "qwen"]:
        for system in SYSTEMS:
            runs = runs_for(system, family)
            dfs = [d for d in (load_run(r) for r in runs) if d is not None]
            if not dfs:
                continue
            entry = {}

            # A. DVR by candidate-set type (chain-level, mean across seeds)
            by_type = {}
            for t in ["A", "B", "C"]:
                per_seed = []
                for df in dfs:
                    sub = df[df["set_id"].map(types) == t]
                    if len(sub):
                        per_seed.append(float(
                            sub.groupby("chain_id")["violation"].mean().mean()))
                if per_seed:
                    by_type[t] = {"mean": round(float(np.mean(per_seed)), 4),
                                  "seed_range": [round(min(per_seed), 4),
                                                 round(max(per_seed), 4)],
                                  "n_chains_per_seed": int(
                                      dfs[0][dfs[0]["set_id"].map(types) == t]
                                      ["chain_id"].nunique())}
            entry["dvr_by_set_type"] = by_type

            # B. margin distribution (transitions pooled across seeds)
            allt = pd.concat(dfs)
            steps = allt["step"].to_numpy()
            key = ["run", "chain_id"]
            per_chain = allt.groupby(key).agg(min_step=("step", "min"))
            # chain-level max |delta| for scale-free margin
            dmax = pd.concat([allt.groupby(key)["delta_k"].apply(
                                  lambda s: s.abs().max()),
                              allt.groupby(key)["delta_k1"].apply(
                                  lambda s: s.abs().max())], axis=1).max(axis=1)
            norm_min = (per_chain["min_step"] / dmax.replace(0, np.nan)).dropna()
            ms = per_chain["min_step"].to_numpy()
            entry["margin"] = {
                "n_transitions": int(len(steps)),
                "step_pctl": pctl(steps),
                "frac_steps_violating": round(float((steps < -EPS).mean()), 4),
                "frac_steps_near_boundary": round(
                    float((np.abs(steps) <= EPS).mean()), 4),
                "n_chains": int(len(ms)),
                "min_step_pctl": pctl(ms),
                "frac_chains_violating": round(float((ms < -EPS).mean()), 4),
                "frac_chains_near_boundary": round(
                    float((np.abs(ms) <= EPS).mean()), 4),
                "norm_min_step_pctl": pctl(norm_min.to_numpy()),
            }
            out[f"{system}_{family}"] = entry
    report["systems"] = out

    dst = ROOT / "results/study3/bclass_margin_report.json"
    dst.write_text(json.dumps(report, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
