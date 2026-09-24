"""Assemble pilot deliverables after all model sweeps finish.

1. Merge per-(model, variant) metric parquets -> results/metrics_by_instance.parquet
2. Run analyze.py over all models x variants (aggregate_results.csv, statistical_tests.json)
3. Emit results/gonogo_precheck.json: quantitative evidence for each §13.2 Go
   condition (suggested operationalizations are recorded in the JSON; the
   final Go/No-Go verdict is a human decision, recorded in the manifest).
Idempotent: safe to rerun at any time.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["qwen2.5-14b-instruct", "llama-3.1-8b-instruct", "glm-4-9b-chat", "rule_baseline"]
VARIANTS = ["plain", "structured", "explain_then_rank", "direction_constraint"]

# Suggested operationalizations (documented, not contract-fixed):
DVR_NONTRIVIAL = 0.02   # cond 1: plain DVR above this counts as non-trivial
DVR_NEAR_ZERO = 0.01    # cond 3: no prompt variant reaches below this
PARSE_FAIL_MAX = 0.10   # cond 4: format failures must stay below this share


def main():
    frames = []
    for p in glob.glob(str(ROOT / "results" / "metrics_*.parquet")):
        if Path(p).name == "metrics_by_instance.parquet":
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit("no metric parquets found")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(ROOT / "results" / "metrics_by_instance.parquet")

    subprocess.run([sys.executable, str(ROOT / "src" / "statistics" / "analyze.py"),
                    "--models", *MODELS, "--variants", *VARIANTS], check=True, cwd=ROOT)

    llm = df[df.model != "rule_baseline"]
    pre = {"suggested_thresholds": {"dvr_nontrivial": DVR_NONTRIVIAL,
                                    "dvr_near_zero": DVR_NEAR_ZERO,
                                    "parse_fail_max": PARSE_FAIL_MAX},
           "conditions": {}}

    plain = llm[llm.variant == "plain"]
    dvr_by_model = plain.groupby("model")["violation"].mean().to_dict()
    pre["conditions"]["1_nontrivial_dvr_two_families"] = {
        "dvr_plain_by_model": dvr_by_model,
        "n_models_above_threshold": int(sum(v > DVR_NONTRIVIAL for v in dvr_by_model.values())),
        "evidence_met": sum(v > DVR_NONTRIVIAL for v in dvr_by_model.values()) >= 2,
    }
    fam = plain.groupby(["model", "template_family"])["violation"].mean()
    pre["conditions"]["2_persists_unseen_wording"] = {
        "dvr_plain_by_family": {f"{m}|{t}": round(v, 4) for (m, t), v in fam.items()},
        "held_out_family": "T3",
        "evidence_met": bool((fam.xs("T3", level="template_family") > DVR_NONTRIVIAL).all()),
    }
    best_variant = llm.groupby(["model", "variant"])["violation"].mean().groupby("model").min().to_dict()
    pre["conditions"]["3_prompt_only_insufficient"] = {
        "min_dvr_across_variants_by_model": {k: round(v, 4) for k, v in best_variant.items()},
        "evidence_met": bool(all(v > DVR_NEAR_ZERO for v in best_variant.values())),
    }
    parse_stats = {}
    for m in MODELS:
        if m == "rule_baseline":
            continue
        rows = []
        for v in VARIANTS:
            p = ROOT / "results" / "raw" / m / f"{v}.jsonl"
            if p.exists():
                rows += [json.loads(l)["parse_status"] for l in open(p)]
        if rows:
            parse_stats[m] = round(sum(s != "ok" for s in rows) / len(rows), 4)
    pre["conditions"]["4_not_format_artifact"] = {
        "parse_failure_share_by_model": parse_stats,
        "evidence_met": bool(all(v < PARSE_FAIL_MAX for v in parse_stats.values())),
    }
    for cid in ("5_preliminary_method_reduces_dvr", "6_responsiveness_no_collapse",
                "7_utility_loss_within_budget"):
        pre["conditions"][cid] = {"status": "not_assessed", "reason": "Study 3+ scope; pilot covers Study 1/2 only per user instruction"}
    real = plain[plain.set_type == "B"]
    pre["conditions"]["8_persists_real_items"] = {
        "dvr_plain_type_B_by_model": real.groupby("model")["violation"].mean().round(4).to_dict(),
        "n_chains_type_B": int(real.groupby(["model"])["set_id"].nunique().min()) if len(real) else 0,
        "scope_note": "type-B covers price/weight/storage (real catalog lacks battery_life)",
        "evidence_met": bool((real.groupby("model")["violation"].mean() > DVR_NONTRIVIAL).all()) if len(real) else False,
    }
    pre["rule_baseline_sanity"] = {
        "DVR": float(df[(df.model == "rule_baseline")]["violation"].mean()),
        "expected": 0.0,
    }
    with open(ROOT / "results" / "gonogo_precheck.json", "w") as f:
        json.dump(pre, f, indent=2, default=str)
    print(json.dumps(pre["conditions"], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
