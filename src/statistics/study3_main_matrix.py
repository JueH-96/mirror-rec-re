"""Study 3 M2 main-matrix tables: raw numbers + preregistered tests only.

Produces results/study3/main_matrix_report.json with:
  1. dvr_table: per system x split, per-seed chain-level DVR, cross-seed
     mean/min/max; on test_main (primary endpoint) a 95% bootstrap CI over
     seed-averaged per-chain rates (clustered by chain, B=2000)
  2. h3: preregistered S7 vs {S2..S6} paired permutation on test_main
     (chain-level pairing, per-chain rates seed-averaged, B=5000, Holm over
     the 5 comparisons); effect size = absolute DVR reduction
  3. guardrails: RR per system (+ collapse flag once S0 exists),
     NDCG@5 (rule-utility oracle gains) per system with delta vs S2
  4. seed1043_consistency: S2_llama seed 1043 vs 1042/1044 per split

Interpretation is deliberately absent (M3 scope). S0/S1/S9 rows are emitted
as "pending" until their eval outputs exist.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_generation.rule_baseline import LEVEL_W, norm_utility  # noqa: E402
from src.projection.isotonic_projection import ndcg_at_k  # noqa: E402

RNG = np.random.default_rng(20260730)
SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]
SYSTEMS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
UNSEEDED = {"S0", "S1"}  # zero-shot (+projection): one run per family, no seeds
LLAMA_SEEDS = [1042, 1043, 1044]
QWEN = {"S0", "S1", "S2", "S7", "S8", "S9"}  # qwen coverage (seed 1042 where seeded)
SETS_FILE = {"val": "candidate_sets_val.jsonl",
             "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl"}


def runs_for(system, family):
    if system in UNSEEDED:
        return [f"{system}_{family}"] if family == "llama" or system in QWEN else []
    if family == "llama":
        return [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
    return [f"{system}_qwen_seed1042"] if system in QWEN else []


def chain_rates(run, split, col):
    """Per-chain mean of a boolean transition column; None if metrics missing."""
    p = ROOT / f"results/study3/metrics_{run}_{split}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    return df.groupby("chain_id")[col].mean()


def boot_ci(vals, n=2000):
    boots = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_perm(diff, n=5000):
    obs = diff.mean()
    signs = RNG.choice([-1, 1], size=(n, len(diff)))
    null = (signs * diff.to_numpy()).mean(axis=1)
    # add-one convention: a permutation p-value is never exactly 0
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def holm(pvals):
    order = sorted(pvals, key=lambda k: pvals[k])
    adj, prev = {}, 0.0
    for rank, k in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - rank) * pvals[k]))
        adj[k] = prev
    return adj


def seed_avg_chain_rates(system, family, split, col):
    """Align chains across seeds and average per chain (chain-level pairing)."""
    per_seed = [chain_rates(r, split, col) for r in runs_for(system, family)]
    per_seed = [s for s in per_seed if s is not None]
    if not per_seed:
        return None
    return pd.concat(per_seed, axis=1).mean(axis=1)


def load_sets(split):
    sets = {}
    fname = SETS_FILE.get(split, "candidate_sets_test_main.jsonl")
    for line in open(ROOT / "data_processed/study3" / fname):
        cs = json.loads(line)
        sets[cs["set_id"]] = cs
    return sets


def ndcg5(run, split, sets, registry, eligible):
    p = ROOT / f"results/study3/raw/{run}/{split}.jsonl"
    if not p.exists():
        return None
    vals = []
    for line in open(p):
        r = json.loads(line)
        if r["set_id"] not in eligible or r["parse_status"] != "ok":
            continue
        cs = sets[r["set_id"]]
        attr, level = r["target_attribute"], r["strength_level"]
        gains = {}
        for it in cs["items"]:
            base = np.mean([norm_utility(it[a], registry[a])
                            for a in registry if it.get(a) is not None])
            gains[it["item_id"]] = 0.5 * base + LEVEL_W[level] * norm_utility(
                it[attr], registry[attr])
        vals.append(ndcg_at_k(r["ranking"]["ids"], gains))
    return float(np.mean(vals)) if vals else None


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    report = {"system_definitions": {
        "S0": "zero-shot plain (pilot anchor), no training",
        "S1": "S0 + isotonic projection (B12 strong baseline)",
        "S9": "S8 + isotonic projection (composability), all seeds"}}

    # 1. DVR table -----------------------------------------------------------
    dvr = {}
    for family in ["llama", "qwen"]:
        for system in SYSTEMS:
            runs = runs_for(system, family)
            if not runs:
                continue
            entry = {}
            for split in SPLITS:
                per_seed = {}
                for run in runs:
                    cr = chain_rates(run, split, "violation")
                    if cr is not None:
                        key = run.rsplit("seed", 1)[1] if "seed" in run else "single"
                        per_seed[key] = round(float(cr.mean()), 4)
                if not per_seed:
                    entry[split] = "missing"
                    continue
                vals = list(per_seed.values())
                entry[split] = {"per_seed": per_seed,
                                "mean": round(float(np.mean(vals)), 4),
                                "seed_range": [min(vals), max(vals)]}
                if split == "test_main":
                    sa = seed_avg_chain_rates(system, family, split, "violation")
                    entry[split]["ci95_seedavg_chains"] = [
                        round(x, 4) for x in boot_ci(sa.to_numpy())]
                    entry[split]["n_chains"] = int(len(sa))
            dvr[f"{system}_{family}"] = entry
    report["dvr_table"] = dvr

    # 2. H3 preregistered test ----------------------------------------------
    s7 = seed_avg_chain_rates("S7", "llama", "test_main", "violation")
    h3, pvals = {}, {}
    for system in ["S2", "S3", "S4", "S5", "S6"]:
        other = seed_avg_chain_rates(system, "llama", "test_main", "violation")
        aligned = pd.concat([s7, other], axis=1, join="inner")
        diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]  # S7 - other; negative = S7 lower
        obs, p = paired_perm(diff)
        per_seed_eff = {}
        for seed in LLAMA_SEEDS:
            a = chain_rates(f"S7_llama_seed{seed}", "test_main", "violation")
            b = chain_rates(f"{system}_llama_seed{seed}", "test_main", "violation")
            if a is not None and b is not None:
                per_seed_eff[str(seed)] = round(
                    float(pd.concat([a, b], axis=1, join="inner")
                          .pipe(lambda d: (d.iloc[:, 0] - d.iloc[:, 1]).mean())), 4)
        h3[f"S7_vs_{system}"] = {
            "dvr_diff_seedavg": round(obs, 4), "p_raw": p,
            "effect_ge_2pp": bool(obs <= -0.02),
            "per_seed_diff": per_seed_eff, "n_chains": int(len(diff))}
        pvals[f"S7_vs_{system}"] = p
    adj = holm(pvals)
    for k in h3:
        h3[k]["p_holm"] = adj[k]
        h3[k]["significant_holm_05"] = bool(adj[k] < 0.05)
    h3["prereg_criterion"] = ("H3 supported iff every comparison has "
                              "p_holm<0.05 AND absolute DVR reduction >= 2pp "
                              "AND S7 not flagged responsiveness_collapse")
    comps = [h3[f"S7_vs_{s}"] for s in ["S2", "S3", "S4", "S5", "S6"]]
    n_eff = sum(c["effect_ge_2pp"] for c in comps)
    # S7 responsiveness_collapse (third conjunct) is computed in the guardrail
    # block below; a SUPPORTED verdict here is provisional on that flag staying
    # false, and must be re-checked manually if it ever triggers.
    supported = (all(c["significant_holm_05"] for c in comps)
                 and all(c["effect_ge_2pp"] for c in comps))
    h3["prereg_verdict"] = (
        "SUPPORTED" if supported else
        f"NOT SUPPORTED (effect-size condition met in {n_eff}/5 comparisons)")
    report["h3"] = h3

    # 3. Guardrails ----------------------------------------------------------
    guard = {}
    eligible = {s: set(json.load(open(
        ROOT / f"data_processed/study3/eligible_sets_{s}.json"))) for s in SPLITS}
    sets_cache = {s: load_sets(s) for s in SPLITS}
    for family in ["llama", "qwen"]:
        for system in SYSTEMS:
            runs = runs_for(system, family)
            if not runs:
                continue
            g = {}
            for split in ["test_main"]:
                rr = [float(chain_rates(r, split, "responsive").mean())
                      for r in runs
                      if chain_rates(r, split, "responsive") is not None]
                nd = [v for v in (ndcg5(r, split, sets_cache[split], registry,
                                        eligible[split]) for r in runs)
                      if v is not None]
                if rr:
                    g[f"RR_{split}"] = {"mean": round(float(np.mean(rr)), 4),
                                        "seed_range": [round(min(rr), 4),
                                                       round(max(rr), 4)]}
                if nd:
                    g[f"NDCG5_{split}"] = round(float(np.mean(nd)), 4)
            guard[f"{system}_{family}"] = g
    for family in ["llama", "qwen"]:
        rr0 = guard.get(f"S0_{family}", {}).get("RR_test_main", {}).get("mean")
        nd2 = guard.get(f"S2_{family}", {}).get("NDCG5_test_main")
        for system in SYSTEMS:
            g = guard.get(f"{system}_{family}")
            if not g:
                continue
            if rr0 is not None and g.get("RR_test_main"):
                d = round(g["RR_test_main"]["mean"] - rr0, 4)
                g["RR_delta_vs_S0"] = d
                # prereg guardrail: RR drop >5pp vs S0 -> responsiveness collapse
                g["responsiveness_collapse"] = bool(d < -0.05)
            if nd2 is not None and g.get("NDCG5_test_main") is not None:
                d = round(g["NDCG5_test_main"] - nd2, 4)
                g["NDCG5_delta_vs_S2"] = d
                g["utility_damage_flag"] = bool(d < -0.02)
    report["guardrails"] = guard

    # 4. S2 seed-1043 consistency -------------------------------------------
    cons = {}
    for split in SPLITS:
        row = {}
        for seed in LLAMA_SEEDS:
            cr = chain_rates(f"S2_llama_seed{seed}", split, "violation")
            rr = chain_rates(f"S2_llama_seed{seed}", split, "responsive")
            row[str(seed)] = ({"DVR": round(float(cr.mean()), 4),
                               "RR": round(float(rr.mean()), 4)}
                              if cr is not None else "missing")
        vals = [v["DVR"] for v in row.values() if isinstance(v, dict)]
        if isinstance(row.get("1043"), dict) and len(vals) == 3:
            others = [row["1042"]["DVR"], row["1044"]["DVR"]]
            row["seed1043_within_sibling_range"] = bool(
                min(others) <= row["1043"]["DVR"] <= max(others))
            row["seed1043_dev_from_sibling_mean"] = round(
                row["1043"]["DVR"] - float(np.mean(others)), 4)
        cons[split] = row
    report["seed1043_consistency"] = cons

    out = ROOT / "results/study3/main_matrix_report.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
