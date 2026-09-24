"""Study 4 statistics: H4 preregistered test + guardrails + descriptive axes.

Produces results/study4/h4_report.json with:
  1. dvr_table: S0/S1/S2/S7/S8/S9 x family x {test_main, test_unseen_word,
     test_unseen_item, test_domain_phone, test_t4_word}; per-seed, mean,
     seed range (test_main / test_unseen_word are the committed Study 3
     metrics; new axes from results/study4/).
  2. h4: preregistered degradation-difference test (study4_plan.md §3).
     For sys in {S7,S8}, axis in {test_unseen_word, test_unseen_item,
     test_domain_phone}: per-chain paired diff d = v_S2 - v_sys, chain
     units (seed, chain) pooled over the 3 llama seeds; statistic
     T = mean(d_axis) - mean(d_main) (= degradation difference
     Delta_S2 - Delta_sys); one-sided sign-flip permutation (B=5000,
     alternative T>0), Holm over the 6 comparisons; effect condition
     T >= 2pp on >= 1 axis per system. Mechanical prereg verdict.
  3. guardrails: RR per system per axis (+ delta vs S0 same axis,
     responsiveness_collapse), NDCG@5 (domain-correct oracle) with delta
     vs S2 same axis.
  4. axis_4c (descriptive, 1 seed): S2c/S7c/S8c on test_main - full-chain
     DVR/RR, unseen-level-3 transition violation rates (2->3, 3->4),
     seen-subchain {1,2,4} violation rate, vs same-seed Study 3 S2/S7/S8.
  5. t4_exploratory (NOT part of the main evidence chain).
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
from src.statistics.study3_main_matrix import holm  # noqa: E402

RNG = np.random.default_rng(20260803)
EPS = 0.01  # configs/pilot.yaml epsilon

AXES = ["test_unseen_word", "test_unseen_item", "test_domain_phone"]
ALL_SPLITS = ["test_main"] + AXES + ["test_t4_word"]
LLAMA_SEEDS = [1042, 1043, 1044]
SYSTEMS = ["S0", "S1", "S2", "S7", "S8", "S9"]
UNSEEDED = {"S0", "S1"}
QWEN = {"S0", "S1", "S2", "S7", "S8"}          # qwen coverage on new axes
QWEN_SPLITS = {"test_unseen_item", "test_domain_phone"}
T4_RUNS = {"S0", "S2", "S7", "S8"}             # llama-only exploratory axis

# split -> (metrics dir prefix, raw dir) ; study3 splits come from Study 3 files
STUDY3_SPLITS = {"test_main", "test_unseen_word"}


def runs_for(system, family):
    if system in UNSEEDED:
        return [f"{system}_{family}"]
    if family == "llama":
        return [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
    return [f"{system}_qwen_seed1042"] if system in QWEN else []


def split_available(system, family, split):
    if family == "qwen" and split not in QWEN_SPLITS:
        return split in STUDY3_SPLITS and system in QWEN  # study3 coverage
    if split == "test_t4_word":
        return family == "llama" and system in T4_RUNS
    if split in STUDY3_SPLITS:
        return True  # study3 committed metrics
    if system == "S9":
        return family == "llama"
    return True


def metrics_path(run, split):
    study = "study3" if split in STUDY3_SPLITS else "study4"
    if run.split("_")[0].endswith("c"):  # 4c retrains (S2c/S7c/S8c) live in study4
        study = "study4"
    return ROOT / f"results/{study}/metrics_{run}_{split}.parquet"


def chain_rates(run, split, col):
    p = metrics_path(run, split)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    return df.groupby("chain_id")[col].mean()


def pooled_paired_diff(sys_hi, sys_lo, split):
    """Per-(seed,chain) diff v_hi - v_lo, pooled across llama seeds."""
    parts = []
    for seed in LLAMA_SEEDS:
        a = chain_rates(f"{sys_hi}_llama_seed{seed}", split, "violation")
        b = chain_rates(f"{sys_lo}_llama_seed{seed}", split, "violation")
        if a is None or b is None:
            return None
        d = pd.concat([a, b], axis=1, join="inner")
        d = (d.iloc[:, 0] - d.iloc[:, 1])
        d.index = [f"{seed}|{c}" for c in d.index]
        parts.append(d)
    return pd.concat(parts)


def degradation_perm(d_axis, d_main, n=5000):
    """One-sided sign-flip permutation for T = mean(d_axis) - mean(d_main)."""
    obs = float(d_axis.mean() - d_main.mean())
    da, dm = d_axis.to_numpy(), d_main.to_numpy()
    null = np.empty(n)
    for i in range(n):
        sa = RNG.choice([-1, 1], size=len(da))
        sm = RNG.choice([-1, 1], size=len(dm))
        null[i] = (sa * da).mean() - (sm * dm).mean()
    p = float((np.sum(null >= obs) + 1) / (n + 1))
    return obs, p


def load_sets(split):
    fname = {"test_unseen_item": "study4/candidate_sets_test_unseen_item.jsonl",
             "test_domain_phone": "study4/candidate_sets_test_domain_phone.jsonl",
             }.get(split, "study3/candidate_sets_test_main.jsonl")
    sets = {}
    for line in open(ROOT / "data_processed" / fname):
        cs = json.loads(line)
        sets[cs["set_id"]] = cs
    return sets


def load_eligible(split):
    study = "study3" if split in STUDY3_SPLITS else "study4"
    return set(json.load(open(ROOT / f"data_processed/{study}/eligible_sets_{split}.json")))


def ndcg5(run, split, sets, registry, eligible):
    study = "study3" if split in STUDY3_SPLITS else "study4"
    p = ROOT / f"results/{study}/raw/{run}/{split}.jsonl"
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


def dvr_entry(system, family, split):
    per_seed = {}
    for run in runs_for(system, family):
        cr = chain_rates(run, split, "violation")
        if cr is not None:
            key = run.rsplit("seed", 1)[1] if "seed" in run else "single"
            per_seed[key] = round(float(cr.mean()), 4)
    if not per_seed:
        return "missing"
    vals = list(per_seed.values())
    return {"per_seed": per_seed, "mean": round(float(np.mean(vals)), 4),
            "seed_range": [min(vals), max(vals)]}


def transition_table(run, split):
    """Full DVR/RR + per-transition violation + seen-subchain {1,2,4} rate."""
    p = metrics_path(run, split)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    out = {"DVR": round(float(df.groupby("chain_id")["violation"].mean().mean()), 4),
           "RR": round(float(df["responsive"].mean()), 4)}
    for tr in ["1->2", "2->3", "3->4"]:
        sub = df[df["transition"] == tr]
        out[f"viol_{tr}"] = round(float(sub["violation"].mean()), 4) if len(sub) else None
    # seen-subchain 2->4: delta at level 2 (delta_k of 2->3) vs level 4 (delta_k1 of 3->4)
    d2 = df[df["transition"] == "2->3"].set_index("chain_id")["delta_k"]
    d4 = df[df["transition"] == "3->4"].set_index("chain_id")["delta_k1"]
    j = pd.concat([d2, d4], axis=1, join="inner")
    out["viol_2->4_jump"] = round(float((j.iloc[:, 1] < j.iloc[:, 0] - EPS).mean()), 4)
    v12 = df[df["transition"] == "1->2"].set_index("chain_id")["violation"]
    sub = pd.concat([v12, (j.iloc[:, 1] < j.iloc[:, 0] - EPS)], axis=1, join="inner")
    out["DVR_seen_subchain_124"] = round(float(sub.mean(axis=1).mean()), 4)
    return out


def main():
    report = {"epsilon": EPS,
              "notes": {
                  "chain_unit": "(seed, set_id|template_family) for pooled tests",
                  "test_main/test_unseen_word": "committed Study 3 metrics",
                  "S1_S9_projection_mode": "generator-truth (oracle) parse; the"
                      " noisy-parse regime was characterized in Study 3 M3"}}

    # 1. DVR table -----------------------------------------------------------
    dvr = {}
    for family in ["llama", "qwen"]:
        for system in SYSTEMS:
            entry = {}
            for split in ALL_SPLITS:
                if not split_available(system, family, split):
                    continue
                entry[split] = dvr_entry(system, family, split)
            if entry:
                dvr[f"{system}_{family}"] = entry
    report["dvr_table"] = dvr

    # 2. H4 ------------------------------------------------------------------
    h4, pvals = {}, {}
    for sysname in ["S7", "S8"]:
        d_main = pooled_paired_diff("S2", sysname, "test_main")
        for axis in AXES:
            key = f"{sysname}_vs_S2_{axis}"
            d_axis = pooled_paired_diff("S2", sysname, axis)
            if d_axis is None or d_main is None:
                h4[key] = "missing"
                continue
            obs, p = degradation_perm(d_axis, d_main)
            per_seed = {}
            for seed in LLAMA_SEEDS:
                a2 = chain_rates(f"S2_llama_seed{seed}", axis, "violation")
                as_ = chain_rates(f"{sysname}_llama_seed{seed}", axis, "violation")
                m2 = chain_rates(f"S2_llama_seed{seed}", "test_main", "violation")
                ms = chain_rates(f"{sysname}_llama_seed{seed}", "test_main", "violation")
                if all(x is not None for x in (a2, as_, m2, ms)):
                    per_seed[str(seed)] = round(
                        float((a2.mean() - as_.mean()) - (m2.mean() - ms.mean())), 4)
            h4[key] = {"degradation_diff_pp": round(obs, 4), "p_raw": p,
                       "one_sided": True, "effect_ge_2pp": bool(obs >= 0.02),
                       "per_seed": per_seed,
                       "n_chain_units": [int(len(d_axis)), int(len(d_main))]}
            pvals[key] = p
    if pvals:
        adj = holm(pvals)
        for k in pvals:
            h4[k]["p_holm"] = adj[k]
            h4[k]["significant_holm_05"] = bool(adj[k] < 0.05)
        h4["prereg_criterion"] = (
            "H4 supported iff all 6 comparisons (S7,S8 x 3 axes) have "
            "p_holm<0.05 AND degradation difference >= 2pp on >= 1 axis for "
            "each of S7 and S8 AND neither S7 nor S8 is flagged "
            "responsiveness_collapse on any H4 axis")
        sig_all = all(h4[k]["significant_holm_05"] for k in pvals)
        eff = {s: any(h4[f"{s}_vs_S2_{a}"]["effect_ge_2pp"] for a in AXES)
               for s in ["S7", "S8"]}
        n_sig = sum(h4[k]["significant_holm_05"] for k in pvals)
        h4["prereg_verdict_pre_guardrail"] = (
            "SUPPORTED" if (sig_all and all(eff.values())) else
            f"NOT SUPPORTED (significance {n_sig}/6; effect>=2pp on >=1 axis: "
            f"S7={eff.get('S7')}, S8={eff.get('S8')})")
    report["h4"] = h4

    # 3. Guardrails per axis -------------------------------------------------
    registries = {"laptop": yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml")),
                  "phone": yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry_phones.yaml"))}
    guard = {}
    for split in AXES:
        reg = registries["phone" if split == "test_domain_phone" else "laptop"]
        sets = load_sets(split)
        eligible = load_eligible(split)
        block = {}
        for family in ["llama", "qwen"]:
            for system in SYSTEMS:
                if not split_available(system, family, split) or \
                        (family == "qwen" and split not in QWEN_SPLITS):
                    continue
                runs = runs_for(system, family)
                rr = [float(chain_rates(r, split, "responsive").mean()) for r in runs
                      if chain_rates(r, split, "responsive") is not None]
                nd = [v for v in (ndcg5(r, split, sets, reg, eligible) for r in runs)
                      if v is not None]
                if not rr and not nd:
                    continue
                g = {}
                if rr:
                    g["RR"] = {"mean": round(float(np.mean(rr)), 4),
                               "seed_range": [round(min(rr), 4), round(max(rr), 4)]}
                if nd:
                    g["NDCG5"] = round(float(np.mean(nd)), 4)
                block[f"{system}_{family}"] = g
        for family in ["llama", "qwen"]:
            rr0 = block.get(f"S0_{family}", {}).get("RR", {}).get("mean")
            nd2 = block.get(f"S2_{family}", {}).get("NDCG5")
            for system in SYSTEMS:
                g = block.get(f"{system}_{family}")
                if not g:
                    continue
                if rr0 is not None and g.get("RR"):
                    d = round(g["RR"]["mean"] - rr0, 4)
                    g["RR_delta_vs_S0"] = d
                    g["responsiveness_collapse"] = bool(d < -0.05)
                if nd2 is not None and g.get("NDCG5") is not None:
                    d = round(g["NDCG5"] - nd2, 4)
                    g["NDCG5_delta_vs_S2"] = d
                    g["utility_damage_flag"] = bool(d < -0.02)
        guard[split] = block
    report["guardrails"] = guard

    # H4 final verdict incl. guardrail conjunct
    if pvals:
        collapse = any(
            guard.get(a, {}).get(f"{s}_llama", {}).get("responsiveness_collapse")
            for s in ["S7", "S8"] for a in AXES)
        v = report["h4"]["prereg_verdict_pre_guardrail"]
        report["h4"]["prereg_verdict"] = (
            v if not collapse else
            v + " [OVERRIDDEN: responsiveness_collapse flagged on an H4 axis]"
            if v == "SUPPORTED" else v)
        report["h4"]["h4_axis_collapse_flagged"] = bool(collapse)

    # 4. 4c descriptive axis -------------------------------------------------
    axis4c = {"note": ("1 seed (1042) per user ruling (a); descriptive only, "
                       "not part of H4; 3-seed backfill required before any "
                       "conclusive claim if selected as Study 6/7 second "
                       "evaluation surface"),
              "levels_seen_in_training": [1, 2, 4]}
    for base, retrain in [("S2", "S2c"), ("S7", "S7c"), ("S8", "S8c")]:
        axis4c[retrain] = {
            "retrained_lv124": transition_table(f"{retrain}_llama_seed1042", "test_main"),
            "study3_full_levels_same_seed": transition_table(f"{base}_llama_seed1042", "test_main")}
    report["axis_4c"] = axis4c

    # 5. T4 exploratory ------------------------------------------------------
    t4 = {"status": "EXPLORATORY (not preregistered; excluded from the main "
                    "evidence chain per user ruling); calibration gate passed "
                    "(results/study4/t4_calibration.json)"}
    for system in ["S0", "S2", "S7", "S8"]:
        row = {}
        for split in ["test_main", "test_unseen_word", "test_t4_word"]:
            row[split] = dvr_entry(system, "llama", split)
        rr = [float(chain_rates(r, "test_t4_word", "responsive").mean())
              for r in runs_for(system, "llama")
              if chain_rates(r, "test_t4_word", "responsive") is not None]
        if rr:
            row["RR_test_t4_word"] = round(float(np.mean(rr)), 4)
        t4[f"{system}_llama"] = row
    report["t4_exploratory"] = t4

    out = ROOT / "results/study4/h4_report.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
