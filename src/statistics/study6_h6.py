"""Study 6 statistics: H6 preregistered test + H6-R secondary robustness.

Produces results/study6/h6_report.json with:
  1. h6: preregistered strength-semantics-OOD test (study6_plan.md §4).
     6a-T4: for sys in {S7,S8}, degradation difference Delta_S2 - Delta_sys
     on test_t4_word (chain units (seed, set|family) pooled over 3 llama
     seeds; T = mean(d_t4) - mean(d_main); one-sided sign-flip permutation
     B=5000). 6b-4c: for sys in {S7c,S8c}, paired per-(seed,chain) diff of
     the POOLED unseen-transition violation rate (2->3 and 3->4) on
     test_main, v_S2c - v_sys (one-sided sign-flip, B=5000). Holm over the
     4 comparisons; effect gates 2pp (6a) / 1pp (6b, pilot-thresholded per
     user ruling 2026-08-04); guardrail conjunct. Mechanical verdict.
  2. guardrails: RR (+delta vs S0, collapse flag) and NDCG@5 (delta vs
     S2 / S2c) on the two H6 axes.
  3. h6_r (secondary, NOT merged into the H6 verdict):
     order invariance, distractor tiers, mixed-family chains, near-tied
     tiers (0.5x descriptive only), conflict sets, fixed-order collapse
     (S5 positive control -> detector validity). Per-axis two-sided pooled
     paired permutation S7/S8 vs S2, Holm over the 5 perturbation axes
     per system.
  4. axis_4c_3seed: transition decomposition for all 3 seeds.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_generation.rule_baseline import LEVEL_W, norm_utility  # noqa: E402
from src.projection.isotonic_projection import ndcg_at_k  # noqa: E402
from src.statistics.study3_main_matrix import holm  # noqa: E402

RNG = np.random.default_rng(20260804)
EPS = 0.01
LLAMA_SEEDS = [1042, 1043, 1044]
ORDER_SPLITS = [f"test_order_p{k}" for k in range(1, 6)]
DISTRACT_SPLITS = ["test_distract_d1", "test_distract_d2", "test_distract_d4"]
MIX_SPLITS = ["test_mix12", "test_mix123", "test_mix1234"]
QWEN_ROBUST = set(ORDER_SPLITS) | {"test_neartied"}

# H6-R axis -> its evaluation faces (near-tied 0.5x tier excluded by plan §4)
AXIS_FACES = {"order": ORDER_SPLITS,
              "distract": DISTRACT_SPLITS,
              "mix": MIX_SPLITS,
              "neartied_2x_1x": ["test_neartied"],   # tier filter applied
              "conflict": ["test_conflict"]}


def metrics_path(run, split):
    sysname = run.split("_")[0]
    if split == "test_main":
        if sysname.endswith("c"):
            return ROOT / ("results/study4" if "seed1042" in run else "results/study6"
                           ) / f"metrics_{run}_test_main.parquet"
        return ROOT / f"results/study3/metrics_{run}_test_main.parquet"
    if split == "test_unseen_word":
        return ROOT / f"results/study3/metrics_{run}_{split}.parquet"
    if split == "test_t4_word":
        study = "study6" if "qwen" in run else "study4"
        return ROOT / f"results/{study}/metrics_{run}_{split}.parquet"
    return ROOT / f"results/study6/metrics_{run}_{split}.parquet"


def load_metrics(run, split, tier_filter=None):
    p = metrics_path(run, split)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if tier_filter is not None:
        df = df[df["set_id"].isin(tier_filter)]
    df = df.copy()
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    return df


def chain_rates(run, split, col="violation", tier_filter=None, transitions=None):
    df = load_metrics(run, split, tier_filter)
    if df is None:
        return None
    if transitions is not None:
        df = df[df["transition"].isin(transitions)]
    return df.groupby("chain_id")[col].mean()


def pooled_paired_diff(sys_hi, sys_lo, split, tier_filter=None, transitions=None):
    parts = []
    for seed in LLAMA_SEEDS:
        a = chain_rates(f"{sys_hi}_llama_seed{seed}", split, "violation",
                        tier_filter, transitions)
        b = chain_rates(f"{sys_lo}_llama_seed{seed}", split, "violation",
                        tier_filter, transitions)
        if a is None or b is None:
            return None
        d = pd.concat([a, b], axis=1, join="inner")
        d = d.iloc[:, 0] - d.iloc[:, 1]
        d.index = [f"{seed}|{c}" for c in d.index]
        parts.append(d)
    return pd.concat(parts)


def degradation_perm(d_axis, d_main, n=5000):
    obs = float(d_axis.mean() - d_main.mean())
    da, dm = d_axis.to_numpy(), d_main.to_numpy()
    null = np.empty(n)
    for i in range(n):
        sa = RNG.choice([-1, 1], size=len(da))
        sm = RNG.choice([-1, 1], size=len(dm))
        null[i] = (sa * da).mean() - (sm * dm).mean()
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


def paired_perm(d, n=5000, one_sided=True):
    obs = float(d.mean())
    arr = d.to_numpy()
    signs = RNG.choice([-1, 1], size=(n, len(arr)))
    null = (signs * arr).mean(axis=1)
    if one_sided:
        return obs, float((np.sum(null >= obs) + 1) / (n + 1))
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def seed_summary(vals):
    if not vals:
        return "missing"
    return {"per_seed": {str(k): round(v, 4) for k, v in vals.items()},
            "mean": round(float(np.mean(list(vals.values()))), 4),
            "seed_range": [round(min(vals.values()), 4), round(max(vals.values()), 4)]}


def dvr_entry(system, family, split, tier_filter=None, transitions=None):
    vals = {}
    if system == "S0":
        runs = [f"S0_{family}"]
    elif family == "llama":
        runs = [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
    else:
        runs = [f"{system}_qwen_seed1042"]
    for run in runs:
        cr = chain_rates(run, split, "violation", tier_filter, transitions)
        if cr is not None:
            key = run.rsplit("seed", 1)[1] if "seed" in run else "single"
            vals[key] = float(cr.mean())
    return seed_summary(vals)


def transition_table(run, split):
    """Same decomposition as study4_h4.transition_table, but routed through
    the study6 metrics_path (seed-1043/1044 c-run metrics live in study6)."""
    df = load_metrics(run, split)
    if df is None:
        return None
    out = {"DVR": round(float(df.groupby("chain_id")["violation"].mean().mean()), 4),
           "RR": round(float(df["responsive"].mean()), 4)}
    for tr in ["1->2", "2->3", "3->4"]:
        sub = df[df["transition"] == tr]
        out[f"viol_{tr}"] = round(float(sub["violation"].mean()), 4) if len(sub) else None
    d2 = df[df["transition"] == "2->3"].set_index("chain_id")["delta_k"]
    d4 = df[df["transition"] == "3->4"].set_index("chain_id")["delta_k1"]
    j = pd.concat([d2, d4], axis=1, join="inner")
    out["viol_2->4_jump"] = round(float((j.iloc[:, 1] < j.iloc[:, 0] - EPS).mean()), 4)
    v12 = df[df["transition"] == "1->2"].set_index("chain_id")["violation"]
    sub = pd.concat([v12, (j.iloc[:, 1] < j.iloc[:, 0] - EPS)], axis=1, join="inner")
    out["DVR_seen_subchain_124"] = round(float(sub.mean(axis=1).mean()), 4)
    return out


def load_sets(path):
    sets = {}
    for line in open(path):
        cs = json.loads(line)
        sets[cs["set_id"]] = cs
    return sets


def raw_path(run, split):
    if split == "test_main" and run.split("_")[0].endswith("c"):
        study = "study4" if "seed1042" in run else "study6"
        return ROOT / f"results/{study}/raw/{run}/test_main.jsonl"
    if split == "test_main":
        return ROOT / f"results/study3/raw/{run}/test_main.jsonl"
    if split == "test_t4_word" and "qwen" not in run:
        return ROOT / f"results/study4/raw/{run}/{split}.jsonl"
    if split == "test_t4_word" and run == "S0_qwen":
        return ROOT / f"results/study4/raw/{run}/{split}.jsonl"
    return ROOT / f"results/study6/raw/{run}/{split}.jsonl"


def ndcg5(run, split, sets, registry, eligible, per_level=False):
    p = raw_path(run, split)
    if not p.exists():
        return None
    vals, by_level = [], {1: [], 2: [], 3: [], 4: []}
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
        v = ndcg_at_k(r["ranking"]["ids"], gains)
        vals.append(v)
        by_level[level].append(v)
    if not vals:
        return None
    if per_level:
        return {str(lv): round(float(np.mean(vs)), 4) for lv, vs in by_level.items() if vs}
    return float(np.mean(vals))


def rr_mean(runs, split):
    rr = [float(chain_rates(r, split, "responsive").mean()) for r in runs
          if chain_rates(r, split, "responsive") is not None]
    return round(float(np.mean(rr)), 4) if rr else None


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    tm_sets = load_sets(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl")
    elig_tm = set(json.load(open(ROOT / "data_processed/study3/eligible_sets_test_main.json")))
    elig_t4 = set(json.load(open(ROOT / "data_processed/study4/eligible_sets_test_t4_word.json")))
    sub = json.loads((ROOT / "data_processed/study6/subsample_test_main.json").read_text())
    nt_sets = load_sets(ROOT / "data_processed/study6/candidate_sets_test_neartied.jsonl")
    tiers = {t: {sid for sid, cs in nt_sets.items() if cs["gap_tier"] == t}
             for t in ("2x", "1x", "05x")}
    tier_2x1x = tiers["2x"] | tiers["1x"]

    report = {"epsilon": EPS,
              "notes": {
                  "chain_unit": "(seed, set_id|template_family) for pooled tests",
                  "6a_transparency": "confirmatory retest of a Study 4 exploratory "
                      "finding; llama T4 raw data collected in Study 4 M2 and its "
                      "point estimates were observed before this prereg; frozen "
                      "here: test formulation + qwen replication. NOT a blind "
                      "prereg.",
                  "6b_gate": "1pp gate pilot-thresholded from the Study 4 1-seed "
                      "run (~2/3 of the 1.5pp point estimate), set 2026-08-04 "
                      "before backfill data collection (user ruling 1)"}}

    # ---- 1. H6 preregistered ------------------------------------------------
    h6, pvals = {}, {}
    d_main_full = {s: pooled_paired_diff("S2", s, "test_main") for s in ("S7", "S8")}
    for sysname in ("S7", "S8"):
        key = f"{sysname}_vs_S2_test_t4_word"
        d_t4 = pooled_paired_diff("S2", sysname, "test_t4_word")
        if d_t4 is None or d_main_full[sysname] is None:
            h6[key] = "missing"
            continue
        obs, p = degradation_perm(d_t4, d_main_full[sysname])
        h6[key] = {"axis": "6a_t4", "degradation_diff_pp": round(obs, 4),
                   "p_raw": p, "one_sided": True,
                   "effect_gate_pp": 0.02, "effect_pass": bool(obs >= 0.02),
                   "n_chain_units": [int(len(d_t4)), int(len(d_main_full[sysname]))]}
        pvals[key] = p
    for sysname in ("S7c", "S8c"):
        key = f"{sysname}_vs_S2c_unseen_transitions"
        d = pooled_paired_diff("S2c", sysname, "test_main",
                               transitions=["2->3", "3->4"])
        if d is None:
            h6[key] = "missing"
            continue
        obs, p = paired_perm(d, one_sided=True)
        per_seed = {}
        for seed in LLAMA_SEEDS:
            a = chain_rates(f"S2c_llama_seed{seed}", "test_main",
                            transitions=["2->3", "3->4"])
            b = chain_rates(f"{sysname}_llama_seed{seed}", "test_main",
                            transitions=["2->3", "3->4"])
            if a is not None and b is not None:
                per_seed[str(seed)] = round(float(a.mean() - b.mean()), 4)
        h6[key] = {"axis": "6b_4c", "unseen_transition_diff_pp": round(obs, 4),
                   "p_raw": p, "one_sided": True,
                   "effect_gate_pp": 0.01, "effect_pass": bool(obs >= 0.01),
                   "per_seed": per_seed, "n_chain_units": int(len(d))}
        pvals[key] = p
    if pvals:
        adj = holm(pvals)
        for k in pvals:
            h6[k]["p_holm"] = adj[k]
            h6[k]["significant_holm_05"] = bool(adj[k] < 0.05)
    h6["prereg_criterion"] = (
        "H6 supported iff for each sys in {S7,S8}: both its comparisons "
        "(6a-T4 degradation diff, 6b-4c pooled unseen-transition diff) have "
        "p_holm<0.05 AND pass their effect gates (2pp / 1pp) AND no "
        "responsiveness_collapse flag on the corresponding axis")
    report["h6"] = h6

    # ---- 2. Guardrails on the two H6 axes ----------------------------------
    guard = {}
    # t4 axis (llama + qwen)
    block = {}
    for family in ("llama", "qwen"):
        for system in ("S0", "S2", "S7", "S8"):
            runs = ([f"S0_{family}"] if system == "S0" else
                    [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
                    if family == "llama" else [f"{system}_qwen_seed1042"])
            rr = rr_mean(runs, "test_t4_word")
            nd = [v for v in (ndcg5(r, "test_t4_word", tm_sets, registry, elig_t4)
                              for r in runs) if v is not None]
            if rr is None and not nd:
                continue
            block[f"{system}_{family}"] = {
                "RR": rr, "NDCG5": round(float(np.mean(nd)), 4) if nd else None}
    for family in ("llama", "qwen"):
        rr0 = block.get(f"S0_{family}", {}).get("RR")
        nd2 = block.get(f"S2_{family}", {}).get("NDCG5")
        for k, g in block.items():
            if not k.endswith(family):
                continue
            if rr0 is not None and g.get("RR") is not None:
                g["RR_delta_vs_S0"] = round(g["RR"] - rr0, 4)
                g["responsiveness_collapse"] = bool(g["RR"] - rr0 < -0.05)
            if nd2 is not None and g.get("NDCG5") is not None:
                g["NDCG5_delta_vs_S2"] = round(g["NDCG5"] - nd2, 4)
                g["utility_damage_flag"] = bool(g["NDCG5"] - nd2 < -0.02)
    guard["test_t4_word"] = block
    # 6b axis: test_main for the c-variants
    block = {}
    rr0 = rr_mean(["S0_llama"], "test_main")
    for system in ("S2c", "S7c", "S8c"):
        runs = [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
        rr = rr_mean(runs, "test_main")
        nd = [v for v in (ndcg5(r, "test_main", tm_sets, registry, elig_tm)
                          for r in runs) if v is not None]
        block[system] = {"RR": rr, "NDCG5": round(float(np.mean(nd)), 4) if nd else None}
    nd2 = block.get("S2c", {}).get("NDCG5")
    for system, g in block.items():
        if rr0 is not None and g.get("RR") is not None:
            g["RR_delta_vs_S0"] = round(g["RR"] - rr0, 4)
            g["responsiveness_collapse"] = bool(g["RR"] - rr0 < -0.05)
        if nd2 is not None and g.get("NDCG5") is not None:
            g["NDCG5_delta_vs_S2c"] = round(g["NDCG5"] - nd2, 4)
            g["utility_damage_flag"] = bool(g["NDCG5"] - nd2 < -0.02)
    guard["test_main_c_variants"] = block
    report["guardrails"] = guard

    # H6 verdict (incl. guardrail conjunct)
    per_sys = {}
    for sysname in ("S7", "S8"):
        ka, kb = f"{sysname}_vs_S2_test_t4_word", f"{sysname}c_vs_S2c_unseen_transitions"
        ea, eb = h6.get(ka), h6.get(kb)
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            per_sys[sysname] = "missing"
            continue
        collapse = (guard["test_t4_word"].get(f"{sysname}_llama", {})
                    .get("responsiveness_collapse", False)
                    or guard["test_main_c_variants"].get(f"{sysname}c", {})
                    .get("responsiveness_collapse", False))
        ok = (ea["significant_holm_05"] and ea["effect_pass"]
              and eb["significant_holm_05"] and eb["effect_pass"] and not collapse)
        per_sys[sysname] = {
            "confirmed": bool(ok), "collapse_flag": bool(collapse),
            "detail": {k: {"p_holm": h6[k]["p_holm"], "effect_pass": h6[k]["effect_pass"]}
                       for k in (ka, kb)}}
    h6["per_system"] = per_sys
    confirmed = [s for s, v in per_sys.items() if isinstance(v, dict) and v["confirmed"]]
    if any(v == "missing" for v in per_sys.values()):
        h6["prereg_verdict"] = "INCOMPLETE (missing comparisons)"
    elif len(confirmed) == 2:
        h6["prereg_verdict"] = "SUPPORTED (S7 and S8 both confirmed)"
    else:
        h6["prereg_verdict"] = f"NOT SUPPORTED (confirmed systems: {confirmed or 'none'})"

    # ---- 3. DVR tables for the H6 axes -------------------------------------
    dvr = {}
    for family in ("llama", "qwen"):
        for system in ("S0", "S2", "S7", "S8"):
            e = {}
            for split in ("test_main", "test_unseen_word", "test_t4_word"):
                e[split] = dvr_entry(system, family, split)
            dvr[f"{system}_{family}"] = e
    report["dvr_t4"] = dvr

    axis4c = {"levels_seen_in_training": [1, 2, 4], "seeds": LLAMA_SEEDS}
    for base, retrain in (("S2", "S2c"), ("S7", "S7c"), ("S8", "S8c")):
        rows = {}
        for seed in LLAMA_SEEDS:
            rows[str(seed)] = transition_table(f"{retrain}_llama_seed{seed}", "test_main")
        axis4c[retrain] = {"per_seed": rows,
                           "unseen_transition_rate": dvr_entry(
                               retrain, "llama", "test_main",
                               transitions=["2->3", "3->4"]),
                           "study3_full_levels": dvr_entry(
                               base, "llama", "test_main",
                               transitions=["2->3", "3->4"])}
    report["axis_4c_3seed"] = axis4c

    # ---- 4. H6-R secondary robustness --------------------------------------
    h6r = {"status": "SECONDARY (not merged into the H6 verdict; user ruling 4)"}

    def face_dvr_block(split, tier_filter=None):
        block = {}
        for family in ("llama", "qwen"):
            if family == "qwen" and split not in QWEN_ROBUST:
                continue
            for system in ("S0", "S2", "S7", "S8"):
                block[f"{system}_{family}"] = dvr_entry(system, family, split,
                                                        tier_filter)
        return block

    # order invariance from raw rankings
    def order_invariance(run_p0_path, run, family):
        base = {}
        p0 = run_p0_path
        if not p0.exists():
            return None
        for line in open(p0):
            r = json.loads(line)
            if r["set_id"] in set(sub) and r["parse_status"] == "ok" and r["ranking"]:
                base[(r["set_id"], r["template_family"], r["strength_level"])] = \
                    tuple(r["ranking"]["ids"])
        perms = {}
        for split in ORDER_SPLITS:
            p = ROOT / f"results/study6/raw/{run}/{split}.jsonl"
            if not p.exists():
                return None
            for line in open(p):
                r = json.loads(line)
                if r["parse_status"] != "ok" or not r["ranking"]:
                    continue
                perms.setdefault((r["set_id"], r["template_family"],
                                  r["strength_level"]), []).append(
                    tuple(r["ranking"]["ids"]))
        full = {k: v for k, v in perms.items() if len(v) == 5 and k in base}
        if not full:
            return None
        all_same = np.mean([len(set(v)) == 1 for v in full.values()])
        match_p0 = np.mean([sum(t == base[k] for t in v) / 5 for k, v in full.items()])
        return {"n_requests": len(full),
                "identical_across_5_perms": round(float(all_same), 4),
                "agreement_with_p0": round(float(match_p0), 4)}

    order = {"per_perm_dvr": {s: face_dvr_block(s) for s in ORDER_SPLITS}}
    inv = {}
    for family in ("llama", "qwen"):
        for system in ("S0", "S2", "S7", "S8"):
            runs = ([f"S0_{family}"] if system == "S0" else
                    [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
                    if family == "llama" else [f"{system}_qwen_seed1042"])
            per_run = {}
            for run in runs:
                p0 = (ROOT / f"results/study3/raw/{run}/test_main.jsonl")
                r = order_invariance(p0, run, family)
                if r:
                    per_run[run] = r
            if per_run:
                inv[f"{system}_{family}"] = per_run
    order["invariance"] = inv
    h6r["order"] = order

    # distractor tiers
    dis = {}
    for split in DISTRACT_SPLITS:
        block = face_dvr_block(split)
        for system in ("S0", "S2", "S7", "S8"):
            runs = ([f"S0_llama"] if system == "S0" else
                    [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS])
            e = block.get(f"{system}_llama")
            if isinstance(e, dict):
                e["RR"] = rr_mean(runs, split)
                nd = [v for v in (ndcg5(r, split, tm_sets, registry, set(sub))
                                  for r in runs) if v is not None]
                e["NDCG5"] = round(float(np.mean(nd)), 4) if nd else None
        dis[split] = block
    # no-distractor baseline restricted to the subsample
    dis["baseline_subsample"] = {
        f"{system}_llama": dvr_entry(system, "llama", "test_main",
                                     tier_filter=set(sub))
        for system in ("S0", "S2", "S7", "S8")}
    h6r["distract"] = dis

    # mixed-family chains
    mix = {s: face_dvr_block(s) for s in MIX_SPLITS}
    mix["same_family_baseline_subsample"] = dis["baseline_subsample"]
    h6r["mix"] = mix

    # near-tied tiers
    nt = {}
    for tier, ids in tiers.items():
        nt[f"tier_{tier}"] = face_dvr_block("test_neartied", tier_filter=ids)
    nt["note_05x"] = ("0.5x tier sits below the eligibility design guarantee; "
                      "descriptive only, excluded from the non-inferiority "
                      "criterion (plan §4)")
    h6r["neartied"] = nt

    # conflict sets
    cf_sets = load_sets(ROOT / "data_processed/study6/candidate_sets_test_conflict.jsonl")
    elig_cf = set(json.load(open(ROOT / "data_processed/study6/eligible_sets_test_conflict.json")))
    cf = {"dvr": face_dvr_block("test_conflict")}
    for system in ("S0", "S2", "S7", "S8"):
        runs = (["S0_llama"] if system == "S0" else
                [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS])
        per_level = [ndcg5(r, "test_conflict", cf_sets, registry, elig_cf,
                           per_level=True) for r in runs]
        per_level = [p for p in per_level if p]
        if per_level:
            cf[f"NDCG5_by_level_{system}_llama"] = {
                lv: round(float(np.mean([p[lv] for p in per_level if lv in p])), 4)
                for lv in ("1", "2", "3", "4")}
    h6r["conflict"] = cf

    # per-axis pooled two-sided non-inferiority tests, Holm over 5 per system
    tests = {}
    for sysname in ("S7", "S8"):
        pv = {}
        for axis, faces in AXIS_FACES.items():
            parts = []
            for split in faces:
                tf = tier_2x1x if axis == "neartied_2x_1x" else None
                d = pooled_paired_diff(sysname, "S2", split, tier_filter=tf)
                if d is not None:
                    d.index = [f"{split}|{c}" for c in d.index]
                    parts.append(d)
            if not parts:
                tests[f"{sysname}_vs_S2_{axis}"] = "missing"
                continue
            d = pd.concat(parts)
            obs, p = paired_perm(d, one_sided=False)
            tests[f"{sysname}_vs_S2_{axis}"] = {
                "dvr_diff_sys_minus_S2": round(obs, 4), "p_raw": p,
                "two_sided": True, "noninferior_desc": bool(obs <= 0.0),
                "n_chain_units": int(len(d))}
            pv[f"{sysname}_vs_S2_{axis}"] = p
        if pv:
            adj = holm(pv)
            for k in pv:
                tests[k]["p_holm"] = adj[k]
                tests[k]["significant_holm_05"] = bool(adj[k] < 0.05)
    h6r["noninferiority_tests"] = tests

    # fixed-order collapse detector (existing test_main raw)
    def collapse_row(run, family):
        p = raw_path(run, "test_main")
        if not p.exists():
            return None
        # a fixed-order collapse shows as position-pattern repetition: compare
        # each ranking to the set's listed item order by POSITION indices
        pos_patterns = []
        for line in open(p):
            r = json.loads(line)
            if r["set_id"] not in elig_tm or r["parse_status"] != "ok" or not r["ranking"]:
                continue
            listed = [it["item_id"] for it in tm_sets[r["set_id"]]["items"]]
            pos_patterns.append(tuple(listed.index(x) for x in r["ranking"]["ids"]
                                      if x in listed))
        if not pos_patterns:
            return None
        cnt = Counter(pos_patterns)
        modal, n_modal = cnt.most_common(1)[0]
        probs = np.array([c / len(pos_patterns) for c in cnt.values()])
        entropy = float(-(probs * np.log2(probs)).sum())
        # descriptive companion (NOT part of the frozen flag): fraction of
        # chains whose rankings are identical across all 4 strength levels
        chains = {}
        for line in open(p):
            r = json.loads(line)
            if r["set_id"] in elig_tm and r["parse_status"] == "ok" and r["ranking"]:
                chains.setdefault((r["set_id"], r["template_family"]),
                                  {})[r["strength_level"]] = tuple(r["ranking"]["ids"])
        full = [v for v in chains.values() if len(v) == 4]
        within_fixed = (round(float(np.mean([len(set(v.values())) == 1
                                             for v in full])), 4) if full else None)
        rr = rr_mean([run], "test_main")
        return {"n_requests": len(pos_patterns),
                "within_chain_fixed_rate_descriptive": within_fixed,
                "modal_position_pattern": list(modal),
                "modal_agreement": round(n_modal / len(pos_patterns), 4),
                "position_pattern_entropy_bits": round(entropy, 4),
                "RR": rr}

    fixed = {}
    for family in ("llama", "qwen"):
        rr0 = rr_mean([f"S0_{family}"], "test_main")
        names = [("S0", [f"S0_{family}"])]
        for system in ("S2", "S7", "S8"):
            names.append((system, [f"{system}_llama_seed{s}" for s in LLAMA_SEEDS]
                          if family == "llama" else [f"{system}_qwen_seed1042"]))
        if family == "llama":
            names.append(("S5", ["S5_llama_seed1042"]))
        for system, runs in names:
            for run in runs:
                row = collapse_row(run, family)
                if row is None:
                    continue
                if rr0 is not None and row["RR"] is not None:
                    row["RR_delta_vs_S0"] = round(row["RR"] - rr0, 4)
                    row["collapse_flag"] = bool(
                        row["modal_agreement"] > 0.9
                        and row["RR"] - rr0 < -0.05)
                fixed[run] = row
    s5 = fixed.get("S5_llama_seed1042", {})
    fixed["detector_validity"] = {
        "S5_positive_control_flagged": bool(s5.get("collapse_flag")),
        "note": ("detector valid iff the S5 positive control triggers the "
                 "flag; otherwise only detector failure is reported (plan §4)")}
    fixed["main_systems_flagged"] = sorted(
        r for r, v in fixed.items()
        if isinstance(v, dict) and v.get("collapse_flag")
        and not r.startswith("S5") and not r.startswith("S0"))
    h6r["fixed_order"] = fixed
    report["h6_r"] = h6r

    out = ROOT / "results/study6/h6_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
