"""Study 7 preregistered tests (plan Sec 5, frozen 2026-08-12).

H7-A (component necessity): for each Stage-A-selected ablation X, paired
per-chain-unit (seed, set|family) diff of 4c unseen-transition violation
rate v_ablX - v_S8c, pooled over 3 seeds, one-sided sign-flip permutation
B=5000, effect gate >= 1pp, Holm across selected ablations. Guardrails
(RR vs S0, NDCG@5 vs S8c) reported separately.

H7-P (permutation augmentation): for sys in {S7, S8}, paired per-chain-unit
(seed, set|family, permutation) diff of order-axis chain DVR
v_sys - v_sysperm, 5 permutations x 3 seeds pooled, one-sided sign-flip
B=5000, effect gate >= 2pp, Holm across the 2 systems. Guardrail: test_main
full-chain DVR increase of perm vs non-aug > 0.5pp flags "in-distribution
cost" (flagged system does not count as supporting evidence). Secondary
descriptive: S7perm/S8perm vs S2perm order-axis non-inferiority.

Transparency (plan Sec 4): Stage A / minval point estimates were observed
before this test; H7 is confirmatory retesting, not blind preregistration.

Output: results/study7_statistical_tests.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study7_common import (  # noqa: E402
    LLAMA_SEEDS, ORDER_SPLITS, UNSEEN_TRANSITIONS, chain_rates, holm,
    load_metrics, load_sets, ndcg5, paired_perm)

B = 5000
GATE_A = 0.01
GATE_P = 0.02
GUARD_MAIN = 0.005


def pooled_diff(runs_hi, runs_lo, split_list, transitions=None):
    """Paired per-chain diff hi - lo, pooled over (seed, split)."""
    parts = []
    for (rh, rl) in zip(runs_hi, runs_lo):
        for split in split_list:
            a = chain_rates(rh, split, "violation", transitions)
            b = chain_rates(rl, split, "violation", transitions)
            if a is None or b is None:
                return None
            d = pd.concat([a, b], axis=1, join="inner")
            d = d.iloc[:, 0] - d.iloc[:, 1]
            d.index = [f"{rh}|{split}|{c}" for c in d.index]
            parts.append(d)
    return pd.concat(parts)


def mean_rate(runs, split_list, transitions=None, col="violation"):
    vals = []
    for r in runs:
        for split in split_list:
            cr = chain_rates(r, split, col, transitions)
            if cr is None:
                return None
            vals.append(float(cr.mean()))
    return float(pd.Series(vals).mean())


def rr(run, split="test_main"):
    df = load_metrics(run, split)
    return None if df is None else float(df["responsive"].mean())


def seeds(sysname):
    return [f"{sysname}_llama_seed{s}" for s in LLAMA_SEEDS]


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    sets = load_sets(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl")
    eligible = set(json.loads(
        (ROOT / "data_processed/study3/eligible_sets_test_main.json").read_text()))
    sel = json.load(open(ROOT / "results/study7/stageA_selection.json"))

    out = {"plan": "experiments/study7_plan.md Sec 5 (frozen 2026-08-12)",
           "transparency": (
               "H7 is confirmatory retesting, NOT blind preregistration: "
               "Stage A ablation point estimates (seed 1042) and the minval "
               "permutation-augmentation point estimates (seed 1042, order-"
               "axis improvements 19.0/24.0pp) were observed before these "
               "tests were run; frozen content = test criteria + the "
               "seeds-1043/1044 replication (plan Sec 4)."),
           "B": B}

    # ---- H7-A ---------------------------------------------------------------
    h7a, pvals_a = {"selected_ablations": sel["selected"],
                    "selection_record": "results/study7/stageA_selection.json"}, {}
    rr_s0 = rr("S0_llama")
    ndcg_ref = [ndcg5(r, "test_main", sets, registry, eligible)
                for r in seeds("S8c")]
    ndcg_ref = (sum(ndcg_ref) / len(ndcg_ref)
                if all(v is not None for v in ndcg_ref) else None)
    for abl in sel["selected"]:
        d = pooled_diff(seeds(abl), seeds("S8c"), ["test_main"],
                        transitions=UNSEEN_TRANSITIONS)
        if d is None:
            h7a[abl] = "missing metrics"
            continue
        obs, p = paired_perm(d, n=B, one_sided=True)
        per_seed = {}
        for s in LLAMA_SEEDS:
            a = chain_rates(f"{abl}_llama_seed{s}", "test_main",
                            transitions=UNSEEN_TRANSITIONS)
            b2 = chain_rates(f"S8c_llama_seed{s}", "test_main",
                             transitions=UNSEEN_TRANSITIONS)
            if a is not None and b2 is not None:
                per_seed[str(s)] = round(float(a.mean() - b2.mean()), 4)
        rr_abl = [rr(r) for r in seeds(abl)]
        rr_abl = (sum(rr_abl) / len(rr_abl)
                  if all(v is not None for v in rr_abl) else None)
        nd_abl = [ndcg5(r, "test_main", sets, registry, eligible)
                  for r in seeds(abl)]
        nd_abl = (sum(nd_abl) / len(nd_abl)
                  if all(v is not None for v in nd_abl) else None)
        h7a[abl] = {
            "unseen_transition_diff_pp": round(obs, 4), "p_raw": p,
            "one_sided": True, "effect_gate_pp": GATE_A,
            "effect_pass": bool(obs >= GATE_A), "per_seed": per_seed,
            "n_chain_units": int(len(d)),
            "guardrails": {
                "RR_3seed": None if rr_abl is None else round(rr_abl, 4),
                "RR_delta_vs_S0": (None if None in (rr_abl, rr_s0)
                                   else round(rr_abl - rr_s0, 4)),
                "responsiveness_collapse": (None if None in (rr_abl, rr_s0)
                                            else bool(rr_abl - rr_s0 < -0.05)),
                "NDCG5_3seed": None if nd_abl is None else round(nd_abl, 4),
                "NDCG5_drop_vs_S8c": (None if None in (nd_abl, ndcg_ref)
                                      else round(ndcg_ref - nd_abl, 4)),
                "ndcg_flag": (None if None in (nd_abl, ndcg_ref)
                              else bool(ndcg_ref - nd_abl > 0.02))}}
        pvals_a[abl] = p
    if pvals_a:
        adj = holm(pvals_a)
        for k in pvals_a:
            h7a[k]["p_holm"] = adj[k]
            h7a[k]["significant_holm_05"] = bool(adj[k] < 0.05)
            h7a[k]["component_necessary"] = bool(
                adj[k] < 0.05 and h7a[k]["effect_pass"])
    else:
        h7a["verdict"] = ("no ablation selected by the Stage A rule: every "
                          "single-component removal raised the 4c unseen-"
                          "transition rate by <1pp and tripped no guardrail; "
                          "H7-A is vacuously untested (reported as such)")
    out["h7a"] = h7a

    # ---- H7-P ---------------------------------------------------------------
    h7p, pvals_p = {}, {}
    for sysname in ("S7", "S8"):
        d = pooled_diff(seeds(sysname), seeds(f"{sysname}perm"), ORDER_SPLITS)
        if d is None:
            h7p[sysname] = "missing metrics"
            continue
        obs, p = paired_perm(d, n=B, one_sided=True)
        v_sys = mean_rate(seeds(sysname), ORDER_SPLITS)
        v_perm = mean_rate(seeds(f"{sysname}perm"), ORDER_SPLITS)
        main_sys = mean_rate(seeds(sysname), ["test_main"])
        main_perm = mean_rate(seeds(f"{sysname}perm"), ["test_main"])
        guard = round(main_perm - main_sys, 4)
        h7p[sysname] = {
            "order_DVR_nonaug": round(v_sys, 4),
            "order_DVR_perm": round(v_perm, 4),
            "order_diff_pp": round(obs, 4), "p_raw": p, "one_sided": True,
            "effect_gate_pp": GATE_P, "effect_pass": bool(obs >= GATE_P),
            "n_chain_units": int(len(d)),
            "guardrail_test_main": {
                "DVR_nonaug": round(main_sys, 4), "DVR_perm": round(main_perm, 4),
                "delta_pp": guard,
                "in_distribution_cost_flag": bool(guard > GUARD_MAIN)}}
        pvals_p[sysname] = p
    if pvals_p:
        adj = holm(pvals_p)
        for k in pvals_p:
            h7p[k]["p_holm"] = adj[k]
            h7p[k]["significant_holm_05"] = bool(adj[k] < 0.05)
            h7p[k]["supported"] = bool(
                adj[k] < 0.05 and h7p[k]["effect_pass"]
                and not h7p[k]["guardrail_test_main"]["in_distribution_cost_flag"])
    # secondary descriptive: non-inferiority vs S2perm (and original vs S2)
    sec = {}
    v_s2perm = mean_rate(seeds("S2perm"), ORDER_SPLITS)
    v_s2 = mean_rate(seeds("S2"), ORDER_SPLITS)
    for sysname in ("S7", "S8"):
        v_perm = mean_rate(seeds(f"{sysname}perm"), ORDER_SPLITS)
        v_orig = mean_rate(seeds(sysname), ORDER_SPLITS)
        if None in (v_perm, v_s2perm, v_orig, v_s2):
            sec[sysname] = "missing"
            continue
        sec[sysname] = {
            "order_DVR_sysperm_minus_S2perm": round(v_perm - v_s2perm, 4),
            "order_DVR_sys_minus_S2_original": round(v_orig - v_s2, 4)}
    h7p["secondary_noninferiority_vs_S2perm"] = {
        "descriptive_only": True, "S2perm_order_DVR":
            None if v_s2perm is None else round(v_s2perm, 4),
        "S2_order_DVR": None if v_s2 is None else round(v_s2, 4), **sec}
    out["h7p"] = h7p

    dst = ROOT / "results/study7_statistical_tests.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
