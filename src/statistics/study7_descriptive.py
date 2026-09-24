"""Study 7 Sec 5b descriptive trade-off analysis (user-required 2026-08-12;
NON-preregistered, no significance tests): does permutation augmentation
trade away strength-semantics OOD ability?

T4 leg: chain-level DVR degradation (test_t4_word - test_main) per seed and
pooled, for S2/S7/S8 (existing raws) vs S2perm/S7perm/S8perm (study7).
4c leg: unseen-transition (2->3, 3->4) pooled violation rate on the full
4-level test_main rendering, for S7c/S8c (existing) vs S7cperm/S8cperm
(study7, levels {1,2,4} + order_aug), per seed and pooled; full-chain DVR
and RR as context.

Output: results/study7/descriptive_tradeoff.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study7_common import (  # noqa: E402
    LLAMA_SEEDS, UNSEEN_TRANSITIONS, chain_rates, load_metrics)


def dvr(run, split, transitions=None):
    cr = chain_rates(run, split, "violation", transitions)
    return None if cr is None else float(cr.mean())


def main():
    out = {"note": "descriptive only, non-preregistered (plan Sec 5b); "
                   "no significance tests; answers whether order_aug trades "
                   "away strength-semantics OOD ability"}

    t4 = {}
    for sysname in ("S2", "S7", "S8", "S2perm", "S7perm", "S8perm"):
        rows, degs = {}, []
        for s in LLAMA_SEEDS:
            run = f"{sysname}_llama_seed{s}"
            a, m = dvr(run, "test_t4_word"), dvr(run, "test_main")
            if a is None or m is None:
                rows[str(s)] = "missing"
                continue
            rows[str(s)] = {"DVR_t4": round(a, 4), "DVR_main": round(m, 4),
                            "degradation": round(a - m, 4)}
            degs.append(a - m)
        t4[sysname] = {"per_seed": rows,
                       "mean_degradation": (round(float(np.mean(degs)), 4)
                                            if degs else None)}
    for sysname in ("S2", "S7", "S8"):
        a = t4[sysname]["mean_degradation"]
        b = t4[f"{sysname}perm"]["mean_degradation"]
        t4[f"{sysname}_delta_perm_minus_nonaug"] = (
            None if None in (a, b) else round(b - a, 4))
    out["t4_leg"] = t4

    c4 = {}
    for sysname in ("S7c", "S8c", "S7cperm", "S8cperm"):
        rows, unseen = {}, []
        for s in LLAMA_SEEDS:
            run = f"{sysname}_llama_seed{s}"
            u = dvr(run, "test_main", transitions=UNSEEN_TRANSITIONS)
            full = dvr(run, "test_main")
            df = load_metrics(run, "test_main")
            if u is None:
                rows[str(s)] = "missing"
                continue
            rows[str(s)] = {"unseen_transition_rate": round(u, 4),
                            "full_chain_DVR": round(full, 4),
                            "RR": round(float(df["responsive"].mean()), 4)}
            unseen.append(u)
        c4[sysname] = {"per_seed": rows,
                       "mean_unseen_rate": (round(float(np.mean(unseen)), 4)
                                            if unseen else None)}
    for sysname in ("S7c", "S8c"):
        a = c4[sysname]["mean_unseen_rate"]
        b = c4[f"{sysname}perm"]["mean_unseen_rate"]
        c4[f"{sysname}_delta_perm_minus_nonaug"] = (
            None if None in (a, b) else round(b - a, 4))
    out["c4_leg"] = c4

    dst = ROOT / "results/study7/descriptive_tradeoff.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
