"""Relational metrics over intervention chains (RESEARCH_BRIEF §9).

For each (set, template_family, model, variant) we form the chain of parsed
rankings over strength levels 1..K and evaluate the focal eligible pair
(i dominates j on the target attribute):

  Delta^k = s(i) - s(j)  (score-based, primary)  and rank-based variant.

Outputs one row per chain transition (metrics_by_instance) plus chain-level
rows. Eligibility (§5.4) is enforced upstream by validate_pipeline; here we
additionally require parse_status == ok for every level of the chain.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def rbo(l1, l2, p=0.9):
    """Rank-biased overlap for two full rankings (lists of ids)."""
    depth = min(len(l1), len(l2))
    if not depth:
        return 0.0
    s = sum(p ** (d - 1) * len(set(l1[:d]) & set(l2[:d])) / d for d in range(1, depth + 1))
    return (1 - p) / (1 - p ** depth) * s


def load_results(model, variant):
    path = ROOT / "results" / "raw" / model / f"{variant}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    return rows


def chain_metrics(chain, focal_i, focal_j, eps, rho):
    """chain: {level: result-row}; returns transition rows + chain summary."""
    levels = sorted(chain)
    deltas_s, deltas_r = {}, {}
    for k in levels:
        r = chain[k]["ranking"]
        ids, scores = r["ids"], r["scores"]
        pos = {x: n for n, x in enumerate(ids)}
        deltas_s[k] = scores[pos[focal_i]] - scores[pos[focal_j]]
        deltas_r[k] = pos[focal_j] - pos[focal_i]  # positive = i ranked above j
    transitions = []
    for k in levels[:-1]:
        d1, d2 = deltas_s[k], deltas_s[k + 1]
        transitions.append({
            "transition": f"{k}->{k+1}",
            "delta_k": d1, "delta_k1": d2,
            "violation": d2 < d1 - eps,
            "strict_reversal": (d1 >= 0) and (d2 < 0),
            "responsive": d2 > d1 + rho,
            "rank_delta_k": deltas_r[k], "rank_delta_k1": deltas_r[k + 1],
            "rank_violation": deltas_r[k + 1] < deltas_r[k],
            "rank_strict_reversal": (deltas_r[k] >= 0) and (deltas_r[k + 1] < 0),
        })
    seq = [deltas_s[k] for k in levels]
    nondec = all(seq[t + 1] >= seq[t] - eps for t in range(len(seq) - 1))
    tau = kendalltau(levels, seq).statistic if len(set(seq)) > 1 else (1.0 if nondec else 0.0)
    rho_s = spearmanr(levels, seq).statistic if len(set(seq)) > 1 else (1.0 if nondec else 0.0)
    chain_row = {"chain_nondecreasing": nondec, "chain_kendall_tau": tau,
                 "chain_spearman": rho_s, "chain_fully_consistent": nondec}
    return transitions, chain_row, deltas_s


def offtarget_metrics(chain):
    """Non-target change (§9.5) across adjacent levels over the FULL ranking."""
    levels = sorted(chain)
    flips, taus, rbos = [], [], []
    for k in levels[:-1]:
        ids1 = chain[k]["ranking"]["ids"]
        ids2 = chain[k + 1]["ranking"]["ids"]
        pos1 = {x: n for n, x in enumerate(ids1)}
        pos2 = {x: n for n, x in enumerate(ids2)}
        pairs = list(itertools.combinations(ids1, 2))
        fl = sum(((pos1[a] - pos1[b]) * (pos2[a] - pos2[b])) < 0 for a, b in pairs)
        flips.append(fl / len(pairs))
        taus.append(kendalltau([pos1[x] for x in ids1], [pos2[x] for x in ids1]).statistic)
        rbos.append(rbo(ids1, ids2))
    return {"pairwise_flip_rate": float(np.mean(flips)),
            "kendall_tau_adjacent": float(np.mean(taus)),
            "rbo_adjacent": float(np.mean(rbos))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help="override raw results jsonl path")
    ap.add_argument("--sets", default=None, help="override candidate sets jsonl")
    ap.add_argument("--eligible", default=None, help="override eligible sets json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    eps, rho = cfg["epsilon"], cfg["rho"]

    sets = {}
    with open(args.sets or ROOT / "data_processed" / "candidate_sets.jsonl") as f:
        for line in f:
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
    eligible = json.loads(Path(args.eligible or ROOT / "data_processed" / "eligible_sets.json").read_text())

    rows = [json.loads(l) for l in open(args.raw)] if args.raw else load_results(args.model, args.variant)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["set_id"], r["template_family"]), {})[r["strength_level"]] = r

    inst_rows, n_parse_fail, n_chains = [], 0, 0
    for (set_id, fam), chain in sorted(by_key.items()):
        if set_id not in eligible:
            continue
        if len(chain) < cfg["strength_levels"] or any(
                chain[k]["parse_status"] != "ok" or chain[k]["ranking"] is None for k in chain):
            n_parse_fail += 1
            continue
        n_chains += 1
        cs = sets[set_id]
        fi, fj = cs["focal_pair"]
        transitions, chain_row, _ = chain_metrics(chain, fi, fj, eps, rho)
        ot = offtarget_metrics(chain)
        base = {"model": args.model, "variant": args.variant, "set_id": set_id,
                "set_type": cs["set_type"], "target_attribute": cs["target_attribute"],
                "template_family": fam, **chain_row, **ot}
        for tr in transitions:
            inst_rows.append({**base, **tr})

    df = pd.DataFrame(inst_rows)
    out = args.out or str(ROOT / "results" / f"metrics_{args.model}_{args.variant}.parquet")
    df.to_parquet(out)
    summary = {
        "model": args.model, "variant": args.variant,
        "n_chains_evaluated": n_chains,
        "n_chains_dropped_parse": n_parse_fail,
        "DVR": float(df["violation"].mean()) if len(df) else None,
        "SRR": float(df["strict_reversal"].mean()) if len(df) else None,
        "RR": float(df["responsive"].mean()) if len(df) else None,
        "rank_DVR": float(df["rank_violation"].mean()) if len(df) else None,
        "rank_SRR": float(df["rank_strict_reversal"].mean()) if len(df) else None,
        "chain_fully_consistent_rate": float(df.groupby(["set_id", "template_family"])
                                             ["chain_fully_consistent"].first().mean()) if len(df) else None,
        "pairwise_flip_rate": float(df["pairwise_flip_rate"].mean()) if len(df) else None,
    }
    print(json.dumps(summary, indent=2))
    with open(ROOT / "results" / f"summary_{args.model}_{args.variant}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
