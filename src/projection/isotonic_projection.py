"""§6.6 inference-time constraint projection on cached pilot outputs.

For each eligible chain (set x template_family) and its focal pair (i,j) with
i dominating j on the target attribute, project the per-level score-delta
sequence d^1..d^K onto nondecreasing sequences, minimizing sum of squared
score adjustments. With the L2 objective and the optimal +/-c/2 split between
s_i and s_j, this reduces exactly to isotonic (PAVA) regression on d; all
other items' scores are untouched (§6.6: constraints only on eligible pairs).

Outputs results/projection_results.json with, per model (plain variant):
- DVR/SRR before vs after (after must be 0 on focal transitions by construction)
- active constraints: share of chains adjusted, share of transitions binding
- responsiveness RR before vs after; flattened-transition share
- NDCG@5 (gains = deterministic rule-utility oracle) before vs after; mean
  |rank shift| of focal items; Kendall tau pre/post per level
- solve time
- staticness diagnostics (identical-ranking chains, DVR conditional on
  responsive chains) to answer whether GLM's low DVR is a static artifact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.rule_baseline import LEVEL_W, norm_utility  # noqa: E402

MODELS = ["qwen2.5-14b-instruct", "llama-3.1-8b-instruct", "glm-4-9b-chat"]
VARIANT = "plain"
K = 4


def pava_nondecreasing(y):
    """Pool-adjacent-violators for nondecreasing fit, unit weights."""
    y = list(map(float, y))
    blocks = [[v, 1] for v in y]  # [mean, weight]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2 = out.pop()
            m1, w1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    fit = []
    for m, w in out:
        fit.extend([m] * w)
    return fit


def ndcg_at_k(ranked_ids, gains, k=5):
    g = [gains[x] for x in ranked_ids[:k]]
    dcg = sum(v / np.log2(r + 2) for r, v in enumerate(g))
    ig = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(v / np.log2(r + 2) for r, v in enumerate(ig))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "pilot.yaml"))
    eps, rho = cfg["epsilon"], cfg["rho"]
    registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry.yaml"))
    eligible = set(json.loads((ROOT / "data_processed" / "eligible_sets.json").read_text()))
    sets = {}
    for line in open(ROOT / "data_processed" / "candidate_sets.jsonl"):
        cs = json.loads(line)
        sets[cs["set_id"]] = cs

    report = {}
    for model in MODELS:
        rows = [json.loads(l) for l in open(ROOT / "results" / "raw" / model / f"{VARIANT}.jsonl")]
        chains = {}
        for r in rows:
            if r["set_id"] in eligible and r["parse_status"] == "ok" and r["ranking"]:
                chains.setdefault((r["set_id"], r["template_family"]), {})[r["strength_level"]] = r
        chains = {k: v for k, v in chains.items() if len(v) == K}

        t0 = time.time()
        n_adj_chains = n_binding = n_trans = 0
        viol_b = viol_a = resp_b = resp_a = flat_new = 0
        srr_b = srr_a = 0
        ndcg_b, ndcg_a, shifts, taus = [], [], [], []
        static_chains = 0
        resp_chain_viol, resp_chain_trans = 0, 0
        for (set_id, fam), chain in chains.items():
            cs = sets[set_id]
            fi, fj = cs["focal_pair"]
            attr = cs["target_attribute"]
            d, packs = [], []
            for k in range(1, K + 1):
                rk = chain[k]["ranking"]
                pos = {x: n for n, x in enumerate(rk["ids"])}
                d.append(rk["scores"][pos[fi]] - rk["scores"][pos[fj]])
                packs.append((rk["ids"], rk["scores"], pos))
            d = np.array(d)
            dt = pava_nondecreasing(d)
            dt = np.array(dt)
            adj = dt - d
            if np.any(np.abs(adj) > 1e-12):
                n_adj_chains += 1
            n_binding += int(np.sum(np.abs(np.diff(dt)) < 1e-12))

            rank_seqs = [tuple(p[0]) for p in packs]
            if all(s == rank_seqs[0] for s in rank_seqs):
                static_chains += 1
            else:
                resp_chain_trans += K - 1
                resp_chain_viol += int(np.sum(d[1:] < d[:-1] - eps))

            for k in range(K - 1):
                n_trans += 1
                viol_b += d[k + 1] < d[k] - eps
                viol_a += dt[k + 1] < dt[k] - eps
                srr_b += (d[k] >= 0) and (d[k + 1] < 0)
                srr_a += (dt[k] >= 0) and (dt[k + 1] < 0)
                resp_b += d[k + 1] > d[k] + rho
                resp_a += dt[k + 1] > dt[k] + rho
                flat_new += (abs(dt[k + 1] - dt[k]) < 1e-12) and (abs(d[k + 1] - d[k]) >= 1e-12)

            for k in range(K):
                ids, scores, pos = packs[k]
                s = dict(zip(ids, scores))
                s[fi] = s[fi] + adj[k] / 2
                s[fj] = s[fj] - adj[k] / 2
                new_order = sorted(ids, key=lambda x: -s[x])
                gains = {}
                for it in cs["items"]:
                    base = np.mean([norm_utility(it[a], registry[a]) for a in registry if it.get(a) is not None])
                    gains[it["item_id"]] = 0.5 * base + LEVEL_W[k + 1] * norm_utility(it[attr], registry[attr])
                ndcg_b.append(ndcg_at_k(ids, gains))
                ndcg_a.append(ndcg_at_k(new_order, gains))
                np_pos = {x: n for n, x in enumerate(new_order)}
                shifts.append(np.mean([abs(np_pos[x] - pos[x]) for x in (fi, fj)]))
                conc = disc = 0
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        s1 = pos[ids[a]] - pos[ids[b]]
                        s2 = np_pos[ids[a]] - np_pos[ids[b]]
                        conc += (s1 * s2) > 0
                        disc += (s1 * s2) < 0
                taus.append((conc - disc) / (conc + disc) if conc + disc else 1.0)
        dt_solve = time.time() - t0

        report[model] = {
            "n_chains": len(chains),
            "DVR_before": round(viol_b / n_trans, 4), "DVR_after": round(viol_a / n_trans, 4),
            "SRR_before": round(srr_b / n_trans, 4), "SRR_after": round(srr_a / n_trans, 4),
            "RR_before": round(resp_b / n_trans, 4), "RR_after": round(resp_a / n_trans, 4),
            "chains_adjusted_share": round(n_adj_chains / len(chains), 4),
            "binding_constraint_share": round(n_binding / n_trans, 4),
            "newly_flattened_transition_share": round(flat_new / n_trans, 4),
            "NDCG5_before": round(float(np.mean(ndcg_b)), 4),
            "NDCG5_after": round(float(np.mean(ndcg_a)), 4),
            "mean_abs_focal_rank_shift": round(float(np.mean(shifts)), 4),
            "mean_kendall_tau_pre_post": round(float(np.mean(taus)), 4),
            "solve_time_s_total": round(dt_solve, 3),
            "solve_time_ms_per_chain": round(1000 * dt_solve / len(chains), 3),
            "staticness": {
                "identical_ranking_chain_share": round(static_chains / len(chains), 4),
                "DVR_on_responsive_chains_only": round(resp_chain_viol / resp_chain_trans, 4) if resp_chain_trans else None,
                "n_responsive_chains": len(chains) - static_chains,
            },
        }
    with open(ROOT / "results" / "projection_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
