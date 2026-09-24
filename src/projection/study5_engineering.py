"""Study 5: engineering characterization of Sec 6.6 isotonic projection
(study4_7_plan_draft Study 5 section; user-authorized 2026-08-11).

For each available run (S0/S2/S7/S8, llama 3 seeds + qwen seed1042) x the
four study3 splits, applies per-chain focal PAVA to the cached raw outputs
(nothing deployed; measurement only) and records:
  pre/post DVR, SRR, RR; chains-adjusted / binding-constraint shares;
  newly flattened transitions; NDCG@5 pre/post (rule-utility oracle gains);
  mean |focal rank shift|; Kendall tau pre/post; PAVA wall time per chain;
  staticness diagnostics.
Metric definitions identical to src/projection/isotonic_projection.py
(pilot pre-experiment), applied per run/split.

Output: results/study5/engineering.json
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
from src.projection.isotonic_projection import ndcg_at_k, pava_nondecreasing  # noqa: E402

SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]
SETS_FILE = {"val": "candidate_sets_val.jsonl",
             "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl"}
RUNS = (["S0_llama", "S0_qwen"]
        + [f"{s}_llama_seed{sd}" for s in ("S2", "S7", "S8")
           for sd in (1042, 1043, 1044)]
        + [f"{s}_qwen_seed1042" for s in ("S2", "S7", "S8")])
K = 4


def characterize(rows, sets, eligible, registry, eps, rho):
    chains = {}
    for r in rows:
        if (r["set_id"] in eligible and r["parse_status"] == "ok"
                and r["ranking"]):
            chains.setdefault((r["set_id"], r["template_family"]),
                              {})[r["strength_level"]] = r
    chains = {k: v for k, v in chains.items() if len(v) == K}
    if not chains:
        return None

    t0 = time.time()
    n_adj_chains = n_binding = n_trans = 0
    viol_b = viol_a = resp_b = resp_a = flat_new = srr_b = srr_a = 0
    ndcg_b, ndcg_a, shifts, taus = [], [], [], []
    static_chains = 0
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
        dt = np.array(pava_nondecreasing(d))
        adj = dt - d
        if np.any(np.abs(adj) > 1e-12):
            n_adj_chains += 1
        n_binding += int(np.sum(np.abs(np.diff(dt)) < 1e-12))
        if all(tuple(p[0]) == tuple(packs[0][0]) for p in packs):
            static_chains += 1
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
                base = np.mean([norm_utility(it[a], registry[a])
                                for a in registry if it.get(a) is not None])
                gains[it["item_id"]] = 0.5 * base + LEVEL_W[k + 1] * norm_utility(
                    it[attr], registry[attr])
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
    return {
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
        "solve_time_ms_per_chain": round(1000 * dt_solve / len(chains), 3),
        "identical_ranking_chain_share": round(static_chains / len(chains), 4),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/pilot.yaml"))
    eps, rho = cfg["epsilon"], cfg["rho"]
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    out = {}
    for split in SPLITS:
        sets = {}
        fname = SETS_FILE.get(split, "candidate_sets_test_main.jsonl")
        for line in open(ROOT / "data_processed/study3" / fname):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
        eligible = set(json.loads(
            (ROOT / f"data_processed/study3/eligible_sets_{split}.json").read_text()))
        for run in RUNS:
            src = ROOT / f"results/study3/raw/{run}/{split}.jsonl"
            if not src.exists():
                continue
            rows = [json.loads(l) for l in open(src)]
            res = characterize(rows, sets, eligible, registry, eps, rho)
            if res:
                out.setdefault(run, {})[split] = res
                print(f"{run} x {split}: DVR {res['DVR_before']}->{res['DVR_after']} "
                      f"adj {res['chains_adjusted_share']} "
                      f"ndcg {res['NDCG5_before']}->{res['NDCG5_after']} "
                      f"{res['solve_time_ms_per_chain']}ms/chain", flush=True)
    dst = ROOT / "results/study5/engineering.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
