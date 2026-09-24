"""Study 6 post-hoc diagnostic (NOT preregistered): mechanism of the
candidate-order position sensitivity found in H6-R (report §4.2).

Question: do raw per-item scores move when ONLY the candidate order in the
prompt changes? For each (set_id, level, family) request the five order
splits test_order_p1..p5 render the identical prompt except for candidate
listing order, so any per-item score movement across the five permutations
is order-induced by construction.

Per run we report:
  - per-item score std across the 5 permutations (raw scale);
  - noise_ratio: mean per-item std / mean within-set score spread
    (scale-free; comparable across systems);
  - margin_exceed_frac: fraction of items whose cross-permutation std
    exceeds half the request's min adjacent ranked-score gap (the
    rank-instability-prone fraction);
  - positional bias: per-item deviations delta_{i,perm} = s - mean_perm(s)
    grouped by the item's listing position (0..4) in that permutation,
    plus the share of delta variance explained by position (R^2).

S0 scores are model-stated numbers parsed from text (not head outputs);
S0 rows are descriptive anchors only.

Output: results/study6/order_diagnostic.json (+ printed summary).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/study6/raw"
DATA = ROOT / "data_processed/study6"
PERMS = [f"test_order_p{k}" for k in range(1, 6)]

RUNS = (
    [f"{s}_llama_seed{seed}" for s in ("S2", "S7", "S8") for seed in (1042, 1043, 1044)]
    + [f"{s}_qwen_seed1042" for s in ("S2", "S7", "S8")]
    + ["S0_llama", "S0_qwen"]
)


def load_positions():
    """split -> set_id -> item_id -> listing position in prompt."""
    pos = {}
    for split in PERMS:
        m = {}
        for line in open(DATA / f"candidate_sets_{split}.jsonl"):
            cs = json.loads(line)
            m[cs["set_id"]] = {it["item_id"]: k for k, it in enumerate(cs["items"])}
        pos[split] = m
    return pos


def load_focal_pairs():
    fp = {}
    for line in open(DATA / f"candidate_sets_{PERMS[0]}.jsonl"):
        cs = json.loads(line)
        fp[cs["set_id"]] = tuple(cs["focal_pair"])
    return fp


def load_run(run, raw_root=None, splits=PERMS):
    """perm split -> request key -> {item_id: score}; key = (set,level,family)."""
    out = {}
    for split in splits:
        f = (raw_root or RAW) / run / f"{split}.jsonl"
        if not f.exists():
            return None
        m = {}
        for line in open(f):
            r = json.loads(line)
            if r.get("parse_status") != "ok":
                continue
            ids = r["ranking"]["ids"]
            scores = r["ranking"].get("scores")
            if not scores or len(scores) != len(ids):
                continue
            key = (r["set_id"], r["strength_level"], r["template_family"])
            m[key] = dict(zip(ids, [float(s) for s in scores]))
        out[split] = m
    return out


def pstd(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def analyze(run, positions, focal_pairs, raw_root=None):
    data = load_run(run, raw_root=raw_root)
    if data is None:
        return None
    keys = set.intersection(*(set(data[s]) for s in PERMS))
    item_stds, cent_stds, spreads, n_items = [], [], [], 0
    margin_exceed = 0
    focal_stds, focal_abs, focal_flips, n_focal = [], [], 0, 0
    focal_margins = {}  # (set,level,family) -> [margin per perm]
    deltas_by_pos = defaultdict(list)
    n_score_incomplete = 0
    for key in sorted(keys):
        per_perm = [data[s][key] for s in PERMS]
        items = set(per_perm[0])
        if any(set(p) != items for p in per_perm):
            n_score_incomplete += 1
            continue
        # per-perm common-mode centering: rankings are invariant to a
        # constant shift of all scores in one prompt, so per-item movement
        # is split into common-mode (rank-irrelevant) and centered parts
        centered = []
        req_spreads, req_gaps = [], []
        for p in per_perm:
            mu = sum(p.values()) / len(p)
            centered.append({k: v - mu for k, v in p.items()})
            vals = sorted(p.values(), reverse=True)
            req_spreads.append(pstd(vals))
            req_gaps.append(min(vals[i] - vals[i + 1] for i in range(len(vals) - 1)))
        spread = sum(req_spreads) / len(req_spreads)
        min_gap = sum(req_gaps) / len(req_gaps)
        spreads.append(spread)
        for it in items:
            xs = [p[it] for p in per_perm]
            cs_ = [c[it] for c in centered]
            item_stds.append(pstd(xs))
            sd_c = pstd(cs_)
            cent_stds.append(sd_c)
            n_items += 1
            if sd_c > 0.5 * min_gap:
                margin_exceed += 1
            mu = sum(cs_) / len(cs_)
            for split, c in zip(PERMS, centered):
                pos = positions[split][key[0]].get(it)
                if pos is not None:
                    deltas_by_pos[pos].append(c[it] - mu)
        # focal-pair margin: s_i - s_j is common-mode free by construction
        fi, fj = focal_pairs[key[0]]
        if fi in items and fj in items:
            ms = [p[fi] - p[fj] for p in per_perm]
            focal_stds.append(pstd(ms))
            focal_abs.append(sum(abs(m) for m in ms) / len(ms))
            n_focal += 1
            if any(m > 0 for m in ms) and any(m < 0 for m in ms):
                focal_flips += 1
            focal_margins[key] = ms
    if not item_stds:
        return None
    # DVR-relevant layer: transition increments of the focal margin,
    # delta_k = Delta^{k+1} - Delta^{k} within one permutation. A violation
    # (relational_metrics) is delta_k < -eps, so order-axis DVR is governed
    # by mean increment vs its cross-permutation noise.
    inc_means, inc_stds, inc_viol, n_inc = [], [], 0, 0
    agg_viol, n_agg = 0, 0  # feasibility estimate: mean-score aggregation over perms
    eps = 0.01
    fam_sets = sorted({(k[0], k[2]) for k in focal_margins})
    for set_id, fam in fam_sets:
        lv = sorted({k[1] for k in focal_margins if k[0] == set_id and k[2] == fam})
        for a, b in zip(lv[:-1], lv[1:]):
            ka, kb = (set_id, a, fam), (set_id, b, fam)
            if ka in focal_margins and kb in focal_margins:
                deltas = [mb - ma for ma, mb in
                          zip(focal_margins[ka], focal_margins[kb])]
                inc_means.append(sum(deltas) / len(deltas))
                inc_stds.append(pstd(deltas))
                inc_viol += sum(d < -eps for d in deltas)
                n_inc += len(deltas)
                # inference-time K=5 permutation score aggregation (offline
                # what-if from existing raws; nothing deployed)
                agg_viol += (sum(deltas) / len(deltas)) < -eps
                n_agg += 1
    all_d = [d for ds in deltas_by_pos.values() for d in ds]
    gmu = sum(all_d) / len(all_d)
    ss_tot = sum((d - gmu) ** 2 for d in all_d)
    ss_between = sum(
        len(ds) * ((sum(ds) / len(ds)) - gmu) ** 2 for ds in deltas_by_pos.values())
    srt = sorted(cent_stds)
    mean_spread = sum(spreads) / len(spreads)
    mean_raw = sum(item_stds) / len(item_stds)
    mean_cent = sum(cent_stds) / len(cent_stds)
    return {
        "n_requests": len(keys),
        "n_requests_item_mismatch_skipped": n_score_incomplete,
        "n_items": n_items,
        "mean_item_std_raw": mean_raw,
        "mean_item_std_centered": mean_cent,
        "common_mode_share": 1.0 - (mean_cent ** 2) / (mean_raw ** 2) if mean_raw else 0.0,
        "median_item_std_centered": srt[len(srt) // 2],
        "p90_item_std_centered": srt[int(len(srt) * 0.9)],
        "mean_within_set_spread": mean_spread,
        "noise_ratio_centered": mean_cent / mean_spread,
        "margin_exceed_frac_centered": margin_exceed / n_items,
        "focal_margin_std": sum(focal_stds) / n_focal,
        "focal_margin_mean_abs": sum(focal_abs) / n_focal,
        "focal_margin_noise_ratio": (sum(focal_stds) / n_focal)
                                    / (sum(focal_abs) / n_focal),
        "focal_sign_flip_rate": focal_flips / n_focal,
        "increment_mean": sum(inc_means) / len(inc_means) if inc_means else None,
        "increment_noise_std": sum(inc_stds) / len(inc_stds) if inc_stds else None,
        "increment_noise_over_mean": (sum(inc_stds) / sum(inc_means))
                                     if inc_means and sum(inc_means) > 0 else None,
        "increment_violation_rate": inc_viol / n_inc if n_inc else None,
        "aggregated_k5_violation_rate": agg_viol / n_agg if n_agg else None,
        "position_mean_delta": {
            str(p): sum(ds) / len(ds) for p, ds in sorted(deltas_by_pos.items())},
        "position_r2": ss_between / ss_tot if ss_tot > 0 else 0.0,
    }


POOL_KEYS = ["mean_item_std_centered", "common_mode_share", "noise_ratio_centered",
             "margin_exceed_frac_centered", "focal_margin_noise_ratio",
             "focal_sign_flip_rate", "position_r2", "increment_mean",
             "increment_noise_std", "increment_violation_rate",
             "aggregated_k5_violation_rate"]


def main():
    positions = load_positions()
    focal_pairs = load_focal_pairs()
    out = {"_note": "post-hoc diagnostic, not preregistered; S0 scores are "
                    "model-stated text numbers (descriptive anchor only)",
           "runs": {}}
    for run in RUNS:
        res = analyze(run, positions, focal_pairs)
        if res is None:
            print(f"[skip] {run}: missing/empty order raws")
            continue
        out["runs"][run] = res
        print(f"{run}: n_req={res['n_requests']} "
              f"cent_std={res['mean_item_std_centered']:.4f} "
              f"cm_share={res['common_mode_share']:.3f} "
              f"noise_c={res['noise_ratio_centered']:.4f} "
              f"marg_exc={res['margin_exceed_frac_centered']:.3f} "
              f"focal_nr={res['focal_margin_noise_ratio']:.4f} "
              f"flip={res['focal_sign_flip_rate']:.4f} "
              f"pos_R2={res['position_r2']:.4f} "
              f"inc_mu={res['increment_mean']:.4f} "
              f"inc_sd={res['increment_noise_std']:.4f} "
              f"inc_dvr={res['increment_violation_rate']:.4f} "
              f"agg_dvr={res['aggregated_k5_violation_rate']:.4f}")
    # pooled per system family (trained llama systems, 3 seeds)
    pooled = {}
    for sys in ("S2", "S7", "S8"):
        rs = [v for k, v in out["runs"].items()
              if k.startswith(f"{sys}_llama_seed")]
        if rs:
            w = sum(r["n_items"] for r in rs)
            pooled[f"{sys}_llama"] = {
                k: sum(r[k] * r["n_items"] for r in rs) / w for k in POOL_KEYS}
            pooled[f"{sys}_llama"]["n_seeds"] = len(rs)
    out["pooled_llama"] = pooled
    dst = ROOT / "results/study6/order_diagnostic.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
