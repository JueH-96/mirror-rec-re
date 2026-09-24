"""Study 5: S1 (projection-of-S0) parser-noise degradation curves extended
beyond test_main to the remaining study3 splits (val, test_unseen_word,
test_unseen_attr).

Faithful replication of the study3 parser-noise machinery per split:
  - corruption tables: per request key, prob rate/100; 50% strength
    off-by-one (+/-1 clamped to [1,4], reflected at borders), 50% attribute
    confusion (uniform over the other 3 registry attributes); deterministic
    rng seed 20260731+rate (per split key list, sorted) - same formula as
    experiments/run_parser_noise.py, tables stored per split under
    results/study5/parser_noise/;
  - chain assembly under the PARSED representation and PAVA in parsed order
    (attr-confused requests exit their chain; non-permutation chains pass
    through unprojected) - same logic as study3_projection_noise.py;
  - violations measured against the TRUE chain structure.

Rate 0 = clean projection (equals S1 on that split). Output:
results/study5/noise_extension.json (DVR/SRR pre/post per split x rate,
plus unprojectable-chain share). No raw dirs are written (study3 raw dirs
stay frozen); this is measurement-only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.projection.isotonic_projection import pava_nondecreasing  # noqa: E402
from src.training.scorer_model import ATTRS  # noqa: E402

SPLITS = ["test_main", "val", "test_unseen_word", "test_unseen_attr"]
SETS_FILE = {"val": "candidate_sets_val.jsonl",
             "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl"}
RATES = [0, 5, 10, 20]
SRC = "S0_llama"
K = 4


def build_table(split, rate):
    keys = []
    eligible = set(json.load(open(
        ROOT / f"data_processed/study3/eligible_sets_{split}.json")))
    for line in open(ROOT / f"data_processed/study3/intervention_specs_{split}.jsonl"):
        sp = json.loads(line)
        if sp.get("rejected") or sp["set_id"] not in eligible:
            continue
        keys.append((sp["set_id"], sp["template_family"],
                     sp["strength_level"], sp["target_attribute"]))
    keys.sort()
    rng = np.random.default_rng(20260731 + rate)
    table = {}
    for set_id, fam, level, attr in keys:
        if rng.random() >= rate / 100:
            continue
        if rng.random() < 0.5:
            delta = 1 if rng.random() < 0.5 else -1
            new = level + delta
            if not 1 <= new <= 4:
                new = level - delta
            table[f"{set_id}|{fam}|{level}"] = {"kind": "level", "level": new}
        else:
            others = [a for a in ATTRS if a != attr]
            table[f"{set_id}|{fam}|{level}"] = {
                "kind": "attr", "attr": str(rng.choice(others))}
    out_dir = ROOT / "results/study5/parser_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"corruptions_{split}_rate{rate}.json").write_text(
        json.dumps(table, indent=0))
    return table


def run_split_rate(split, rate, sets, eligible, eps):
    rows = [json.loads(l) for l in
            open(ROOT / f"results/study3/raw/{SRC}/{split}.jsonl")]
    table = build_table(split, rate) if rate else {}

    parsed = {}
    for r in rows:
        key = f"{r['set_id']}|{r['template_family']}|{r['strength_level']}"
        c = table.get(key)
        p_attr, p_level = r["target_attribute"], r["strength_level"]
        if c and c["kind"] == "level":
            p_level = c["level"]
        elif c and c["kind"] == "attr":
            p_attr = c["attr"]
        parsed[r["request_id"]] = (p_attr, p_level)

    chains = {}
    for r in rows:
        if r["set_id"] not in eligible or r["parse_status"] != "ok" or not r["ranking"]:
            continue
        p_attr, p_level = parsed[r["request_id"]]
        if p_attr != r["target_attribute"]:
            continue
        chains.setdefault((r["set_id"], r["template_family"]),
                          {}).setdefault(p_level, []).append(r)

    # true chains for measurement
    true_chains = {}
    for r in rows:
        if r["set_id"] in eligible and r["parse_status"] == "ok" and r["ranking"]:
            true_chains.setdefault((r["set_id"], r["template_family"]),
                                   {})[r["strength_level"]] = r
    true_chains = {k: v for k, v in true_chains.items() if len(v) == K}

    n_broken = 0
    viol_b = viol_a = srr_b = srr_a = n_trans = 0
    for ck, tchain in true_chains.items():
        set_id, fam = ck
        fi, fj = sets[set_id]["focal_pair"]
        d_true = []
        for k in range(1, K + 1):
            rk = tchain[k]["ranking"]
            pos = {x: n for n, x in enumerate(rk["ids"])}
            d_true.append(rk["scores"][pos[fi]] - rk["scores"][pos[fj]])
        d_true = np.array(d_true)

        chain = chains.get(ck, {})
        ok = (sorted(chain.keys()) == list(range(1, K + 1))
              and all(len(v) == 1 for v in chain.values()))
        if ok:
            # PAVA over the PARSED slot ordering
            d_parsed = []
            for k in range(1, K + 1):
                rk = chain[k][0]["ranking"]
                pos = {x: n for n, x in enumerate(rk["ids"])}
                d_parsed.append(rk["scores"][pos[fi]] - rk["scores"][pos[fj]])
            dt = np.array(pava_nondecreasing(np.array(d_parsed)))
            # each TRUE level's post-projection delta = its parsed slot's fit
            d_after = np.empty(K)
            for k in range(1, K + 1):
                p_level = parsed[tchain[k]["request_id"]][1]
                d_after[k - 1] = dt[p_level - 1]
        else:
            n_broken += 1
            d_after = d_true
        for k in range(K - 1):
            n_trans += 1
            viol_b += d_true[k + 1] < d_true[k] - eps
            viol_a += d_after[k + 1] < d_after[k] - eps
            srr_b += (d_true[k] >= 0) and (d_true[k + 1] < 0)
            srr_a += (d_after[k] >= 0) and (d_after[k + 1] < 0)
    return {
        "n_true_chains": len(true_chains),
        "unprojectable_chain_share": round(n_broken / len(true_chains), 4),
        "DVR_before": round(viol_b / n_trans, 4),
        "DVR_after_projection": round(viol_a / n_trans, 4),
        "SRR_before": round(srr_b / n_trans, 4),
        "SRR_after_projection": round(srr_a / n_trans, 4),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/pilot.yaml"))
    eps = cfg["epsilon"]
    out = {"_src": SRC, "_note": "S1-style projection under parser noise; "
           "violations measured against true chains; rate 0 = clean"}
    for split in SPLITS:
        sets = {}
        fname = SETS_FILE.get(split, "candidate_sets_test_main.jsonl")
        for line in open(ROOT / "data_processed/study3" / fname):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
        eligible = set(json.loads(
            (ROOT / f"data_processed/study3/eligible_sets_{split}.json").read_text()))
        for rate in RATES:
            res = run_split_rate(split, rate, sets, eligible, eps)
            out.setdefault(split, {})[str(rate)] = res
            print(f"{split} rate {rate}%: DVR {res['DVR_before']}->"
                  f"{res['DVR_after_projection']} broken {res['unprojectable_chain_share']}",
                  flush=True)
    dst = ROOT / "results/study5/noise_extension.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
