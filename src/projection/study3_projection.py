"""§6.6 isotonic projection applied to Study 3 raw eval outputs.

Takes any run's raw results dir (zero-shot S0 or a trained system), projects
each eligible chain's focal score-delta sequence onto nondecreasing order via
PAVA (optimal +/-c/2 split on the focal pair, all other scores untouched),
re-sorts the ranking, and writes a new raw dir in the identical schema so
relational_metrics.py / NDCG tooling consume it unchanged.

Usage:
    python src/projection/study3_projection.py --src S0_llama --dst S1_llama
    python src/projection/study3_projection.py --src S8_llama_seed1042 \
        --dst S9_llama_seed1042
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.projection.isotonic_projection import pava_nondecreasing  # noqa: E402

SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]
SETS_FILE = {"val": "candidate_sets_val.jsonl",
             "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl"}
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    for split in SPLITS:
        sets = {}
        fname = SETS_FILE.get(split, "candidate_sets_test_main.jsonl")
        for line in open(ROOT / "data_processed/study3" / fname):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs

        src = ROOT / f"results/study3/raw/{args.src}/{split}.jsonl"
        rows = [json.loads(l) for l in open(src)]
        chains = {}
        for r in rows:
            if r["parse_status"] == "ok" and r["ranking"]:
                chains.setdefault((r["set_id"], r["template_family"]),
                                  {})[r["strength_level"]] = r

        out_rows, n_adjusted = [], 0
        for r in rows:
            chain = chains.get((r["set_id"], r["template_family"]), {})
            if r["parse_status"] != "ok" or not r["ranking"] or len(chain) != K:
                out_rows.append({**r, "model": args.dst})  # pass through unchanged
                continue
            fi, fj = sets[r["set_id"]]["focal_pair"]
            d = []
            for k in range(1, K + 1):
                rk = chain[k]["ranking"]
                pos = {x: n for n, x in enumerate(rk["ids"])}
                d.append(rk["scores"][pos[fi]] - rk["scores"][pos[fj]])
            dt = np.array(pava_nondecreasing(np.array(d)))
            adj = dt[r["strength_level"] - 1] - d[r["strength_level"] - 1]
            rk = r["ranking"]
            s = dict(zip(rk["ids"], rk["scores"]))
            s[fi] += adj / 2
            s[fj] -= adj / 2
            new_ids = sorted(s, key=lambda x: -s[x])
            if abs(adj) > 1e-12:
                n_adjusted += 1
            out_rows.append({**r, "model": args.dst,
                             "request_id": r["request_id"] + "-proj",
                             "ranking": {"ids": new_ids,
                                         "scores": [s[x] for x in new_ids]}})

        out_dir = ROOT / f"results/study3/raw/{args.dst}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(out_rows)} rows, {n_adjusted} adjusted -> {out_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
