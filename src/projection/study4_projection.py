"""§6.6 isotonic projection over Study 4 raw eval outputs.

Same PAVA mechanics as study3_projection.py, with Study 4 split locations
(results/study4/raw). Splits are selectable so S0-derived S1 (three splits,
family-dependent) and S8-derived S9 (two splits) share one script.

Usage:
    python src/projection/study4_projection.py --src S0_llama --dst S1_llama \
        --splits test_unseen_item,test_domain_phone
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.projection.isotonic_projection import pava_nondecreasing  # noqa: E402

SETS = {
    "test_unseen_item": "data_processed/study4/candidate_sets_test_unseen_item.jsonl",
    "test_domain_phone": "data_processed/study4/candidate_sets_test_domain_phone.jsonl",
    "test_t4_word": "data_processed/study3/candidate_sets_test_main.jsonl",
}
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="test_unseen_item,test_domain_phone")
    args = ap.parse_args()

    for split in args.splits.split(","):
        sets = {}
        for line in open(ROOT / SETS[split]):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs

        src = ROOT / f"results/study4/raw/{args.src}/{split}.jsonl"
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

        out_dir = ROOT / f"results/study4/raw/{args.dst}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(out_rows)} rows, {n_adjusted} adjusted -> {out_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
