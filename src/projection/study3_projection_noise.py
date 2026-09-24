"""S1 (isotonic projection) under parser noise (study3_plan §0-P2).

The projection pipeline depends on the parsed (attribute, strength-level) to
assemble the intervention chain before applying PAVA. Under a noisy parser
(shared corruption tables from experiments/run_parser_noise.py):

  - a request whose LEVEL is parsed off-by-one lands in the wrong slot; if
    the parsed levels of a chain are no longer a permutation of 1..4 the
    chain cannot be assembled and the whole chain passes through UNPROJECTED
    (missed constraint); if they are a permutation (two corruptions that
    swap), PAVA is applied in the wrong order (constraint applied wrongly);
  - a request whose ATTRIBUTE is confused exits its true chain (single-
    utterance-per-level regime: one stray request cannot form a 4-level
    chain for the confused attribute), so its true chain is incomplete and
    passes through unprojected.

Violations are then measured against the TRUE chain structure.

Usage:
  python src/projection/study3_projection_noise.py --rate 10 \
      --src S0_llama --dst S1N10_llama
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.projection.isotonic_projection import pava_nondecreasing  # noqa: E402

SPLIT = "test_main"
K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, required=True)
    ap.add_argument("--src", default="S0_llama")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    table = json.loads((ROOT / "results/study3/parser_noise" /
                        f"corruptions_rate{args.rate}.json").read_text())

    sets = {}
    for line in open(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl"):
        cs = json.loads(line)
        sets[cs["set_id"]] = cs

    rows = [json.loads(l) for l in
            open(ROOT / f"results/study3/raw/{args.src}/{SPLIT}.jsonl")]

    # parsed view of every request
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

    # assemble chains under the PARSED representation (true attribute only;
    # attr-confused requests exit their chain)
    chains = {}
    for r in rows:
        if r["parse_status"] != "ok" or not r["ranking"]:
            continue
        p_attr, p_level = parsed[r["request_id"]]
        if p_attr != r["target_attribute"]:
            continue  # stray request: cannot be located in any chain
        chains.setdefault((r["set_id"], r["template_family"]),
                          {}).setdefault(p_level, []).append(r)

    out_rows, n_proj, n_broken = [], 0, 0
    seen_chain = set()
    for r in rows:
        ck = (r["set_id"], r["template_family"])
        chain = chains.get(ck, {})
        # projectable iff parsed levels form exactly one request per level 1..4
        ok = (sorted(chain.keys()) == list(range(1, K + 1))
              and all(len(v) == 1 for v in chain.values()))
        if r["parse_status"] != "ok" or not r["ranking"] or not ok:
            if not ok and ck not in seen_chain:
                seen_chain.add(ck)
                n_broken += 1
            out_rows.append({**r, "model": args.dst})
            continue
        fi, fj = sets[r["set_id"]]["focal_pair"]
        d = []
        for k in range(1, K + 1):
            rk = chain[k][0]["ranking"]
            pos = {x: n for n, x in enumerate(rk["ids"])}
            d.append(rk["scores"][pos[fi]] - rk["scores"][pos[fj]])
        dt = np.array(pava_nondecreasing(np.array(d)))
        # this request's slot in the PARSED ordering
        p_level = parsed[r["request_id"]][1]
        adj = dt[p_level - 1] - d[p_level - 1]
        rk = r["ranking"]
        s = dict(zip(rk["ids"], rk["scores"]))
        s[fi] += adj / 2
        s[fj] -= adj / 2
        new_ids = sorted(s, key=lambda x: -s[x])
        if abs(adj) > 1e-12:
            n_proj += 1
        out_rows.append({**r, "model": args.dst,
                         "request_id": r["request_id"] + f"-N{args.rate}",
                         "ranking": {"ids": new_ids,
                                     "scores": [s[x] for x in new_ids]}})

    out_dir = ROOT / f"results/study3/raw/{args.dst}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{SPLIT}.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"rate {args.rate}%: {len(out_rows)} rows, {n_proj} adjusted, "
          f"{n_broken} chains unprojectable -> {out_dir / (SPLIT + '.jsonl')}")


if __name__ == "__main__":
    main()
