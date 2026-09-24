"""Deterministic rule baseline (§13.1): monotone by construction.

score(i) = 0.5 * mean(normalized utility over all attributes)
         + w(level) * normalized utility on the target attribute,
with w strictly increasing in strength level. Serves as (a) the required rule
baseline and (b) an end-to-end oracle check: its DVR/SRR must be exactly 0.
Writes results in the same raw-results format as run_pilot.py under
results/raw/rule_baseline/{variant}.jsonl (variant is nominal: prompts unused).
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEVEL_W = {1: 0.2, 2: 0.4, 3: 0.7, 4: 1.0}


def norm_utility(value, spec):
    lo, hi = spec["valid_range"]
    x = (value - lo) / (hi - lo)
    return 1 - x if spec["preference_direction"] == "lower" else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="plain")
    args = ap.parse_args()
    registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry.yaml"))

    sets = {}
    with open(ROOT / "data_processed" / "candidate_sets.jsonl") as f:
        for line in f:
            cs = json.loads(line)
            sets[cs["set_id"]] = cs

    out_dir = ROOT / "results" / "raw" / "rule_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(ROOT / "data_processed" / "intervention_specs.jsonl") as fin, \
         open(out_dir / f"{args.variant}.jsonl", "w") as fout:
        for line in fin:
            sp = json.loads(line)
            if sp.get("rejected"):
                continue
            cs = sets[sp["set_id"]]
            attr, level = sp["target_attribute"], sp["strength_level"]
            scored = []
            for it in cs["items"]:
                base = sum(norm_utility(it[a], registry[a]) for a in registry if it.get(a) is not None)
                base /= len(registry)
                s = 0.5 * base + LEVEL_W[level] * norm_utility(it[attr], registry[attr])
                scored.append((it["item_id"], s))
            scored.sort(key=lambda t: -t[1])
            mx = max(s for _, s in scored) or 1.0
            fout.write(json.dumps({
                "request_id": f"{sp['spec_id']}-{args.variant}",
                "spec_id": sp["spec_id"], "set_id": sp["set_id"],
                "model": "rule_baseline", "prompt_variant": args.variant,
                "target_attribute": attr, "strength_level": level,
                "template_family": sp["template_family"],
                "parse_status": "ok",
                "ranking": {"ids": [i for i, _ in scored],
                            "scores": [round(s / mx, 6) for _, s in scored]},
            }) + "\n")
            n += 1
    print(f"rule baseline wrote {n} rows")


if __name__ == "__main__":
    main()
