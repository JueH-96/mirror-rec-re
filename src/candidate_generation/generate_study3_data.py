"""Study 3 data generation (M0) — approved plan experiments/study3_plan.md.

Splits (five-way leakage isolation per §7.7: item ids, base contexts,
templates, seeds, intervention chains):
  train             1600 sets, attrs {price,weight,battery_life}, types A/C, families T1/T2
  val                200 sets, same distribution as train
  test_main          400 sets, held-in attrs, types A/C/B(real; price/weight only), families T1/T2
  test_unseen_word   = test_main sets rendered with held-out family T3
  test_unseen_attr   200 sets targeting held-out attribute storage, types A/C/B, families T1/T2

Seed base 20260728* — disjoint from pilot (42). Base contexts are assigned
per split from disjoint pools. Set-id prefixes keep item ids disjoint from
pilot (CS*) and across splits.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.generate_candidates import (  # noqa: E402
    build_set_A, build_set_B, build_set_C, load_registry)
from src.intervention_generation.generate_interventions import (  # noqa: E402
    ATTR_NOUN, ATTR_PHRASES, TEMPLATES, validate_surface)

HELD_IN_ATTRS = ["price", "weight", "battery_life"]
HELD_OUT_ATTR = "storage"

BASE_CONTEXTS = {
    "train": [
        "I am looking for a laptop for everyday office work (spreadsheets, email, video calls).",
        "I need a laptop for university coursework (writing, research, online lectures).",
        "I want a laptop for managing my small business (invoicing, inventory, web).",
    ],
    "val": [
        "I am after a laptop for remote software support work (ticketing, terminals, calls).",
    ],
    "test": [
        "I need a laptop for frequent business travel (presentations, email on the go).",
        "I am looking for a laptop for graduate research (data analysis, writing, reading).",
    ],
}

SPLITS = {
    "train":            {"n": 1600, "attrs": HELD_IN_ATTRS, "types": {"A": 800, "C": 800}, "families": ["T1", "T2"], "ctx": "train", "seed": 20260728, "prefix": "S3TR"},
    "val":              {"n": 200,  "attrs": HELD_IN_ATTRS, "types": {"A": 100, "C": 100}, "families": ["T1", "T2"], "ctx": "val",   "seed": 20260729, "prefix": "S3VA"},
    "test_main":        {"n": 400,  "attrs": HELD_IN_ATTRS, "types": {"A": 160, "C": 120, "B": 120}, "families": ["T1", "T2"], "ctx": "test", "seed": 20260730, "prefix": "S3TE"},
    "test_unseen_attr": {"n": 200,  "attrs": [HELD_OUT_ATTR], "types": {"A": 80, "C": 60, "B": 60}, "families": ["T1", "T2"], "ctx": "test", "seed": 20260731, "prefix": "S3UA"},
}
# test_unseen_word: rendered from test_main sets with family T3 (no new sets).


def gen_split(name, spec, registry, catalog, out_dir):
    rng = random.Random(spec["seed"])
    plan = []
    for stype, count in spec["types"].items():
        for n in range(count):
            if stype == "B":
                battrs = [a for a in spec["attrs"] if catalog is not None and a in catalog.columns]
                if not battrs:
                    battrs = spec["attrs"]
                plan.append((stype, battrs[n % len(battrs)]))
            else:
                plan.append((stype, spec["attrs"][n % len(spec["attrs"])]))
    rng.shuffle(plan)
    ctxs = BASE_CONTEXTS[spec["ctx"]]
    n_b_fallback = 0
    sets_path = out_dir / f"candidate_sets_{name}.jsonl"
    with open(sets_path, "w") as f:
        for k, (stype, attr) in enumerate(plan):
            set_id = f"{spec['prefix']}{k:05d}"
            built = None
            if stype == "B":
                built = build_set_B(rng, registry, attr, set_id, catalog) if catalog is not None else None
                if built is None:
                    built, stype = build_set_A(rng, registry, attr, set_id), "A_fallback_from_B"
                    n_b_fallback += 1
            elif stype == "C":
                built = build_set_C(rng, registry, attr, set_id)
            else:
                built = build_set_A(rng, registry, attr, set_id)
            items, focal = built
            f.write(json.dumps({
                "set_id": set_id, "split": name, "set_type": stype,
                "target_attribute": attr, "focal_pair": list(focal),
                "base_context": ctxs[k % len(ctxs)], "items": items,
            }) + "\n")
    return sets_path, n_b_fallback


def gen_interventions(name, sets_path, families, out_dir):
    n_reject = 0
    out = out_dir / f"intervention_specs_{name}.jsonl"
    with open(sets_path) as fin, open(out, "w") as fout:
        for line in fin:
            cs = json.loads(line)
            attr = cs["target_attribute"]
            for level in range(1, 5):
                for fam in families:
                    tmpl = TEMPLATES[level][fam]
                    text = tmpl.format(attr=ATTR_PHRASES[attr],
                                       noun=ATTR_NOUN[attr].capitalize() if tmpl.startswith("{noun}") else ATTR_NOUN[attr])
                    spec = {
                        "spec_id": f"{cs['set_id']}-L{level}-{fam}",
                        "set_id": cs["set_id"], "split": name,
                        "target_attribute": attr, "polarity": "positive",
                        "strength_level": level, "template_family": fam,
                        "base_context": cs["base_context"], "surface_form": text,
                    }
                    ok, reason = validate_surface(spec, text)
                    spec["validation"] = reason
                    if not ok:
                        spec["rejected"] = True
                        n_reject += 1
                    fout.write(json.dumps(spec) + "\n")
    return out, n_reject


def main():
    registry = load_registry()
    out_dir = ROOT / "data_processed" / "study3"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = None
    real = ROOT / "data_processed" / "real_laptops.csv"
    if real.exists():
        catalog = pd.read_csv(real).dropna(axis=1, how="all")

    summary = {}
    for name, spec in SPLITS.items():
        sets_path, n_fb = gen_split(name, spec, registry, catalog, out_dir)
        _, n_rej = gen_interventions(name, sets_path, spec["families"], out_dir)
        summary[name] = {"sets": spec["n"], "b_fallback": n_fb, "rejected_surfaces": n_rej}
    # unseen-wording split: T3 renderings over test_main candidate sets
    _, n_rej = gen_interventions("test_unseen_word", out_dir / "candidate_sets_test_main.jsonl",
                                 ["T3"], out_dir)
    summary["test_unseen_word"] = {"sets": "reuses test_main", "rejected_surfaces": n_rej}
    print(json.dumps(summary, indent=2))
    total_rej = sum(v["rejected_surfaces"] for v in summary.values())
    if total_rej:
        sys.exit(f"FATAL: {total_rej} surface forms failed round-trip validation")


if __name__ == "__main__":
    main()
