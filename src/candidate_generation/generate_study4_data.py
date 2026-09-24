"""Study 4 data generation (M0) — approved plan experiments/study4_plan.md.

Splits:
  test_unseen_item   200 sets (A 80 / C 60 / B 60), laptop domain, held-in
                     attrs, families T1/T2; new seed/prefix/base contexts;
                     B rows drawn only from catalog rows whose
                     (price, weight, storage) triple never appeared in any
                     Study 3 B-type item (unseen real items).
  test_domain_phone  150 sets (A 60 / C 45 / B 45), phone domain (real
                     catalog data_processed/real_phones.csv + phone
                     registry), held-in attrs, families T1/T2.
  test_t4_word       T4 (exploratory) renderings over the Study 3 test_main
                     candidate sets — only generated if the T4 calibration
                     gate passed (results/study4/t4_calibration.json).

Seed base 20260803+ — disjoint from pilot (42) and Study 3 (20260728-31).
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
    SECONDARY, build_set_A, build_set_B, build_set_C, load_registry)
from src.intervention_generation.generate_interventions import (  # noqa: E402
    ATTR_NOUN, ATTR_PHRASES, TEMPLATES, validate_surface)
from src.intervention_generation.t4_family import (  # noqa: E402
    render_t4, validate_t4_surface)

HELD_IN_ATTRS = ["price", "weight", "battery_life"]
OUT = ROOT / "data_processed" / "study4"

BASE_CONTEXTS = {
    "test_unseen_item": [
        "I need a laptop for hybrid teaching (slides, video recording, grading).",
        "I am shopping for a laptop for freelance graphic design (photo editing, client demos).",
        "I want a laptop for community nonprofit administration (documents, budgeting, email).",
    ],
    "test_domain_phone": [
        "I am looking for a smartphone for everyday personal use (messaging, photos, navigation).",
        "I need a smartphone for field sales work (calls, email, mobile hotspot).",
    ],
}

SPLITS = {
    "test_unseen_item":  {"n": 200, "types": {"A": 80, "C": 60, "B": 60},
                          "seed": 20260803, "prefix": "S4NI", "domain": "laptop"},
    "test_domain_phone": {"n": 150, "types": {"A": 60, "C": 45, "B": 45},
                          "seed": 20260804, "prefix": "S4PH", "domain": "phone"},
}

# ---- phone-domain synthetic builders (profiles differ from the laptop ones
# in generate_candidates.py, which stays frozen for Study 3 reproducibility) --
PHONE_STORAGE_STEPS = [4, 8, 16, 32, 64, 128, 256, 512]
PHONE_CPUS = ["Dual-core", "Quad-core", "Hexa-core", "Octa-core"]
PHONE_RAMS = [1, 2, 3, 4, 6, 8]


def phone_rand_attr(rng, spec):
    lo, hi = spec["valid_range"]
    if spec["unit"] == "gb":
        return rng.choice([s for s in PHONE_STORAGE_STEPS if lo <= s <= hi])
    v = rng.uniform(lo, hi)
    return round(v, 3 if spec["unit"] == "kg" else 0)


def phone_make_item(rng, registry, item_id):
    item = {"item_id": item_id}
    for attr, spec in registry.items():
        item[attr] = phone_rand_attr(rng, spec)
    item["cpu"] = rng.choice(PHONE_CPUS)
    item["ram"] = rng.choice(PHONE_RAMS)
    return item


def phone_dominant_pair_values(rng, spec):
    lo, hi = spec["valid_range"]
    gap = spec["min_pair_gap"] * rng.uniform(1.5, 3.0)
    if spec["unit"] == "gb":
        steps = [s for s in PHONE_STORAGE_STEPS if lo <= s <= hi]
        j_idx = rng.randrange(len(steps) - 1)
        i_idx = rng.randrange(j_idx + 1, len(steps))
        hi_v, lo_v = steps[i_idx], steps[j_idx]
    else:
        mid = rng.uniform(lo + gap, hi - gap)
        hi_v, lo_v = round(mid + gap / 2, 3), round(mid - gap / 2, 3)
    if spec["preference_direction"] == "higher":
        return hi_v, lo_v
    return lo_v, hi_v


def phone_build_set_A(rng, registry, attr, set_id):
    base = phone_make_item(rng, registry, "TMP")
    spec = registry[attr]
    vi, vj = phone_dominant_pair_values(rng, spec)
    items = []
    for n, v in enumerate([vi, vj]):
        it = dict(base)
        it["item_id"] = f"{set_id}-I{n}"
        it[attr] = v
        items.append(it)
    for n in range(2, 5):
        it = dict(base)
        it["item_id"] = f"{set_id}-I{n}"
        it[attr] = phone_rand_attr(rng, spec)
        items.append(it)
    return items, (f"{set_id}-I0", f"{set_id}-I1")


def phone_build_set_C(rng, registry, attr, set_id):
    items, (fi, fj) = phone_build_set_A(rng, registry, attr, set_id)
    sec = SECONDARY[attr]
    spec = registry[sec]
    lo, hi = spec["valid_range"]
    span = hi - lo
    a, b = items[0][sec], items[1][sec]
    mid = (a + b) / 2 if isinstance(a, (int, float)) else a
    delta = 0.1 * span
    worse = mid + delta if spec["preference_direction"] == "lower" else mid - delta
    better = mid - delta / 2 if spec["preference_direction"] == "lower" else mid + delta / 2
    items[0][sec] = round(max(lo, min(hi, worse)), 3)
    items[1][sec] = round(max(lo, min(hi, better)), 3)
    return items, (fi, fj)


def study3_b_item_triples():
    """(price, weight, storage) triples of every B-type item used in Study 3."""
    triples = set()
    d3 = ROOT / "data_processed" / "study3"
    for fname in ("candidate_sets_test_main.jsonl", "candidate_sets_test_unseen_attr.jsonl"):
        for line in open(d3 / fname):
            cs = json.loads(line)
            if cs["set_type"] != "B":
                continue
            for it in cs["items"]:
                triples.add((it.get("price"), it.get("weight"), it.get("storage")))
    return triples


def gen_split(name, spec, out_dir):
    domain = spec["domain"]
    if domain == "laptop":
        registry = load_registry()
        catalog = pd.read_csv(ROOT / "data_processed" / "real_laptops.csv").dropna(axis=1, how="all")
        used = study3_b_item_triples()
        mask = catalog.apply(lambda r: (r["price"], r["weight"], r["storage"]) in used, axis=1)
        n_excluded = int(mask.sum())
        catalog = catalog[~mask].reset_index(drop=True)
        build_a, build_c = build_set_A, build_set_C
    else:
        registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry_phones.yaml"))
        catalog = pd.read_csv(ROOT / "data_processed" / "real_phones.csv")
        n_excluded = 0
        build_a, build_c = phone_build_set_A, phone_build_set_C

    rng = random.Random(spec["seed"])
    plan = []
    for stype, count in spec["types"].items():
        for n in range(count):
            if stype == "B":
                battrs = [a for a in HELD_IN_ATTRS if a in catalog.columns]
                plan.append((stype, battrs[n % len(battrs)]))
            else:
                plan.append((stype, HELD_IN_ATTRS[n % len(HELD_IN_ATTRS)]))
    rng.shuffle(plan)
    ctxs = BASE_CONTEXTS[name]
    n_b_fallback = 0
    sets_path = out_dir / f"candidate_sets_{name}.jsonl"
    with open(sets_path, "w") as f:
        for k, (stype, attr) in enumerate(plan):
            set_id = f"{spec['prefix']}{k:05d}"
            if stype == "B":
                built = build_set_B(rng, registry, attr, set_id, catalog)
                if built is None:
                    built, stype = build_a(rng, registry, attr, set_id), "A_fallback_from_B"
                    n_b_fallback += 1
            elif stype == "C":
                built = build_c(rng, registry, attr, set_id)
            else:
                built = build_a(rng, registry, attr, set_id)
            items, focal = built
            f.write(json.dumps({
                "set_id": set_id, "split": name, "set_type": stype, "domain": domain,
                "target_attribute": attr, "focal_pair": list(focal),
                "base_context": ctxs[k % len(ctxs)], "items": items,
            }) + "\n")
    return sets_path, n_b_fallback, n_excluded


def gen_interventions(name, sets_path, families, out_dir):
    """T1/T2 renderings (same mechanics as Study 3 gen_interventions)."""
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
    return n_reject


def gen_t4_specs(out_dir):
    """T4 (exploratory) specs over Study 3 test_main sets; calibration-gated."""
    calib = json.loads((ROOT / "results" / "study4" / "t4_calibration.json").read_text())
    if not calib["calibration_passed"]:
        return None
    n_reject = 0
    out = out_dir / "intervention_specs_test_t4_word.jsonl"
    with open(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl") as fin, \
            open(out, "w") as fout:
        for line in fin:
            cs = json.loads(line)
            attr = cs["target_attribute"]
            for level in range(1, 5):
                text = render_t4(attr, level)
                spec = {
                    "spec_id": f"{cs['set_id']}-L{level}-T4",
                    "set_id": cs["set_id"], "split": "test_t4_word",
                    "target_attribute": attr, "polarity": "positive",
                    "strength_level": level, "template_family": "T4",
                    "base_context": cs["base_context"], "surface_form": text,
                    "exploratory": True,
                }
                ok, reason = validate_t4_surface(spec, text)
                spec["validation"] = reason
                if not ok:
                    spec["rejected"] = True
                    n_reject += 1
                fout.write(json.dumps(spec) + "\n")
    return n_reject


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in SPLITS.items():
        sets_path, n_fb, n_excl = gen_split(name, spec, OUT)
        n_rej = gen_interventions(name, sets_path, ["T1", "T2"], OUT)
        summary[name] = {"sets": spec["n"], "b_fallback": n_fb,
                         "catalog_rows_excluded_as_seen": n_excl,
                         "rejected_surfaces": n_rej}
    t4_rej = gen_t4_specs(OUT)
    summary["test_t4_word"] = ({"sets": "reuses study3 test_main", "rejected_surfaces": t4_rej}
                               if t4_rej is not None else {"skipped": "T4 calibration failed"})
    print(json.dumps(summary, indent=2))
    total_rej = sum(v.get("rejected_surfaces") or 0 for v in summary.values())
    if total_rej:
        sys.exit(f"FATAL: {total_rej} surface forms failed round-trip validation")


if __name__ == "__main__":
    main()
