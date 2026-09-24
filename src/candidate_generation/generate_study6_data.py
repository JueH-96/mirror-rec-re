"""Study 6 data generation (M0) — frozen plan experiments/study6_plan.md.

Robustness surfaces (§5 of the plan):
  test_order_p1..p5    common 100-set subsample of study3 test_main eligible
                       sets with the candidate list permuted (set ids kept so
                       chains pair across permutations; p0 = original order
                       reuses study3 raw results).
  test_distract_d1/2/4 same subsample; T1/T2 specs with 1/2/4 attribute-
                       neutral filler sentences appended to base_context.
  test_mix12/123/1234  same subsample; one chain per set whose per-level
                       surface rotates across wording families
                       (T1/T2, T1/T2/T3, T1/T2/T3/T4).
  test_neartied        350 new A-type sets, focal gap = min_pair_gap x
                       {2x:100, 1x:100, 0.5x:150}, continuous attrs only.
  test_conflict        100 new sets: focal item wins the target attribute by
                       >= min_pair_gap but loses every other registry
                       attribute by >= that attribute's min_pair_gap.

Seeds 20260805-20260807 — disjoint from pilot (42), study3 (20260728-31),
study4 (20260803-04).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.generate_candidates import (  # noqa: E402
    STORAGE_STEPS, dominant_pair_values, load_registry, make_item)
from src.intervention_generation.generate_interventions import (  # noqa: E402
    ATTR_CUES, ATTR_NOUN, ATTR_PHRASES, NEGATION, STRENGTH_CUES, TEMPLATES,
    validate_surface)
from src.intervention_generation.t4_family import (  # noqa: E402
    T4_STRENGTH_CUES, render_t4, validate_t4_surface)

import re  # noqa: E402

D3 = ROOT / "data_processed" / "study3"
OUT = ROOT / "data_processed" / "study6"

CONTINUOUS_ATTRS = ["price", "weight", "battery_life"]
ATTR_PRECISION = {"price": 0, "weight": 2, "battery_life": 1}
NEARTIED_TIERS = [("2x", 2.0, 100), ("1x", 1.0, 100), ("05x", 0.5, 150)]

SUBSAMPLE_N = 100
SUBSAMPLE_SEED = 20260807
PERM_SEEDS = {k: 20260807 + k for k in range(1, 6)}

BASE_CONTEXTS = {
    "test_neartied": [
        "I am choosing a laptop for a small architecture studio (CAD previews, drawings, site photos).",
        "I need a laptop for volunteer tutoring sessions (worksheets, video lessons, quizzes).",
    ],
    "test_conflict": [
        "I am selecting a laptop for hospital ward documentation (records, scheduling, telehealth).",
        "I want a laptop for restaurant inventory management (orders, spreadsheets, supplier calls).",
    ],
}

# Attribute-neutral filler sentences: mechanically checked below against every
# attribute cue, every T1-T4 strength cue, and the negation lexicon.
DISTRACTORS = [
    "My cousin visited over the weekend and we cooked dinner together.",
    "The park near my apartment has been busy with runners lately.",
    "I listened to a history podcast on the train this morning.",
    "Our neighborhood cafe repainted its walls a cheerful yellow.",
]
DISTRACT_TIERS = {"d1": 1, "d2": 2, "d4": 4}

MIX_SCHEMES = {"M12": ["T1", "T2"],
               "M123": ["T1", "T2", "T3"],
               "M1234": ["T1", "T2", "T3", "T4"]}


def check_distractors():
    """Filler sentences must be free of attribute, strength, and negation cues."""
    cues = ([c for cs in ATTR_CUES.values() for c in cs]
            + [c for cs in STRENGTH_CUES.values() for c in cs]
            + [c for cs in T4_STRENGTH_CUES.values() for c in cs])
    for s in DISTRACTORS:
        for c in cues:
            assert not re.search(c, s, re.I), f"distractor hits cue {c!r}: {s}"
        assert not NEGATION.search(s), f"distractor hits negation: {s}"


def load_subsample():
    eligible = sorted(json.loads((D3 / "eligible_sets_test_main.json").read_text()))
    ids = set(random.Random(SUBSAMPLE_SEED).sample(eligible, SUBSAMPLE_N))
    sets, specs = [], []
    for line in open(D3 / "candidate_sets_test_main.jsonl"):
        cs = json.loads(line)
        if cs["set_id"] in ids:
            sets.append(cs)
    for line in open(D3 / "intervention_specs_test_main.jsonl"):
        sp = json.loads(line)
        if sp["set_id"] in ids and not sp.get("rejected"):
            specs.append(sp)
    sets.sort(key=lambda c: c["set_id"])
    return sets, specs


def render_family(attr, level, fam):
    if fam == "T4":
        return render_t4(attr, level)
    tmpl = TEMPLATES[level][fam]
    return tmpl.format(
        attr=ATTR_PHRASES[attr],
        noun=ATTR_NOUN[attr].capitalize() if tmpl.startswith("{noun}") else ATTR_NOUN[attr])


def write_specs(split, rows):
    with open(OUT / f"intervention_specs_{split}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_sets(split, sets):
    with open(OUT / f"candidate_sets_{split}.jsonl", "w") as f:
        for cs in sets:
            f.write(json.dumps(cs) + "\n")


def gen_order(sets, specs, summary):
    for k, seed in PERM_SEEDS.items():
        rng = random.Random(seed)
        perm_sets = []
        for cs in sets:
            cs2 = dict(cs)
            items = list(cs["items"])
            rng.shuffle(items)
            cs2["items"] = items
            cs2["split"] = f"test_order_p{k}"
            perm_sets.append(cs2)
        write_sets(f"test_order_p{k}", perm_sets)
        write_specs(f"test_order_p{k}",
                    [{**sp, "split": f"test_order_p{k}"} for sp in specs])
    summary["test_order_p1..p5"] = {"sets": len(sets), "perms": 5,
                                    "chains_per_perm": len(specs) // 4}


def gen_distract(sets, specs, summary):
    idx = {cs["set_id"]: n for n, cs in enumerate(sets)}
    for tier, n_sent in DISTRACT_TIERS.items():
        rows = []
        for sp in specs:
            start = idx[sp["set_id"]] % len(DISTRACTORS)
            extra = " ".join(DISTRACTORS[(start + i) % len(DISTRACTORS)]
                             for i in range(n_sent))
            rows.append({**sp, "split": f"test_distract_{tier}",
                         "base_context": sp["base_context"] + " " + extra,
                         "distract_sentences": n_sent})
        write_specs(f"test_distract_{tier}", rows)
    summary["test_distract_d1/d2/d4"] = {"sets": len(sets),
                                         "sentences_per_tier": DISTRACT_TIERS}


def gen_mix(sets, summary):
    n_reject = 0
    for scheme, fams in MIX_SCHEMES.items():
        split = f"test_mix{scheme[1:]}"
        rows = []
        for k, cs in enumerate(sets):
            attr = cs["target_attribute"]
            for level in range(1, 5):
                fam = fams[(k + level) % len(fams)]
                text = render_family(attr, level, fam)
                spec = {
                    "spec_id": f"{cs['set_id']}-L{level}-{scheme}",
                    "set_id": cs["set_id"], "split": split,
                    "target_attribute": attr, "polarity": "positive",
                    "strength_level": level, "template_family": scheme,
                    "source_family": fam,
                    "base_context": cs["base_context"], "surface_form": text,
                }
                vs = dict(spec)  # validators read target/polarity/strength only
                ok, reason = (validate_t4_surface(vs, text) if fam == "T4"
                              else validate_surface(vs, text))
                spec["validation"] = reason
                if not ok:
                    spec["rejected"] = True
                    n_reject += 1
                rows.append(spec)
        write_specs(split, rows)
        fams_used = {r["source_family"] for r in rows}
        assert fams_used == set(fams), f"{split}: families {fams_used} != {fams}"
    summary["test_mix"] = {"schemes": {k: v for k, v in MIX_SCHEMES.items()},
                           "sets": len(sets), "rejected_surfaces": n_reject}
    return n_reject


def controlled_pair(rng, spec, gap, prec):
    """Focal values with an exact (at attr precision) raw gap inside range."""
    lo, hi = spec["valid_range"]
    mid = rng.uniform(lo + gap, hi - gap)
    lo_v = round(mid - gap / 2, prec) if prec else round(mid - gap / 2)
    hi_v = round(lo_v + gap, prec) if prec else round(lo_v + gap)
    if spec["preference_direction"] == "higher":
        return hi_v, lo_v
    return lo_v, hi_v


def gen_neartied(registry, summary):
    rng = random.Random(20260805)
    ctxs = BASE_CONTEXTS["test_neartied"]
    sets, k = [], 0
    for tier, mult, count in NEARTIED_TIERS:
        for n in range(count):
            attr = CONTINUOUS_ATTRS[n % len(CONTINUOUS_ATTRS)]
            spec = registry[attr]
            gap = spec["min_pair_gap"] * mult
            set_id = f"S6NT{k:05d}"
            base = make_item(rng, registry, "TMP")
            vi, vj = controlled_pair(rng, spec, gap, ATTR_PRECISION[attr])
            items = []
            for m, v in enumerate([vi, vj]):
                it = dict(base)
                it["item_id"] = f"{set_id}-I{m}"
                it[attr] = v
                items.append(it)
            for m in range(2, 5):
                it = dict(base)
                it["item_id"] = f"{set_id}-I{m}"
                it[attr] = round(rng.uniform(*spec["valid_range"]),
                                 ATTR_PRECISION[attr]) if ATTR_PRECISION[attr] \
                    else round(rng.uniform(*spec["valid_range"]))
                items.append(it)
            sets.append({"set_id": set_id, "split": "test_neartied",
                         "set_type": "A", "domain": "laptop",
                         "gap_tier": tier, "gap_multiplier": mult,
                         "target_attribute": attr,
                         "focal_pair": [f"{set_id}-I0", f"{set_id}-I1"],
                         "base_context": ctxs[k % len(ctxs)], "items": items})
            k += 1
    write_sets("test_neartied", sets)
    summary["test_neartied"] = {"sets": len(sets),
                                "tiers": {t: c for t, _, c in NEARTIED_TIERS}}
    return sets


def gen_conflict(registry, summary):
    rng = random.Random(20260806)
    ctxs = BASE_CONTEXTS["test_conflict"]
    sets = []
    for n in range(100):
        attr = CONTINUOUS_ATTRS[n % len(CONTINUOUS_ATTRS)]
        spec = registry[attr]
        set_id = f"S6CF{n:05d}"
        base = make_item(rng, registry, "TMP")
        vi, vj = dominant_pair_values(rng, spec)  # I0 wins target by >= 1.5x gap
        it_i, it_j = dict(base), dict(base)
        it_i["item_id"], it_j["item_id"] = f"{set_id}-I0", f"{set_id}-I1"
        it_i[attr], it_j[attr] = vi, vj
        # I1 wins every other registry attribute by >= its min_pair_gap
        for other, ospec in registry.items():
            if other == attr:
                continue
            if ospec["unit"] == "gb":
                steps = [s for s in STORAGE_STEPS
                         if ospec["valid_range"][0] <= s <= ospec["valid_range"][1]]
                pairs = [(steps[b], steps[a]) for a in range(len(steps))
                         for b in range(a + 1, len(steps))
                         if steps[b] - steps[a] >= ospec["min_pair_gap"]]
                it_j[other], it_i[other] = rng.choice(pairs)
            else:
                prec = ATTR_PRECISION[other]
                win, lose = controlled_pair(
                    rng, ospec, ospec["min_pair_gap"] * rng.uniform(1.5, 3.0), prec)
                it_j[other], it_i[other] = win, lose
        items = [it_i, it_j]
        for m in range(2, 5):
            items.append(make_item(rng, registry, f"{set_id}-I{m}"))
        sets.append({"set_id": set_id, "split": "test_conflict",
                     "set_type": "CF", "domain": "laptop",
                     "target_attribute": attr,
                     "focal_pair": [f"{set_id}-I0", f"{set_id}-I1"],
                     "base_context": ctxs[n % len(ctxs)], "items": items})
    write_sets("test_conflict", sets)
    summary["test_conflict"] = {"sets": len(sets)}
    return sets


def gen_new_set_interventions(name, sets, summary):
    n_reject = 0
    rows = []
    for cs in sets:
        attr = cs["target_attribute"]
        for level in range(1, 5):
            for fam in ("T1", "T2"):
                text = render_family(attr, level, fam)
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
                rows.append(spec)
    write_specs(name, rows)
    summary[name]["rejected_surfaces"] = n_reject
    return n_reject


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_distractors()
    registry = load_registry()
    summary = {}
    sets, specs = load_subsample()
    assert len(sets) == SUBSAMPLE_N and len(specs) == SUBSAMPLE_N * 8
    (OUT / "subsample_test_main.json").write_text(
        json.dumps([cs["set_id"] for cs in sets]))
    gen_order(sets, specs, summary)
    gen_distract(sets, specs, summary)
    n_rej = gen_mix(sets, summary)
    nt_sets = gen_neartied(registry, summary)
    cf_sets = gen_conflict(registry, summary)
    n_rej += gen_new_set_interventions("test_neartied", nt_sets, summary)
    n_rej += gen_new_set_interventions("test_conflict", cf_sets, summary)
    print(json.dumps(summary, indent=2))
    if n_rej:
        sys.exit(f"FATAL: {n_rej} surface forms failed round-trip validation")


if __name__ == "__main__":
    main()
