"""Generate the synthetic laptop catalog and candidate sets (RESEARCH_BRIEF §7.4).

Set types:
  A: fully-controlled synthetic pairs — only the target attribute differs.
  B: matched real item pairs — real catalog rows, minimal non-target distance.
  C: trade-off sets — target-dominant item slightly worse on one secondary attribute.

Every set has exactly `candidates_per_set` items and one designated eligible
focal pair (i dominates j on the target attribute); remaining items are
distractors whose target values sit outside the focal gap.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

SECONDARY = {  # secondary attribute used for type-C trade-offs
    "price": "weight",
    "weight": "battery_life",
    "battery_life": "storage",
    "storage": "price",
}

STORAGE_STEPS = [128, 256, 512, 1024, 2048, 4096]


def load_registry():
    with open(ROOT / "attribute_registry" / "attribute_registry.yaml") as f:
        return yaml.safe_load(f)


def rand_attr(rng, spec):
    lo, hi = spec["valid_range"]
    if spec["unit"] == "gb":
        return rng.choice([s for s in STORAGE_STEPS if lo <= s <= hi])
    v = rng.uniform(lo, hi)
    return round(v, 2 if spec["unit"] == "kg" else 0)


def make_item(rng, registry, item_id, overrides=None):
    item = {"item_id": item_id}
    for attr, spec in registry.items():
        item[attr] = rand_attr(rng, spec)
    item["cpu"] = rng.choice(["i5-1340P", "i7-1360P", "Ryzen 5 7640U", "Ryzen 7 7840U"])
    item["ram"] = rng.choice([8, 16, 32])
    if overrides:
        item.update(overrides)
    return item


def dominant_pair_values(rng, spec):
    """Return (better, worse) raw values separated by >= 1.5x min_pair_gap."""
    lo, hi = spec["valid_range"]
    gap = spec["min_pair_gap"] * rng.uniform(1.5, 3.0)
    if spec["unit"] == "gb":
        steps = [s for s in STORAGE_STEPS if lo <= s <= hi]
        j_idx = rng.randrange(len(steps) - 1)
        i_idx = rng.randrange(j_idx + 1, len(steps))
        hi_v, lo_v = steps[i_idx], steps[j_idx]
    else:
        mid = rng.uniform(lo + gap, hi - gap)
        hi_v, lo_v = round(mid + gap / 2, 2), round(mid - gap / 2, 2)
    if spec["preference_direction"] == "higher":
        return hi_v, lo_v
    return lo_v, hi_v


def build_set_A(rng, registry, attr, set_id):
    """Only the target attribute varies between focal items; distractors clones."""
    base = make_item(rng, registry, "TMP")
    spec = registry[attr]
    vi, vj = dominant_pair_values(rng, spec)
    items = []
    for n, v in enumerate([vi, vj]):
        it = dict(base)
        it["item_id"] = f"{set_id}-I{n}"
        it[attr] = v
        items.append(it)
    lo, hi = spec["valid_range"]
    for n in range(2, 5):
        it = dict(base)
        it["item_id"] = f"{set_id}-I{n}"
        it[attr] = rand_attr(rng, spec)
        items.append(it)
    return items, (f"{set_id}-I0", f"{set_id}-I1")


def build_set_B(rng, registry, attr, set_id, catalog):
    """Matched real pairs: min non-target distance, target gap >= min_pair_gap."""
    if attr not in catalog.columns:
        return None
    spec = registry[attr]
    others = [a for a in registry if a != attr and a in catalog.columns]
    df = catalog.dropna(subset=[attr] + others)
    if len(df) < 20:
        return None
    rows = df.sample(min(400, len(df)), random_state=rng.randrange(1 << 30))
    norm = {a: (rows[a] - rows[a].min()) / max(rows[a].max() - rows[a].min(), 1e-9) for a in others}
    best, best_d = None, None
    idx = list(rows.index)
    for _ in range(2000):
        i, j = rng.sample(idx, 2)
        if abs(rows.at[i, attr] - rows.at[j, attr]) < spec["min_pair_gap"]:
            continue
        d = sum(abs(norm[a][i] - norm[a][j]) for a in others)
        if best_d is None or d < best_d:
            best, best_d = (i, j), d
    if best is None:
        return None
    i, j = best
    if spec["preference_direction"] == "lower":
        if rows.at[i, attr] > rows.at[j, attr]:
            i, j = j, i
    elif rows.at[i, attr] < rows.at[j, attr]:
        i, j = j, i
    picked = [i, j] + rng.sample([k for k in idx if k not in (i, j)], 3)
    items = []
    for n, k in enumerate(picked):
        it = {"item_id": f"{set_id}-I{n}"}
        for a in registry:
            it[a] = float(rows.at[k, a]) if a in rows.columns and pd.notna(rows.at[k, a]) else None
        it["cpu"] = str(rows.at[k, "cpu"]) if "cpu" in rows.columns else "n/a"
        it["ram"] = int(rows.at[k, "ram"]) if "ram" in rows.columns else 16
        items.append(it)
    return items, (f"{set_id}-I0", f"{set_id}-I1")


def build_set_C(rng, registry, attr, set_id):
    """Target-dominant item is slightly worse on one secondary attribute."""
    items, (fi, fj) = build_set_A(rng, registry, attr, set_id)
    sec = SECONDARY[attr]
    spec = registry[sec]
    lo, hi = spec["valid_range"]
    span = hi - lo
    a, b = items[0][sec], items[1][sec]
    mid = (a + b) / 2 if isinstance(a, (int, float)) else a
    delta = 0.1 * span
    worse = mid + delta if spec["preference_direction"] == "lower" else mid - delta
    better = mid - delta / 2 if spec["preference_direction"] == "lower" else mid + delta / 2
    items[0][sec] = round(max(lo, min(hi, worse)), 2)
    items[1][sec] = round(max(lo, min(hi, better)), 2)
    return items, (fi, fj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--real-catalog", default=str(ROOT / "data_processed" / "real_laptops.csv"))
    ap.add_argument("--out", default=str(ROOT / "data_processed" / "candidate_sets.jsonl"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    registry = load_registry()
    rng = random.Random(cfg["seed"])
    attrs = cfg["attributes"]

    catalog = None
    real_path = Path(args.real_catalog)
    if real_path.exists():
        catalog = pd.read_csv(real_path).dropna(axis=1, how="all")

    plan = []
    for stype, count in cfg["candidate_set_types"].items():
        for n in range(count):
            plan.append((stype, attrs[n % len(attrs)]))
    rng.shuffle(plan)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_b_fallback = 0
    with open(out_path, "w") as f:
        for k, (stype, attr) in enumerate(plan):
            set_id = f"CS{k:04d}"
            built = None
            if stype == "B":
                if catalog is not None:
                    built = build_set_B(rng, registry, attr, set_id, catalog)
                if built is None:
                    built, stype, n_b_fallback = build_set_A(rng, registry, attr, set_id), "A_fallback_from_B", n_b_fallback + 1
            elif stype == "C":
                built = build_set_C(rng, registry, attr, set_id)
            else:
                built = build_set_A(rng, registry, attr, set_id)
            items, focal = built
            rec = {
                "set_id": set_id,
                "set_type": stype,
                "target_attribute": attr,
                "focal_pair": list(focal),
                "items": items,
            }
            f.write(json.dumps(rec) + "\n")

    manifest = {
        "data_version": cfg["data_version"],
        "seed": cfg["seed"],
        "n_sets": len(plan),
        "b_fallback_to_synthetic": n_b_fallback,
        "real_catalog_used": catalog is not None,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    with open(ROOT / "data_processed" / "catalog_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
