"""Study 6 M0 validation: per-split eligibility + derivation/leakage checks.

Writes data_processed/study6/eligible_sets_<split>.json and
results/validation_report_study6.json. Exits non-zero on any failed check.

Eligibility policy (frozen plan §5):
  - test_conflict: standard Study 3 eligibility (gap >= min_pair_gap etc.).
  - test_neartied: CUSTOM gate — range + focal direction + exact tier gap
    (|gap - mult x min_pair_gap| < 1e-6). The standard gap-below-threshold
    rejection is waived BY DESIGN for the 1x/0.5x tiers; recorded here.
  - derived splits (order/distract/mix): eligible = the frozen 100-set
    subsample (already a subset of study3 test_main eligible).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.validation.validate_study3 import eligibility  # noqa: E402
from src.candidate_generation.generate_study6_data import (  # noqa: E402
    MIX_SCHEMES, NEARTIED_TIERS)

D3 = ROOT / "data_processed" / "study3"
D4 = ROOT / "data_processed" / "study4"
D6 = ROOT / "data_processed" / "study6"

DERIVED = ([f"test_order_p{k}" for k in range(1, 6)]
           + [f"test_distract_{t}" for t in ("d1", "d2", "d4")]
           + [f"test_mix{s[1:]}" for s in MIX_SCHEMES])


def rows(path):
    return [json.loads(l) for l in open(path)]


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    report = {"splits": {}, "checks": {}, "fatal": []}
    checks = report["checks"]

    sub = json.loads((D6 / "subsample_test_main.json").read_text())
    elig_tm = set(json.loads((D3 / "eligible_sets_test_main.json").read_text()))
    checks["subsample_subset_of_test_main_eligible"] = set(sub) <= elig_tm
    checks["subsample_size_100"] = len(sub) == 100

    tm_sets = {cs["set_id"]: cs for cs in rows(D3 / "candidate_sets_test_main.jsonl")
               if cs["set_id"] in set(sub)}

    # --- derived splits: eligible = subsample; structural checks ------------
    for split in DERIVED:
        specs = rows(D6 / f"intervention_specs_{split}.jsonl")
        report["splits"][split] = {
            "n_specs": len(specs),
            "n_rejected_specs": sum(1 for s in specs if s.get("rejected")),
            "families": sorted({s["template_family"] for s in specs}),
        }
        (D6 / f"eligible_sets_{split}.json").write_text(json.dumps(sub))
        if any(s.get("rejected") for s in specs):
            report["fatal"].append(f"{split}: rejected specs present")
        if {s["set_id"] for s in specs} != set(sub):
            report["fatal"].append(f"{split}: spec set ids != subsample")

    # order perms: same items (as multisets), same focal/target, order differs
    perm_ok, order_changed = True, 0
    for k in range(1, 6):
        for cs in rows(D6 / f"candidate_sets_test_order_p{k}.jsonl"):
            orig = tm_sets[cs["set_id"]]
            key = lambda it: tuple(sorted(it.items()))
            if (sorted(map(key, cs["items"])) != sorted(map(key, orig["items"]))
                    or cs["focal_pair"] != orig["focal_pair"]
                    or cs["target_attribute"] != orig["target_attribute"]):
                perm_ok = False
            if [it["item_id"] for it in cs["items"]] != [it["item_id"] for it in orig["items"]]:
                order_changed += 1
    checks["order_perms_item_preserving"] = perm_ok
    checks["order_perms_actually_permuted"] = order_changed > 400  # of 500

    # distract: original context is a strict prefix; tier sentence counts
    ok = True
    for t, n_sent in (("d1", 1), ("d2", 2), ("d4", 4)):
        for sp in rows(D6 / f"intervention_specs_test_distract_{t}.jsonl"):
            orig_ctx = tm_sets[sp["set_id"]]["base_context"]
            if not sp["base_context"].startswith(orig_ctx + " ") \
                    or sp.get("distract_sentences") != n_sent:
                ok = False
    checks["distract_context_prefix_and_tiers"] = ok

    # mix: every chain uses >= 2 source families; scheme family sets exact
    ok = True
    for scheme, fams in MIX_SCHEMES.items():
        split = f"test_mix{scheme[1:]}"
        chains = {}
        for sp in rows(D6 / f"intervention_specs_{split}.jsonl"):
            chains.setdefault(sp["set_id"], set()).add(sp["source_family"])
        if not all(len(v) >= 2 and v <= set(fams) for v in chains.values()):
            ok = False
    checks["mix_chains_multi_family"] = ok

    # --- test_conflict: standard eligibility --------------------------------
    n, elig, reasons = eligibility(registry, D6 / "candidate_sets_test_conflict.jsonl")
    (D6 / "eligible_sets_test_conflict.json").write_text(json.dumps(elig))
    report["splits"]["test_conflict"] = {
        "n_sets": n, "n_eligible": len(elig),
        "ineligible_reasons": dict(Counter(sum(reasons.values(), [])))}
    # conflict structure: focal loser wins every non-target attribute
    ok = True
    for cs in rows(D6 / "candidate_sets_test_conflict.jsonl"):
        items = {it["item_id"]: it for it in cs["items"]}
        fi, fj = cs["focal_pair"]
        for a, spec in registry.items():
            if a == cs["target_attribute"]:
                continue
            gap = items[fj][a] - items[fi][a]
            if spec["preference_direction"] == "lower":
                gap = -gap
            if gap < spec["min_pair_gap"] - 1e-6:
                ok = False
    checks["conflict_nontarget_domination"] = ok

    # --- test_neartied: custom gate -----------------------------------------
    elig_nt, tier_counts, bad = [], Counter(), []
    for cs in rows(D6 / "candidate_sets_test_neartied.jsonl"):
        spec = registry[cs["target_attribute"]]
        items = {it["item_id"]: it for it in cs["items"]}
        fi, fj = cs["focal_pair"]
        vi, vj = items[fi][cs["target_attribute"]], items[fj][cs["target_attribute"]]
        lo, hi = spec["valid_range"]
        in_range = lo <= vi <= hi and lo <= vj <= hi
        direction = vi < vj if spec["preference_direction"] == "lower" else vi > vj
        want = spec["min_pair_gap"] * cs["gap_multiplier"]
        exact = abs(abs(vi - vj) - want) < 1e-6
        if in_range and direction and exact:
            elig_nt.append(cs["set_id"])
            tier_counts[cs["gap_tier"]] += 1
        else:
            bad.append(cs["set_id"])
    (D6 / "eligible_sets_test_neartied.json").write_text(json.dumps(elig_nt))
    report["splits"]["test_neartied"] = {
        "n_sets": sum(c for _, _, c in NEARTIED_TIERS), "n_eligible": len(elig_nt),
        "tier_counts": dict(tier_counts), "failed_custom_gate": bad,
        "policy_note": "standard gap>=min_pair_gap check waived by design for "
                       "1x/0.5x tiers (frozen plan section 5); custom gate = "
                       "range + direction + exact tier gap"}
    checks["neartied_custom_gate_all_pass"] = not bad
    checks["neartied_tier_counts"] = dict(tier_counts) == {
        t: c for t, _, c in NEARTIED_TIERS}

    # spec counts for the two new-set splits
    for split in ("test_neartied", "test_conflict"):
        specs = rows(D6 / f"intervention_specs_{split}.jsonl")
        report["splits"][split].update({
            "n_specs": len(specs),
            "n_rejected_specs": sum(1 for s in specs if s.get("rejected"))})
        if any(s.get("rejected") for s in specs):
            report["fatal"].append(f"{split}: rejected specs present")

    # --- leakage: new ids/contexts disjoint from pilot/study3/study4 --------
    prev_ids, prev_ctx = set(), set()
    for f in list(D3.glob("candidate_sets_*.jsonl")) + list(D4.glob("candidate_sets_*.jsonl")):
        for cs in rows(f):
            prev_ids.add(cs["set_id"])
            prev_ctx.add(cs["base_context"])
    for cs in rows(ROOT / "data_processed/candidate_sets.jsonl"):
        prev_ids.add(cs["set_id"])
    new_ids, new_ctx = set(), set()
    for split in ("test_neartied", "test_conflict"):
        for cs in rows(D6 / f"candidate_sets_{split}.jsonl"):
            new_ids.add(cs["set_id"])
            new_ctx.add(cs["base_context"])
    checks["new_set_ids_disjoint"] = not (new_ids & prev_ids)
    checks["new_contexts_disjoint"] = not (new_ctx & prev_ctx)

    for k, v in checks.items():
        if v is not True:
            report["fatal"].append(f"check failed: {k}={v}")

    out = ROOT / "results/validation_report_study6.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks, "fatal": report["fatal"],
                      "splits": {k: {kk: vv for kk, vv in v.items()
                                     if kk in ("n_sets", "n_eligible", "n_specs",
                                               "n_rejected_specs", "tier_counts")}
                                 for k, v in report["splits"].items()}}, indent=2))
    if report["fatal"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
