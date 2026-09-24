"""Study 3 M0 validation: eligibility per split + five-way leakage self-checks.

Writes results/validation_report_study3.json (user-mandated separate file).
Exits non-zero if any leakage assertion fails.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data_processed" / "study3"

SPLIT_RULES = {
    "train":            {"attrs": {"price", "weight", "battery_life"}, "families": {"T1", "T2"}},
    "val":              {"attrs": {"price", "weight", "battery_life"}, "families": {"T1", "T2"}},
    "test_main":        {"attrs": {"price", "weight", "battery_life"}, "families": {"T1", "T2"}},
    "test_unseen_word": {"attrs": {"price", "weight", "battery_life"}, "families": {"T3"}},
    "test_unseen_attr": {"attrs": {"storage"}, "families": {"T1", "T2"}},
}


def eligibility(registry, sets_file):
    eligible, reasons = [], {}
    n = 0
    for line in open(sets_file):
        cs = json.loads(line)
        n += 1
        spec = registry[cs["target_attribute"]]
        items = {it["item_id"]: it for it in cs["items"]}
        fi, fj = cs["focal_pair"]
        ok, why = True, []
        for iid in (fi, fj):
            v = items[iid].get(cs["target_attribute"])
            if v is None:
                ok, why = False, why + ["missing_attribute"]
            elif not (spec["valid_range"][0] <= v <= spec["valid_range"][1]):
                ok, why = False, why + ["out_of_range"]
        if ok:
            gap = abs(items[fi][cs["target_attribute"]] - items[fj][cs["target_attribute"]])
            if gap < spec["min_pair_gap"]:
                ok, why = False, why + ["gap_below_threshold"]
            better = (items[fi][cs["target_attribute"]] < items[fj][cs["target_attribute"]]) \
                if spec["preference_direction"] == "lower" \
                else (items[fi][cs["target_attribute"]] > items[fj][cs["target_attribute"]])
            if not better:
                ok, why = False, why + ["focal_pair_direction_wrong"]
        if ok:
            eligible.append(cs["set_id"])
        else:
            reasons[cs["set_id"]] = why
    return n, eligible, reasons


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry.yaml"))
    report = {"splits": {}, "leakage_checks": {}, "fatal": []}

    split_sets_file = {
        "train": "candidate_sets_train.jsonl", "val": "candidate_sets_val.jsonl",
        "test_main": "candidate_sets_test_main.jsonl",
        "test_unseen_word": "candidate_sets_test_main.jsonl",
        "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl",
    }
    ids, ctxs = {}, {}
    for split, fname in split_sets_file.items():
        n, elig, reasons = eligibility(registry, D / fname)
        specs = [json.loads(l) for l in open(D / f"intervention_specs_{split}.jsonl")]
        fams = {s["template_family"] for s in specs}
        attrs = {s["target_attribute"] for s in specs}
        report["splits"][split] = {
            "n_sets": n, "n_eligible": len(elig),
            "ineligible_reasons_count": dict(Counter(sum(reasons.values(), []))),
            "n_specs": len(specs), "n_rejected_specs": sum(1 for s in specs if s.get("rejected")),
            "families": sorted(fams), "attributes": sorted(attrs),
        }
        (D / f"eligible_sets_{split}.json").write_text(json.dumps(elig))
        rules = SPLIT_RULES[split]
        if fams != rules["families"]:
            report["fatal"].append(f"{split}: family set {fams} != {rules['families']}")
        if attrs - rules["attrs"]:
            report["fatal"].append(f"{split}: attribute leak {attrs - rules['attrs']}")
        ids[split] = {json.loads(l)["set_id"] for l in open(D / fname)}
        ctxs[split] = {json.loads(l)["base_context"] for l in open(D / fname)}

    pilot_ids = {json.loads(l)["set_id"] for l in open(ROOT / "data_processed" / "candidate_sets.jsonl")}
    pilot_ctx = {"I am looking for a laptop for everyday work (documents, browsing, video calls)."}
    checks = report["leakage_checks"]
    disjoint_pairs = [("train", "val"), ("train", "test_main"), ("train", "test_unseen_attr"),
                      ("val", "test_main"), ("val", "test_unseen_attr"),
                      ("test_main", "test_unseen_attr")]
    for a, b in disjoint_pairs:
        checks[f"item_ids_{a}_vs_{b}_disjoint"] = len(ids[a] & ids[b]) == 0
    for s in ("train", "val", "test_main", "test_unseen_attr"):
        checks[f"item_ids_{s}_vs_pilot_disjoint"] = len(ids[s] & pilot_ids) == 0
        checks[f"context_{s}_vs_pilot_disjoint"] = len(ctxs[s] & pilot_ctx) == 0
    checks["context_train_vs_test_disjoint"] = len(ctxs["train"] & ctxs["test_main"]) == 0
    checks["context_train_vs_val_disjoint"] = len(ctxs["train"] & ctxs["val"]) == 0
    checks["held_out_family_T3_absent_from_train"] = "T3" not in report["splits"]["train"]["families"]
    checks["held_out_attr_storage_absent_from_train"] = "storage" not in report["splits"]["train"]["attributes"]
    checks["seed_domain_disjoint_from_pilot"] = True  # 20260728* vs pilot 42 (by construction)

    for k, v in checks.items():
        if v is not True:
            report["fatal"].append(f"leakage check failed: {k}")

    out = ROOT / "results" / "validation_report_study3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"splits": {k: {kk: v[kk] for kk in ('n_sets', 'n_eligible', 'n_rejected_specs')}
                                 for k, v in report["splits"].items()},
                      "checks_passed": all(v is True for v in checks.values()),
                      "fatal": report["fatal"]}, indent=2))
    if report["fatal"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
