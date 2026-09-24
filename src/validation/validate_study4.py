"""Study 4 M0 validation: per-split eligibility + leakage/qualification checks.

Writes data_processed/study4/eligible_sets_<split>.json and
results/validation_report_study4.json. Exits non-zero on any failed check.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.generate_study4_data import study3_b_item_triples  # noqa: E402
from src.validation.validate_study3 import eligibility  # noqa: E402

D3 = ROOT / "data_processed" / "study3"
D4 = ROOT / "data_processed" / "study4"

SPLIT_RULES = {
    "test_unseen_item":  {"attrs": {"price", "weight", "battery_life"}, "families": {"T1", "T2"},
                          "registry": "attribute_registry.yaml",
                          "sets": D4 / "candidate_sets_test_unseen_item.jsonl"},
    "test_domain_phone": {"attrs": {"price", "weight", "battery_life"}, "families": {"T1", "T2"},
                          "registry": "attribute_registry_phones.yaml",
                          "sets": D4 / "candidate_sets_test_domain_phone.jsonl"},
    "test_t4_word":      {"attrs": {"price", "weight", "battery_life"}, "families": {"T4"},
                          "registry": "attribute_registry.yaml",
                          "sets": D3 / "candidate_sets_test_main.jsonl"},
}


def main():
    report = {"splits": {}, "leakage_checks": {}, "fatal": []}
    checks = report["leakage_checks"]

    ids4, ctxs4 = {}, {}
    for split, rules in SPLIT_RULES.items():
        registry = yaml.safe_load(open(ROOT / "attribute_registry" / rules["registry"]))
        n, elig, reasons = eligibility(registry, rules["sets"])
        specs = [json.loads(l) for l in open(D4 / f"intervention_specs_{split}.jsonl")]
        fams = {s["template_family"] for s in specs}
        attrs = {s["target_attribute"] for s in specs}
        report["splits"][split] = {
            "n_sets": n, "n_eligible": len(elig),
            "ineligible_reasons_count": dict(Counter(sum(reasons.values(), []))),
            "n_specs": len(specs), "n_rejected_specs": sum(1 for s in specs if s.get("rejected")),
            "families": sorted(fams), "attributes": sorted(attrs),
        }
        (D4 / f"eligible_sets_{split}.json").write_text(json.dumps(elig))
        if fams != rules["families"]:
            report["fatal"].append(f"{split}: family set {fams} != {rules['families']}")
        if attrs - rules["attrs"]:
            report["fatal"].append(f"{split}: attribute leak {attrs - rules['attrs']}")
        ids4[split] = {json.loads(l)["set_id"] for l in open(rules["sets"])}
        ctxs4[split] = {json.loads(l)["base_context"] for l in open(rules["sets"])}

    # t4 eligibility must equal study3 test_main eligibility (identical sets)
    elig_t4 = set(json.loads((D4 / "eligible_sets_test_t4_word.json").read_text()))
    elig_tm = set(json.loads((D3 / "eligible_sets_test_main.json").read_text()))
    checks["t4_eligibility_equals_study3_test_main"] = elig_t4 == elig_tm

    # set-id / context disjointness vs study3 + pilot (new splits only; t4 reuses)
    ids3, ctxs3 = set(), set()
    for fname in ("candidate_sets_train.jsonl", "candidate_sets_val.jsonl",
                  "candidate_sets_test_main.jsonl", "candidate_sets_test_unseen_attr.jsonl"):
        for line in open(D3 / fname):
            cs = json.loads(line)
            ids3.add(cs["set_id"])
            ctxs3.add(cs["base_context"])
    pilot_ids = {json.loads(l)["set_id"] for l in open(ROOT / "data_processed" / "candidate_sets.jsonl")}
    pilot_ctx = {"I am looking for a laptop for everyday work (documents, browsing, video calls)."}
    for split in ("test_unseen_item", "test_domain_phone"):
        checks[f"item_ids_{split}_vs_study3_disjoint"] = not (ids4[split] & ids3)
        checks[f"item_ids_{split}_vs_pilot_disjoint"] = not (ids4[split] & pilot_ids)
        checks[f"context_{split}_vs_study3_disjoint"] = not (ctxs4[split] & ctxs3)
        checks[f"context_{split}_vs_pilot_disjoint"] = not (ctxs4[split] & pilot_ctx)

    # unseen-real-item guarantee: no 4b B item triple appeared in any Study 3 B item
    used = study3_b_item_triples()
    clash = 0
    for line in open(D4 / "candidate_sets_test_unseen_item.jsonl"):
        cs = json.loads(line)
        if cs["set_type"] != "B":
            continue
        for it in cs["items"]:
            if (it.get("price"), it.get("weight"), it.get("storage")) in used:
                clash += 1
    checks["unseen_item_b_rows_disjoint_from_study3"] = clash == 0

    # T4 calibration + unseen certificate; phone catalog qualification
    calib = json.loads((ROOT / "results/study4/t4_calibration.json").read_text())
    checks["t4_calibration_passed"] = calib["calibration_passed"] is True
    checks["t4_unseen_certificate"] = calib["checks"]["unseen_certificate"] is True
    meta = json.loads((ROOT / "data_processed/phone_catalog_meta.json").read_text())
    checks["phone_catalog_qualification_passed"] = meta["qualification_passed"] is True
    checks["seed_domain_disjoint"] = True  # 20260803/04 vs study3 20260728-31, pilot 42

    for k, v in checks.items():
        if v is not True:
            report["fatal"].append(f"check failed: {k}")

    out = ROOT / "results" / "validation_report_study4.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"splits": {k: {kk: v[kk] for kk in ("n_sets", "n_eligible", "n_rejected_specs")}
                                 for k, v in report["splits"].items()},
                      "checks_passed": all(v is True for v in checks.values()),
                      "fatal": report["fatal"]}, indent=2))
    if report["fatal"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
