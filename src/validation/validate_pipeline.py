"""Automatic eligibility + quality gates (RESEARCH_BRIEF §5.4, §8.5).

Checks every candidate set against the registry and emits eligible_sets.json
(set ids whose focal pair passes conditions 1-6; conditions 7-8 are enforced
per-response in llm_client.parse_ranking). Also writes validation_report.json
with gate outcomes and skipped-model records.
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry" / "attribute_registry.yaml"))
    cfg = yaml.safe_load(open(ROOT / "configs" / "pilot.yaml"))
    report = {"gates": {}, "skipped_models": [], "counts": {}}

    eligible, reasons = [], {}
    n_sets = 0
    with open(ROOT / "data_processed" / "candidate_sets.jsonl") as f:
        for line in f:
            cs = json.loads(line)
            n_sets += 1
            attr = cs["target_attribute"]
            spec = registry[attr]
            items = {it["item_id"]: it for it in cs["items"]}
            fi, fj = cs["focal_pair"]
            ok, why = True, []
            for iid in (fi, fj):
                v = items[iid].get(attr)
                if v is None:
                    ok, why = False, why + ["missing_attribute"]
                elif not (spec["valid_range"][0] <= v <= spec["valid_range"][1]):
                    ok, why = False, why + ["out_of_range"]
            if ok:
                gap = abs(items[fi][attr] - items[fj][attr])
                if gap < spec["min_pair_gap"]:
                    ok, why = False, why + ["gap_below_threshold"]
                better = items[fi][attr] < items[fj][attr] if spec["preference_direction"] == "lower" \
                    else items[fi][attr] > items[fj][attr]
                if not better:
                    ok, why = False, why + ["focal_pair_direction_wrong"]
            if ok:
                eligible.append(cs["set_id"])
            else:
                reasons[cs["set_id"]] = why

    # Chain invariance gate (§8.5): specs of one chain must share identical set.
    specs_ok = True
    seen = {}
    with open(ROOT / "data_processed" / "intervention_specs.jsonl") as f:
        for line in f:
            sp = json.loads(line)
            key = sp["set_id"]
            fields = (sp["target_attribute"], sp["polarity"], sp["base_context"])
            if key in seen and seen[key] != fields:
                specs_ok = False
            seen[key] = fields

    report["gates"]["candidate_sets_total"] = n_sets
    report["gates"]["eligible_sets"] = len(eligible)
    report["gates"]["ineligible_reasons"] = reasons
    report["gates"]["chain_invariance_ok"] = specs_ok
    report["gates"]["rejected_surface_forms"] = 0  # generator aborts on any rejection

    import os
    for mkey, mcfg in cfg["models"].items():
        if mcfg.get("provider") == "deepseek" and not os.environ.get(mcfg["api_key_env"], ""):
            report["skipped_models"].append(
                {"model": mkey, "reason": f"env {mcfg['api_key_env']} not set"})

    (ROOT / "data_processed" / "eligible_sets.json").write_text(json.dumps(eligible))
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"eligible {len(eligible)}/{n_sets}; chain_invariance_ok={specs_ok}; "
          f"skipped_models={[m['model'] for m in report['skipped_models']]}")
    if not specs_ok:
        sys.exit("FATAL: chain invariance violated")


if __name__ == "__main__":
    main()
