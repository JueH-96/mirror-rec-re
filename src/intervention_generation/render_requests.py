"""Render (candidate set x intervention spec x prompt variant) into request records.

Candidate serialization is byte-identical across all levels of a chain (§5.2:
only the preference sentence changes). Rendered requests carry the prompt
version so §8.4 logging can reference it.
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

UNITS = {"price": "USD", "weight": "kg", "battery_life": "hours", "storage": "GB"}


def serialize_candidates(items, units=None):
    units = units or UNITS
    lines = []
    for it in items:
        parts = [f"id={it['item_id']}"]
        for a in ("price", "weight", "battery_life", "storage"):
            v = it.get(a)
            if v is not None:
                parts.append(f"{a}={v} {units[a]}")
        parts.append(f"cpu={it.get('cpu', 'n/a')}")
        parts.append(f"ram={it.get('ram', 'n/a')} GB")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--variant", required=True, help="prompt variant key from config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    prompt_path = ROOT / cfg["prompt_versions"][args.variant]
    template = prompt_path.read_text()

    sets = {}
    with open(ROOT / "data_processed" / "candidate_sets.jsonl") as f:
        for line in f:
            cs = json.loads(line)
            sets[cs["set_id"]] = cs

    n = 0
    with open(ROOT / "data_processed" / "intervention_specs.jsonl") as fin, open(args.out, "w") as fout:
        for line in fin:
            spec = json.loads(line)
            if spec.get("rejected"):
                continue
            cs = sets[spec["set_id"]]
            prompt = template.format(
                base_context=spec["base_context"],
                preference=spec["surface_form"],
                candidates=serialize_candidates(cs["items"]),
                n=len(cs["items"]),
            )
            fout.write(json.dumps({
                "request_id": f"{spec['spec_id']}-{args.variant}",
                "spec_id": spec["spec_id"],
                "set_id": spec["set_id"],
                "prompt_variant": args.variant,
                "prompt_version": str(cfg["prompt_versions"][args.variant]),
                "target_attribute": spec["target_attribute"],
                "strength_level": spec["strength_level"],
                "template_family": spec["template_family"],
                "prompt": prompt,
            }) + "\n")
            n += 1
    print(f"rendered {n} requests -> {args.out}")


if __name__ == "__main__":
    main()
