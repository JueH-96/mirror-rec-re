"""Generate ordered intervention chains + surface forms with round-trip validation.

Spec-first (RESEARCH_BRIEF §7.5): structured specs are the ground truth; natural
language is rendered from three deterministic template families per strength
level and must pass the independent round-trip parser (§7.6). Family T3 is
reserved as the held-out wording family for the unseen-paraphrase analysis.
"""
import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

ATTR_PHRASES = {
    "price": "a low price",
    "weight": "a light weight",
    "battery_life": "a long battery life",
    "storage": "a large storage capacity",
}
ATTR_NOUN = {
    "price": "price",
    "weight": "weight",
    "battery_life": "battery life",
    "storage": "storage capacity",
}

# Strength templates: level -> family -> template. All positive polarity.
# {attr} = ATTR_PHRASES entry, {noun} = ATTR_NOUN entry.
TEMPLATES = {
    1: {
        "T1": "I care a little about {attr}.",
        "T2": "{noun} matters somewhat to me.",
        "T3": "It would be mildly nice to have {attr}.",
    },
    2: {
        "T1": "I care about {attr}.",
        "T2": "{noun} is important to me.",
        "T3": "I would definitely prefer {attr}.",
    },
    3: {
        "T1": "I care a lot about {attr}.",
        "T2": "{noun} is very important to me.",
        "T3": "I strongly prefer {attr}.",
    },
    4: {
        "T1": "I care about {attr} above everything else.",
        "T2": "{noun} is my single most important requirement.",
        "T3": "My absolute top priority is {attr}.",
    },
}

# Independent round-trip parser vocabulary (kept deliberately separate from the
# generation templates: it matches cue words, not whole templates).
STRENGTH_CUES = {
    4: [r"above everything", r"single most important", r"absolute top priority"],
    3: [r"\ba lot\b", r"very important", r"strongly prefer"],
    2: [r"(?<!mildly )(?<!very )important(?! requirement)", r"I care about(?! .*above)", r"definitely prefer"],
    1: [r"\ba little\b", r"somewhat", r"mildly"],
}
ATTR_CUES = {
    "price": [r"price"],
    "weight": [r"weight"],
    "battery_life": [r"battery"],
    "storage": [r"storage"],
}
NEGATION = re.compile(r"\b(not|don't|do not|hardly|no longer)\b", re.I)


def roundtrip_parse(text):
    """Parse surface form back to (attribute, polarity, strength); None if ambiguous."""
    attrs = [a for a, cues in ATTR_CUES.items() if any(re.search(c, text, re.I) for c in cues)]
    if len(attrs) != 1:
        return None
    if NEGATION.search(text):
        return None
    for level in (4, 3, 2, 1):  # strongest cue wins
        if any(re.search(c, text, re.I) for c in STRENGTH_CUES[level]):
            return {"attribute": attrs[0], "polarity": "positive", "strength": level}
    return None


def validate_surface(spec, text):
    """§7.6 checks: round-trip, attribute/polarity/strength consistency, forbidden words."""
    parsed = roundtrip_parse(text)
    if parsed is None:
        return False, "roundtrip_unparseable"
    if parsed["attribute"] != spec["target_attribute"]:
        return False, "attribute_mismatch"
    if parsed["polarity"] != spec["polarity"]:
        return False, "polarity_mismatch"
    if parsed["strength"] != spec["strength_level"]:
        return False, "strength_mismatch"
    for other, cues in ATTR_CUES.items():
        if other != spec["target_attribute"] and any(re.search(c, text, re.I) for c in cues):
            return False, "off_target_attribute_word"
    return True, "ok"


BASE_CONTEXT = ("I am looking for a laptop for everyday work "
                "(documents, browsing, video calls).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--candidate-sets", default=str(ROOT / "data_processed" / "candidate_sets.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data_processed" / "intervention_specs.jsonl"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    levels = range(1, cfg["strength_levels"] + 1)
    n_reject = 0
    with open(args.candidate_sets) as fin, open(args.out, "w") as fout:
        for line in fin:
            cs = json.loads(line)
            attr = cs["target_attribute"]
            for level in levels:
                for fam, tmpl in TEMPLATES[level].items():
                    text = tmpl.format(attr=ATTR_PHRASES[attr], noun=ATTR_NOUN[attr].capitalize()
                                       if tmpl.startswith("{noun}") else ATTR_NOUN[attr])
                    spec = {
                        "spec_id": f"{cs['set_id']}-L{level}-{fam}",
                        "set_id": cs["set_id"],
                        "target_attribute": attr,
                        "polarity": "positive",
                        "strength_level": level,
                        "template_family": fam,
                        "base_context": BASE_CONTEXT,
                        "surface_form": text,
                    }
                    ok, reason = validate_surface(spec, text)
                    spec["validation"] = reason
                    if not ok:
                        n_reject += 1
                        spec["rejected"] = True
                    fout.write(json.dumps(spec) + "\n")
    print(f"interventions written; rejected={n_reject}")
    if n_reject:
        raise SystemExit(f"FATAL: {n_reject} surface forms failed round-trip validation")


if __name__ == "__main__":
    main()
