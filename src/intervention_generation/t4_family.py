"""T4 reinforced unseen-wording family (Study 4a, EXPLORATORY — not preregistered).

Colloquial/indirect strength ladder whose cue vocabulary is disjoint from the
T1-T3 cue table. Calibration gate (study4_plan.md §2/4a) — all three checks
must pass before any T4 evaluation data is generated:
  1. round-trip: every rendered T4 surface parses back to the correct
     (attribute, positive, level) under the T4-extended parser;
  2. unseen-ness certificate: the ORIGINAL T1-T3 parser finds no strength
     cue in any T4 surface (proves the wording sits outside the trained
     cue vocabulary);
  3. collision: no T4 surface matches any T1-T3 strength cue and no T1-T3
     surface matches any T4 strength cue.
The 4-level ladder is additionally submitted for human monotonicity review
in the M3 stop report.

Usage: python src/intervention_generation/t4_family.py   # runs calibration
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.intervention_generation.generate_interventions import (  # noqa: E402
    ATTR_CUES, ATTR_NOUN, ATTR_PHRASES, NEGATION, STRENGTH_CUES, TEMPLATES,
    roundtrip_parse)

T4_TEMPLATES = {
    1: "Having {attr} would be a small plus for me.",
    2: "I'm keen on getting {attr}.",
    3: "Honestly, I'm really keen on getting {attr}.",
    4: "For me, nothing matters more than {noun}.",
}

T4_STRENGTH_CUES = {
    4: [r"nothing matters more"],
    3: [r"really keen"],
    2: [r"keen on"],
    1: [r"a small plus"],
}


def render_t4(attr, level):
    tmpl = T4_TEMPLATES[level]
    return tmpl.format(attr=ATTR_PHRASES[attr], noun=ATTR_NOUN[attr])


def t4_parse(text):
    """T4-extended round-trip parser (same contract as roundtrip_parse)."""
    attrs = [a for a, cues in ATTR_CUES.items() if any(re.search(c, text, re.I) for c in cues)]
    if len(attrs) != 1:
        return None
    if NEGATION.search(text):
        return None
    for level in (4, 3, 2, 1):
        if any(re.search(c, text, re.I) for c in T4_STRENGTH_CUES[level]):
            return {"attribute": attrs[0], "polarity": "positive", "strength": level}
    return None


def validate_t4_surface(spec, text):
    """validate_surface contract for T4 surfaces (T4-extended parser)."""
    parsed = t4_parse(text)
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


def calibrate():
    report = {"family": "T4", "status": "exploratory (not preregistered)",
              "templates": T4_TEMPLATES, "cues": {k: v for k, v in T4_STRENGTH_CUES.items()},
              "checks": {}, "failures": []}
    attrs = list(ATTR_PHRASES)

    # 1. round-trip under T4-extended parser
    for attr in attrs:
        for level in range(1, 5):
            text = render_t4(attr, level)
            p = t4_parse(text)
            if p != {"attribute": attr, "polarity": "positive", "strength": level}:
                report["failures"].append(f"roundtrip: {attr} L{level} -> {p} ({text!r})")
    report["checks"]["roundtrip_all_correct"] = not report["failures"]

    # 2. unseen-ness certificate: original parser must find NO strength cue
    unseen_fail = []
    for attr in attrs:
        for level in range(1, 5):
            text = render_t4(attr, level)
            if roundtrip_parse(text) is not None:
                unseen_fail.append(f"original parser parsed T4: {attr} L{level} ({text!r})")
    report["checks"]["unseen_certificate"] = not unseen_fail
    report["failures"] += unseen_fail

    # 3. cross-family cue collision, both directions
    coll = []
    for attr in attrs:
        for level in range(1, 5):
            t4_text = render_t4(attr, level)
            for lv, cues in STRENGTH_CUES.items():
                if any(re.search(c, t4_text, re.I) for c in cues):
                    coll.append(f"T4 {attr} L{level} hits T1-T3 cue of L{lv}")
            for fam in ("T1", "T2", "T3"):
                tmpl = TEMPLATES[level][fam]
                old = tmpl.format(attr=ATTR_PHRASES[attr],
                                  noun=ATTR_NOUN[attr].capitalize() if tmpl.startswith("{noun}") else ATTR_NOUN[attr])
                for lv, cues in T4_STRENGTH_CUES.items():
                    if any(re.search(c, old, re.I) for c in cues):
                        coll.append(f"{fam} {attr} L{level} hits T4 cue of L{lv}")
    report["checks"]["no_cross_family_collision"] = not coll
    report["failures"] += coll

    report["calibration_passed"] = all(report["checks"].values())
    return report


def main():
    report = calibrate()
    out = ROOT / "results" / "study4" / "t4_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["calibration_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
