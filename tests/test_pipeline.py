"""Unit tests per RESEARCH_BRIEF §14.3."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generation.rule_baseline import norm_utility  # noqa: E402
from src.intervention_generation.generate_interventions import (  # noqa: E402
    ATTR_PHRASES, ATTR_NOUN, TEMPLATES, roundtrip_parse, validate_surface)
from src.metrics.relational_metrics import chain_metrics, rbo  # noqa: E402
from src.model_adapters.llm_client import parse_ranking  # noqa: E402


# --- attribute direction normalization ---------------------------------------
def test_direction_normalization():
    lower_spec = {"valid_range": [0, 10], "preference_direction": "lower"}
    higher_spec = {"valid_range": [0, 10], "preference_direction": "higher"}
    assert norm_utility(2, lower_spec) > norm_utility(8, lower_spec)
    assert norm_utility(8, higher_spec) > norm_utility(2, higher_spec)


# --- intervention ordering: every rendered level parses back to itself -------
@pytest.mark.parametrize("attr", list(ATTR_PHRASES))
@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("fam", ["T1", "T2", "T3"])
def test_roundtrip_all_templates(attr, level, fam):
    tmpl = TEMPLATES[level][fam]
    text = tmpl.format(attr=ATTR_PHRASES[attr],
                       noun=ATTR_NOUN[attr].capitalize() if tmpl.startswith("{noun}") else ATTR_NOUN[attr])
    spec = {"target_attribute": attr, "polarity": "positive", "strength_level": level}
    ok, reason = validate_surface(spec, text)
    assert ok, f"{attr} L{level} {fam}: {reason} :: {text}"


def test_roundtrip_rejects_negation_and_ambiguity():
    assert roundtrip_parse("I do not care about price.") is None
    assert roundtrip_parse("I care about price and weight.") is None


# --- parse_ranking: §8.2 output validation -----------------------------------
VALID = ["A", "B", "C"]


def test_parse_ok_and_last_json_wins():
    txt = 'reasoning {"ranking": [{"item_id": "A", "score": 1}]} final: ' \
          '{"ranking": [{"item_id": "A", "score": 0.9}, {"item_id": "B", "score": 0.5}, ' \
          '{"item_id": "C", "score": 0.1}]}'
    parsed, status = parse_ranking(txt, VALID)
    assert status == "ok" and parsed["ids"] == ["A", "B", "C"]


@pytest.mark.parametrize("txt,expected", [
    ("no json here", "no_json"),
    ('{"ranking": [{"item_id": "A", "score": 0.9}, {"item_id": "A", "score": 0.1}, '
     '{"item_id": "B", "score": 0.2}]}', "duplicate_items"),
    ('{"ranking": [{"item_id": "A", "score": 0.9}, {"item_id": "B", "score": 0.5}, '
     '{"item_id": "X", "score": 0.1}]}', "foreign_item"),
    ('{"ranking": [{"item_id": "A", "score": 0.9}, {"item_id": "B", "score": 0.5}]}',
     "missing_items"),
])
def test_parse_failures(txt, expected):
    _, status = parse_ranking(txt, VALID)
    assert status == expected


# --- metric implementations ---------------------------------------------------
def _mk_chain(deltas):
    """Build a 2-item chain whose focal score gap follows `deltas`."""
    chain = {}
    for k, d in enumerate(deltas, start=1):
        si, sj = 0.5 + d / 2, 0.5 - d / 2
        ids = ["I", "J"] if d >= 0 else ["J", "I"]
        scores = [max(si, sj), min(si, sj)]
        chain[k] = {"parse_status": "ok", "ranking": {"ids": ids, "scores": scores}}
    return chain


def test_monotone_chain_no_violation():
    tr, ch, _ = chain_metrics(_mk_chain([0.1, 0.2, 0.3, 0.4]), "I", "J", 0.01, 0.01)
    assert not any(t["violation"] for t in tr)
    assert ch["chain_fully_consistent"]
    assert all(t["responsive"] for t in tr)


def test_violation_and_strict_reversal():
    tr, ch, _ = chain_metrics(_mk_chain([0.2, 0.1, -0.1, 0.3]), "I", "J", 0.01, 0.01)
    assert tr[0]["violation"] and not tr[0]["strict_reversal"]
    assert tr[1]["violation"] and tr[1]["strict_reversal"]
    assert not ch["chain_fully_consistent"]


def test_epsilon_tolerance():
    tr, _, _ = chain_metrics(_mk_chain([0.100, 0.095, 0.11, 0.12]), "I", "J", 0.01, 0.01)
    assert not tr[0]["violation"]  # drop of 0.005 within epsilon


def test_rbo_identical_is_one():
    assert rbo(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


# --- eligible pair + candidate invariance (integration on tiny generated data) -
def test_generation_pipeline_eligibility(tmp_path):
    env = {"PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, str(ROOT / "src/candidate_generation/generate_candidates.py")],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "src/intervention_generation/generate_interventions.py")],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "src/validation/validate_pipeline.py")],
                   check=True, cwd=ROOT)
    eligible = json.loads((ROOT / "data_processed" / "eligible_sets.json").read_text())
    report = json.loads((ROOT / "results" / "validation_report.json").read_text())
    assert report["gates"]["chain_invariance_ok"]
    assert len(eligible) >= 0.9 * report["gates"]["candidate_sets_total"]


def test_rule_baseline_has_zero_dvr():
    subprocess.run([sys.executable, str(ROOT / "src/candidate_generation/rule_baseline.py")],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "src/metrics/relational_metrics.py"),
                    "--model", "rule_baseline", "--variant", "plain"], check=True, cwd=ROOT)
    summary = json.loads((ROOT / "results" / "summary_rule_baseline_plain.json").read_text())
    assert summary["DVR"] == 0.0
    assert summary["SRR"] == 0.0
