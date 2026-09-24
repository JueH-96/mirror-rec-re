"""Study 7 Stage A -> Stage B ablation selection (plan Sec 5 rule, applied
mechanically; user released both stages 2026-08-12 so this script IS the
selection step -- no discretion).

Entry rule per ablation X (seed 1042, test_main metrics):
  select X iff  (4c unseen-transition pooled violation rate of ablX)
                - (same rate of full S8c seed1042)  >= 1pp
           or   RR guardrail trips  (RR_ablX - RR_S0_llama < -0.05)
           or   NDCG guardrail trips (NDCG5_S8c - NDCG5_ablX > 0.02)

Outputs:
  results/study7/stageA_selection.json          (audit record)
  configs/study7/queue_stageB_ablation.txt      (selected runs x seeds 1043/1044)
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study7_common import (  # noqa: E402
    UNSEEN_TRANSITIONS, chain_rates, load_metrics, load_sets, ndcg5)

ABLATIONS = ["S8c_ablg", "S8c_ablga", "S8c_ablicr", "S8c_ablchain",
             "S8c_ablresp", "S8c_ablot"]
REF = "S8c_llama_seed1042"
GATE_4C = 0.01
GATE_RR = -0.05
GATE_NDCG = 0.02


def rate_4c(run):
    cr = chain_rates(run, "test_main", transitions=UNSEEN_TRANSITIONS)
    return None if cr is None else float(cr.mean())


def rr(run):
    df = load_metrics(run, "test_main")
    return None if df is None else float(df["responsive"].mean())


def main():
    registry = yaml.safe_load(open(ROOT / "attribute_registry/attribute_registry.yaml"))
    sets = load_sets(ROOT / "data_processed/study3/candidate_sets_test_main.jsonl")
    eligible = set(json.loads(
        (ROOT / "data_processed/study3/eligible_sets_test_main.json").read_text()))

    ref_4c, ref_rr = rate_4c(REF), rr(REF)
    ref_ndcg = ndcg5(REF, "test_main", sets, registry, eligible)
    rr_s0 = rr("S0_llama")
    assert None not in (ref_4c, ref_rr, ref_ndcg, rr_s0), "missing baselines"

    out = {"rule": "select iff d4c>=1pp or RR-RR_S0<-5pp or NDCG5 drop vs "
                   "S8c>0.02 (plan Sec 5, frozen 2026-08-12)",
           "reference": {"run": REF, "rate_4c": round(ref_4c, 4),
                         "RR": round(ref_rr, 4), "NDCG5": round(ref_ndcg, 4),
                         "RR_S0_llama": round(rr_s0, 4)},
           "ablations": {}, "selected": []}
    for abl in ABLATIONS:
        run = f"{abl}_llama_seed1042"
        r4c, rrr = rate_4c(run), rr(run)
        nd = ndcg5(run, "test_main", sets, registry, eligible)
        if None in (r4c, rrr, nd):
            raise SystemExit(f"missing Stage A metrics for {run}")
        d4c = r4c - ref_4c
        rr_trip = (rrr - rr_s0) < GATE_RR
        nd_trip = (ref_ndcg - nd) > GATE_NDCG
        sel = (d4c >= GATE_4C) or rr_trip or nd_trip
        out["ablations"][abl] = {
            "rate_4c": round(r4c, 4), "d4c_vs_S8c": round(d4c, 4),
            "RR": round(rrr, 4), "RR_delta_vs_S0": round(rrr - rr_s0, 4),
            "RR_guardrail_trip": bool(rr_trip),
            "NDCG5": round(nd, 4), "NDCG5_drop_vs_S8c": round(ref_ndcg - nd, 4),
            "NDCG_guardrail_trip": bool(nd_trip), "selected": bool(sel)}
        if sel:
            out["selected"].append(abl)

    dst = ROOT / "results/study7/stageA_selection.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    q = ROOT / "configs/study7/queue_stageB_ablation.txt"
    q.write_text("".join(f"{abl}_llama_seed{s}\n"
                         for abl in out["selected"] for s in (1043, 1044)))
    print(f"selected: {out['selected']} -> {dst}, queue {q}")


if __name__ == "__main__":
    main()
