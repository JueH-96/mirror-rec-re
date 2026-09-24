"""Shared helpers for Study 7 statistics: metric/raw routing that unifies
study7 runs (results/study7/) with the study3/4/6 baselines routed by
study6_h6. All rate definitions are identical to study6_h6 (chain-level
means over per-transition violation rows)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study6_h6 import (  # noqa: E402,F401
    LLAMA_SEEDS, ORDER_SPLITS, load_metrics as _load_metrics_base,
    metrics_path as _metrics_path_base, raw_path as _raw_path_base,
    ndcg5 as _ndcg5_base, load_sets, seed_summary, paired_perm, holm)

STUDY7_PREFIXES = ("S2perm", "S7perm", "S8perm", "S7cperm", "S8cperm",
                   "S8c_abl")
UNSEEN_TRANSITIONS = ["2->3", "3->4"]


def is_study7(run):
    return run.startswith(STUDY7_PREFIXES)


def metrics_path(run, split):
    if is_study7(run):
        return ROOT / f"results/study7/metrics_{run}_{split}.parquet"
    return _metrics_path_base(run, split)


def raw_path(run, split):
    if is_study7(run):
        return ROOT / f"results/study7/raw/{run}/{split}.jsonl"
    return _raw_path_base(run, split)


def load_metrics(run, split):
    p = metrics_path(run, split)
    if not p.exists():
        return None
    df = pd.read_parquet(p).copy()
    df["chain_id"] = df["set_id"] + "|" + df["template_family"]
    return df


def chain_rates(run, split, col="violation", transitions=None):
    df = load_metrics(run, split)
    if df is None:
        return None
    if transitions is not None:
        df = df[df["transition"].isin(transitions)]
    return df.groupby("chain_id")[col].mean()


def ndcg5(run, split, sets, registry, eligible):
    """study6_h6.ndcg5 semantics with study7-aware raw routing."""
    import json

    import numpy as np

    from src.candidate_generation.rule_baseline import LEVEL_W, norm_utility
    from src.projection.isotonic_projection import ndcg_at_k
    p = raw_path(run, split)
    if not p.exists():
        return None
    vals = []
    for line in open(p):
        r = json.loads(line)
        if r["set_id"] not in eligible or r["parse_status"] != "ok":
            continue
        cs = sets[r["set_id"]]
        attr, level = r["target_attribute"], r["strength_level"]
        gains = {}
        for it in cs["items"]:
            base = np.mean([norm_utility(it[a], registry[a])
                            for a in registry if it.get(a) is not None])
            gains[it["item_id"]] = 0.5 * base + LEVEL_W[level] * norm_utility(
                it[attr], registry[attr])
        vals.append(ndcg_at_k(r["ranking"]["ids"], gains))
    return float(np.mean(vals)) if vals else None
