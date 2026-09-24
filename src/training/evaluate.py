"""Evaluate a trained scorer on a Study 3 split, emitting pilot-schema raw
results so src/metrics/relational_metrics.py and the stats pipeline apply
unchanged.

Deployment mode is single-request: each level scored independently (no chain
information at inference), matching study3_plan.md §0-P1.

Usage:
  python src/training/evaluate.py --config <run config> --ckpt <path.pt> \
      --split test_main --out results/study3/raw/<system_seed>/<split>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.data_module import ChainDataset  # noqa: E402
from src.training.scorer_model import RerankScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda"
    model = RerankScorer(cfg["model_name"], mode=cfg.get("mode", "base"),
                         lora_r=cfg.get("lora_r", 16),
                         g_const=cfg.get("g_const", False),
                         g_shared=cfg.get("g_shared", False)).to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    ds = ChainDataset(args.split, model.tokenizer)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["request_id"] for l in open(out_path)}
    with open(out_path, "a") as fout, torch.no_grad():
        for idx in range(len(ds)):
            ex = ds.get(idx)
            cs = ds.sets[ex["set_id"]]
            item_ids = [it["item_id"] for it in cs["items"]]
            for lv in ex["levels"]:
                rid = f"{ex['set_id']}-L{lv['level']}-{ex['family']}-trained"
                if rid in done:
                    continue
                s = model(lv["input_ids"].to(device), lv["attention_mask"].to(device),
                          lv["marker_positions"].to(device), ex["attr_idx"], lv["level"])
                scores = s.float().cpu().tolist()
                order = sorted(range(len(item_ids)), key=lambda i: -scores[i])
                fout.write(json.dumps({
                    "request_id": rid, "spec_id": rid, "set_id": ex["set_id"],
                    "model": cfg["run_name"], "prompt_variant": args.split,
                    "target_attribute": ex["attr"], "strength_level": lv["level"],
                    "template_family": ex["family"], "parse_status": "ok",
                    "ranking": {"ids": [item_ids[i] for i in order],
                                "scores": [round(scores[i], 6) for i in order]},
                }) + "\n")
            if (idx + 1) % 100 == 0:
                print(f"{idx + 1}/{len(ds)} chains", flush=True)
    print(f"eval done -> {out_path}")


if __name__ == "__main__":
    main()
