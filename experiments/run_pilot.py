"""Pilot runner: execute one (model x prompt-variant) sweep over rendered requests.

Usage:
    python experiments/run_pilot.py --model qwen2.5-14b-instruct --variant plain \
        [--limit N] [--sets CS0000,CS0001]

Outputs per-request parsed rankings to results/raw/{model}/{variant}.jsonl.
Thread-pooled for vLLM throughput; the disk cache makes reruns idempotent.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_adapters.llm_client import LLMClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", default=None, help="comma-separated set_id filter")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg = cfg["models"][args.model]
    client = LLMClient(args.model, model_cfg, temperature=cfg["temperature"],
                       seed=cfg["seed"], max_retries=cfg["max_retries"])

    req_path = ROOT / "data_processed" / f"rendered_{args.variant}.jsonl"
    requests = [json.loads(l) for l in open(req_path)]
    if args.sets:
        keep = set(args.sets.split(","))
        requests = [r for r in requests if r["set_id"] in keep]
    if args.limit:
        requests = requests[: args.limit]

    sets = {}
    with open(ROOT / "data_processed" / "candidate_sets.jsonl") as f:
        for line in f:
            cs = json.loads(line)
            sets[cs["set_id"]] = cs

    out_dir = ROOT / "results" / "raw" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.variant}.jsonl"
    done_ids = set()
    if out_path.exists():
        done_ids = {json.loads(l)["request_id"] for l in open(out_path)}
    todo = [r for r in requests if r["request_id"] not in done_ids]
    print(f"{len(todo)} requests to run ({len(done_ids)} already done)")

    def work(req):
        ids = [it["item_id"] for it in sets[req["set_id"]]["items"]]
        max_tok = 2048 if args.variant == "explain_then_rank" else 512
        res = client.rank(req, ids, max_tokens=max_tok)
        return {
            "request_id": req["request_id"],
            "spec_id": req["spec_id"],
            "set_id": req["set_id"],
            "model": args.model,
            "prompt_variant": args.variant,
            "target_attribute": req["target_attribute"],
            "strength_level": req["strength_level"],
            "template_family": req["template_family"],
            "parse_status": res["parse_status"],
            "ranking": res["parsed"],
        }

    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
        futures = [ex.submit(work, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            fout.write(json.dumps(fut.result()) + "\n")
            fout.flush()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
