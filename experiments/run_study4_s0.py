"""S0 (zero-shot plain) evaluation on Study 4 splits.

Same mechanics as run_study3_s0.py (plain prompt via served vLLM model, raw
schema unchanged, idempotent), with per-split sets/eligible locations and
domain-correct candidate units (phone split serializes battery_life in mAh).

Usage:
    python experiments/run_study4_s0.py --model llama-3.1-8b-instruct \
        --family llama [--splits test_unseen_item,test_domain_phone,test_t4_word]

Output: results/study4/raw/S0_<family>/<split>.jsonl
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

from src.intervention_generation.render_requests import serialize_candidates  # noqa: E402
from src.model_adapters.llm_client import LLMClient  # noqa: E402

PHONE_UNITS = {"price": "USD", "weight": "kg", "battery_life": "mAh", "storage": "GB"}

SPLITS = {
    "test_unseen_item": {
        "sets": "data_processed/study4/candidate_sets_test_unseen_item.jsonl",
        "units": None},
    "test_domain_phone": {
        "sets": "data_processed/study4/candidate_sets_test_domain_phone.jsonl",
        "units": PHONE_UNITS},
    "test_t4_word": {
        "sets": "data_processed/study3/candidate_sets_test_main.jsonl",
        "units": None},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--model", required=True, help="pilot.yaml models key")
    ap.add_argument("--family", required=True, help="output tag, e.g. llama")
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    template = (ROOT / cfg["prompt_versions"]["plain"]).read_text()
    client = LLMClient(args.model, cfg["models"][args.model],
                       temperature=cfg["temperature"], seed=cfg["seed"],
                       max_retries=cfg["max_retries"])
    run = f"S0_{args.family}"

    for split in args.splits.split(","):
        loc = SPLITS[split]
        sets = {}
        for line in open(ROOT / loc["sets"]):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
        eligible = set(json.load(open(
            ROOT / f"data_processed/study4/eligible_sets_{split}.json")))

        requests = []
        for line in open(ROOT / f"data_processed/study4/intervention_specs_{split}.jsonl"):
            spec = json.loads(line)
            if spec.get("rejected") or spec["set_id"] not in eligible:
                continue
            cs = sets[spec["set_id"]]
            requests.append({
                "request_id": f"{spec['spec_id']}-{run}",
                "spec_id": spec["spec_id"], "set_id": spec["set_id"],
                "target_attribute": spec["target_attribute"],
                "strength_level": spec["strength_level"],
                "template_family": spec["template_family"],
                "prompt": template.format(
                    base_context=spec["base_context"],
                    preference=spec["surface_form"],
                    candidates=serialize_candidates(cs["items"], units=loc["units"]),
                    n=len(cs["items"])),
            })

        out_dir = ROOT / "results/study4/raw" / run
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.jsonl"
        done = ({json.loads(l)["request_id"] for l in open(out_path)}
                if out_path.exists() else set())
        todo = [r for r in requests if r["request_id"] not in done]
        print(f"{split}: {len(todo)} to run ({len(done)} done)")

        def work(req):
            ids = [it["item_id"] for it in sets[req["set_id"]]["items"]]
            res = client.rank(req, ids, max_tokens=512)
            return {"request_id": req["request_id"], "spec_id": req["spec_id"],
                    "set_id": req["set_id"], "model": run,
                    "prompt_variant": split,
                    "target_attribute": req["target_attribute"],
                    "strength_level": req["strength_level"],
                    "template_family": req["template_family"],
                    "parse_status": res["parse_status"],
                    "ranking": res["parsed"]}

        with open(out_path, "a") as fout, \
                ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
            futures = [ex.submit(work, r) for r in todo]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                fout.write(json.dumps(fut.result()) + "\n")
                fout.flush()
        print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
