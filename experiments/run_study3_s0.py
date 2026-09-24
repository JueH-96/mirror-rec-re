"""S0 (zero-shot plain) evaluation on Study 3 eval splits.

Renders the plain prompt over study3 intervention specs and sweeps them
through a served vLLM model, writing results in the same raw schema as the
trained-run evals so relational_metrics.py consumes them unchanged.

Usage:
    python experiments/run_study3_s0.py --model llama-3.1-8b-instruct \
        --family llama [--splits val,test_main,...]

Output: results/study3/raw/S0_<family>/<split>.jsonl (idempotent via
request-id skip + LLMClient disk cache).
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

SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]
SETS_FILE = {"val": "candidate_sets_val.jsonl",
             "test_unseen_attr": "candidate_sets_test_unseen_attr.jsonl"}


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
        sets = {}
        fname = SETS_FILE.get(split, "candidate_sets_test_main.jsonl")
        for line in open(ROOT / "data_processed/study3" / fname):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
        eligible = set(json.load(open(
            ROOT / f"data_processed/study3/eligible_sets_{split}.json")))

        requests = []
        for line in open(ROOT / f"data_processed/study3/intervention_specs_{split}.jsonl"):
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
                    candidates=serialize_candidates(cs["items"]),
                    n=len(cs["items"])),
            })

        out_dir = ROOT / "results/study3/raw" / run
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
