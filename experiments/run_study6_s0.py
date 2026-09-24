"""S0 (zero-shot plain) evaluation on Study 6 robustness splits.

Same mechanics as run_study4_s0.py (plain prompt via served vLLM model, raw
schema unchanged, idempotent). All Study 6 splits are laptop-domain (default
units). Specs/eligible live in data_processed/study6/; sets location varies
per split (derived splits reuse study3 test_main sets).

Usage:
    python experiments/run_study6_s0.py --model llama-3.1-8b-instruct \
        --family llama [--splits test_order_p1,...]

Output: results/study6/raw/S0_<family>/<split>.jsonl
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

TM_SETS = "data_processed/study3/candidate_sets_test_main.jsonl"
SPLIT_SETS = {}
for _k in range(1, 6):
    SPLIT_SETS[f"test_order_p{_k}"] = f"data_processed/study6/candidate_sets_test_order_p{_k}.jsonl"
for _t in ("d1", "d2", "d4"):
    SPLIT_SETS[f"test_distract_{_t}"] = TM_SETS
for _s in ("12", "123", "1234"):
    SPLIT_SETS[f"test_mix{_s}"] = TM_SETS
SPLIT_SETS["test_neartied"] = "data_processed/study6/candidate_sets_test_neartied.jsonl"
SPLIT_SETS["test_conflict"] = "data_processed/study6/candidate_sets_test_conflict.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--model", required=True, help="pilot.yaml models key")
    ap.add_argument("--family", required=True, help="output tag, e.g. llama")
    ap.add_argument("--splits", default=",".join(SPLIT_SETS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    template = (ROOT / cfg["prompt_versions"]["plain"]).read_text()
    client = LLMClient(args.model, cfg["models"][args.model],
                       temperature=cfg["temperature"], seed=cfg["seed"],
                       max_retries=cfg["max_retries"])
    run = f"S0_{args.family}"

    for split in args.splits.split(","):
        sets = {}
        for line in open(ROOT / SPLIT_SETS[split]):
            cs = json.loads(line)
            sets[cs["set_id"]] = cs
        eligible = set(json.load(open(
            ROOT / f"data_processed/study6/eligible_sets_{split}.json")))

        requests = []
        for line in open(ROOT / f"data_processed/study6/intervention_specs_{split}.jsonl"):
            spec = json.loads(line)
            if spec.get("rejected") or spec["set_id"] not in eligible:
                continue
            cs = sets[spec["set_id"]]
            requests.append({
                "request_id": f"{spec['spec_id']}-{run}-{split}",
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

        out_dir = ROOT / "results/study6/raw" / run
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
