"""Parser-noise axis (study3_plan §0-P2; user-ruled M3 evidence axis).

Simulates a noisy (attribute, strength-level) parser at deployment time:
with probability p per request the parsed representation is corrupted —
50% strength off-by-one (±1, clamped to [1,4]), 50% attribute confusion
(uniform over the other 3 registry attributes). Corruption tables are
deterministic (seed 20260731 + rate) and SHARED across systems so the
comparison is paired.

Only parse-consuming systems need recomputation:
  - S8 (mirror): g_a(l) conditions on parsed (a, l). The backbone hidden
    states depend only on the prompt text, so we run the backbone once per
    request and re-score the heads under each noise rate (clean scores are
    spot-checked against the stored M2 raw outputs).
  - S1 (projection): handled by src/projection/study3_projection_noise.py.
S7/S2..S5 (base mode) consume no parsed inputs -> outputs are identical
under parser noise by construction (recorded as such, not re-run).

Usage:
  python experiments/run_parser_noise.py --rates 5,10,20 --seeds 1042,1043,1044
Outputs:
  results/study3/parser_noise/corruptions_rate{p}.json
  results/study3/raw/S8N{p}_llama_seed{s}/test_main.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.data_module import ChainDataset  # noqa: E402
from src.training.scorer_model import ATTRS, RerankScorer  # noqa: E402

SPLIT = "test_main"


def build_tables(rates):
    """One corruption table per rate, keyed 'set|fam|level' -> corruption."""
    keys = []
    eligible = set(json.load(open(
        ROOT / f"data_processed/study3/eligible_sets_{SPLIT}.json")))
    for line in open(ROOT / f"data_processed/study3/intervention_specs_{SPLIT}.jsonl"):
        sp = json.loads(line)
        if sp.get("rejected") or sp["set_id"] not in eligible:
            continue
        keys.append((sp["set_id"], sp["template_family"],
                     sp["strength_level"], sp["target_attribute"]))
    keys.sort()
    out_dir = ROOT / "results/study3/parser_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for rate in rates:
        rng = np.random.default_rng(20260731 + rate)
        table = {}
        for set_id, fam, level, attr in keys:
            if rng.random() >= rate / 100:
                continue
            if rng.random() < 0.5:  # strength off-by-one
                delta = 1 if rng.random() < 0.5 else -1
                new = level + delta
                if not 1 <= new <= 4:
                    new = level - delta
                table[f"{set_id}|{fam}|{level}"] = {"kind": "level", "level": new}
            else:  # attribute confusion
                others = [a for a in ATTRS if a != attr]
                table[f"{set_id}|{fam}|{level}"] = {
                    "kind": "attr", "attr": str(rng.choice(others))}
        p = out_dir / f"corruptions_rate{rate}.json"
        p.write_text(json.dumps(table, indent=0))
        print(f"rate {rate}%: {len(table)}/{len(keys)} requests corrupted -> {p}")
        tables[rate] = table
    return tables


def score_heads(model, hi, attr_idx, level):
    """Head math identical to RerankScorer.forward (mirror mode)."""
    b = model.head_base(hi).squeeze(-1)
    u = model.head_util(hi).squeeze(-1)
    r = 0.1 * model.head_resid(hi).squeeze(-1)
    g = model.g_value(attr_idx, level).to(b.dtype)
    return b + g * u + r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="5,10,20")
    ap.add_argument("--seeds", default="1042,1043,1044")
    ap.add_argument("--spotcheck", type=int, default=50,
                    help="chains to verify clean re-scores against stored M2 raw")
    args = ap.parse_args()
    rates = [int(r) for r in args.rates.split(",")]
    tables = build_tables(rates)

    for seed in args.seeds.split(","):
        run = f"S8_llama_seed{seed}"
        cfg = yaml.safe_load(open(ROOT / f"configs/study3/{run}.yaml"))
        assert cfg.get("mode") == "mirror", "parser-noise re-scoring is mirror-only"
        device = "cuda"
        model = RerankScorer(cfg["model_name"], mode="mirror",
                             lora_r=cfg.get("lora_r", 16)).to(device)
        state = torch.load(ROOT / f"experiments/study3/{run}/final.pt",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()

        stored = {}
        for line in open(ROOT / f"results/study3/raw/{run}/{SPLIT}.jsonl"):
            r = json.loads(line)
            stored[r["request_id"]] = r["ranking"]

        outs = {}
        for rate in rates:
            d = ROOT / f"results/study3/raw/S8N{rate}_llama_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            outs[rate] = open(d / f"{SPLIT}.jsonl", "w")

        ds = ChainDataset(SPLIT, model.tokenizer)
        max_dev, checked = 0.0, 0
        with torch.no_grad():
            for idx in range(len(ds)):
                ex = ds.get(idx)
                cs = ds.sets[ex["set_id"]]
                item_ids = [it["item_id"] for it in cs["items"]]
                for lv in ex["levels"]:
                    out = model.backbone(
                        input_ids=lv["input_ids"].unsqueeze(0).to(device),
                        attention_mask=lv["attention_mask"].unsqueeze(0).to(device),
                        output_hidden_states=True)
                    hi = out.hidden_states[-1][0][lv["marker_positions"].to(device)]
                    key = f"{ex['set_id']}|{ex['family']}|{lv['level']}"
                    if checked < args.spotcheck * 4:
                        clean = score_heads(model, hi, ex["attr_idx"], lv["level"])
                        sc = clean.float().cpu().tolist()
                        ref = stored[f"{ex['set_id']}-L{lv['level']}-{ex['family']}-trained"]
                        refmap = dict(zip(ref["ids"], ref["scores"]))
                        max_dev = max(max_dev, max(
                            abs(sc[i] - refmap[item_ids[i]]) for i in range(len(item_ids))))
                        checked += 1
                    for rate in rates:
                        c = tables[rate].get(key)
                        attr_idx, level = ex["attr_idx"], lv["level"]
                        if c and c["kind"] == "level":
                            level = c["level"]
                        elif c and c["kind"] == "attr":
                            attr_idx = ATTRS.index(c["attr"])
                        s = score_heads(model, hi, attr_idx, level).float().cpu().tolist()
                        order = sorted(range(len(item_ids)), key=lambda i: -s[i])
                        outs[rate].write(json.dumps({
                            "request_id": f"{ex['set_id']}-L{lv['level']}-{ex['family']}-N{rate}",
                            "spec_id": key, "set_id": ex["set_id"],
                            "model": f"S8N{rate}_llama_seed{seed}",
                            "prompt_variant": SPLIT,
                            # TRUE metadata: metrics are computed on the true
                            # chain structure; only the model inputs were noisy
                            "target_attribute": ex["attr"],
                            "strength_level": lv["level"],
                            "template_family": ex["family"], "parse_status": "ok",
                            "corrupted": bool(c), "corruption": c,
                            "ranking": {"ids": [item_ids[i] for i in order],
                                        "scores": [round(s[i], 6) for i in order]},
                        }) + "\n")
                if (idx + 1) % 100 == 0:
                    print(f"{run}: {idx + 1}/{len(ds)} chains", flush=True)
        for f in outs.values():
            f.close()
        print(f"{run}: SPOTCHECK max abs dev vs stored M2 scores = {max_dev:.6g} "
              f"({checked} requests)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
