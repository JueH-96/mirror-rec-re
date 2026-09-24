"""Verify a trainable-only checkpoint reproduces the recorded eval scores.
Usage: spotcheck_ckpt.py <config.yaml> <ckpt.pt> <raw_results.jsonl>"""
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.data_module import ChainDataset  # noqa: E402
from src.training.scorer_model import RerankScorer  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
model = RerankScorer(cfg["model_name"], mode=cfg.get("mode", "base"),
                     lora_r=cfg.get("lora_r", 16)).to("cuda")
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state["model"], strict=False)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
model.eval()

recorded = {}
for l in open(sys.argv[3]):
    r = json.loads(l)
    recorded[r["request_id"]] = dict(zip(r["ranking"]["ids"], r["ranking"]["scores"]))

ds = ChainDataset("val", model.tokenizer)
ex = ds.get(0)
cs = ds.sets[ex["set_id"]]
item_ids = [it["item_id"] for it in cs["items"]]
ok = True
with torch.no_grad():
    for lv in ex["levels"][:2]:
        rid = f"{ex['set_id']}-L{lv['level']}-{ex['family']}-trained"
        s = model(lv["input_ids"].to("cuda"), lv["attention_mask"].to("cuda"),
                  lv["marker_positions"].to("cuda"), ex["attr_idx"], lv["level"])
        got = {item_ids[i]: round(float(s[i]), 6) for i in range(len(item_ids))}
        want = recorded[rid]
        diffs = [abs(got[k] - want[k]) for k in want]
        print(rid, "max diff:", max(diffs))
        ok = ok and max(diffs) < 1e-3
print("SPOTCHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
