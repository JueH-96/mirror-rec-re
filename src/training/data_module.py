"""Chain-level dataset for Study 3 training/eval.

One example = one chain: (set, template_family) x levels 1..4. Provides
tokenized prompts with candidate marker positions, rule-oracle target order
per level, focal-pair indices, and registry margins for L_ICR (§6.3).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_generation.rule_baseline import LEVEL_W, norm_utility  # noqa: E402
from src.training.scorer_model import ATTRS  # noqa: E402

UNITS = {"price": "USD", "weight": "kg", "battery_life": "hours", "storage": "GB"}
UNITS_BY_DOMAIN = {
    "laptop": UNITS,
    "phone": {"price": "USD", "weight": "kg", "battery_life": "mAh", "storage": "GB"},
}
MARKER = " ##"

# split -> (data_processed subdir holding specs+eligible, candidate-sets relpath)
SPLIT_LOC = {
    "train":             ("study3", "study3/candidate_sets_train.jsonl"),
    "val":               ("study3", "study3/candidate_sets_val.jsonl"),
    "test_main":         ("study3", "study3/candidate_sets_test_main.jsonl"),
    "test_unseen_word":  ("study3", "study3/candidate_sets_test_main.jsonl"),
    "test_unseen_attr":  ("study3", "study3/candidate_sets_test_unseen_attr.jsonl"),
    "test_unseen_item":  ("study4", "study4/candidate_sets_test_unseen_item.jsonl"),
    "test_domain_phone": ("study4", "study4/candidate_sets_test_domain_phone.jsonl"),
    "test_t4_word":      ("study4", "study3/candidate_sets_test_main.jsonl"),
    "test_neartied":     ("study6", "study6/candidate_sets_test_neartied.jsonl"),
    "test_conflict":     ("study6", "study6/candidate_sets_test_conflict.jsonl"),
}
for _k in range(1, 6):
    SPLIT_LOC[f"test_order_p{_k}"] = ("study6", f"study6/candidate_sets_test_order_p{_k}.jsonl")
for _t in ("d1", "d2", "d4"):
    SPLIT_LOC[f"test_distract_{_t}"] = ("study6", "study3/candidate_sets_test_main.jsonl")
for _s in ("12", "123", "1234"):
    SPLIT_LOC[f"test_mix{_s}"] = ("study6", "study3/candidate_sets_test_main.jsonl")
REGISTRY_BY_DOMAIN = {
    "laptop": "attribute_registry.yaml",
    "phone": "attribute_registry_phones.yaml",
}


def serialize(items, base_context, preference, units=UNITS):
    lines = []
    for it in items:
        parts = [f"id={it['item_id']}"]
        for a in units:
            if it.get(a) is not None:
                parts.append(f"{a}={it[a]} {units[a]}")
        parts.append(f"cpu={it.get('cpu','n/a')}, ram={it.get('ram','n/a')} GB")
        lines.append("- " + ", ".join(parts) + MARKER)
    return (f"User request: {base_context} {preference}\n"
            f"Candidates:\n" + "\n".join(lines) + "\nScore each candidate.")


class ChainDataset:
    def __init__(self, split, tokenizer, registry_path=None, paraphrase_aug=False, seed=0,
                 levels=(1, 2, 3, 4), order_aug=False):
        self.tok = tokenizer
        self.levels = tuple(levels)
        self.order_aug = order_aug
        self.domain = "phone" if split == "test_domain_phone" else "laptop"
        self.units = UNITS_BY_DOMAIN[self.domain]
        self.registry = yaml.safe_load(open(registry_path or
                                            ROOT / "attribute_registry" / REGISTRY_BY_DOMAIN[self.domain]))
        subdir, sets_rel = SPLIT_LOC[split]
        d = ROOT / "data_processed" / subdir
        self.sets = {}
        for line in open(ROOT / "data_processed" / sets_rel):
            cs = json.loads(line)
            self.sets[cs["set_id"]] = cs
        eligible = set(json.loads((d / f"eligible_sets_{split}.json").read_text()))
        chains = {}
        for line in open(d / f"intervention_specs_{split}.jsonl"):
            sp = json.loads(line)
            if sp.get("rejected") or sp["set_id"] not in eligible:
                continue
            chains.setdefault((sp["set_id"], sp["template_family"]), {})[sp["strength_level"]] = sp
        self.chains = [(k, v) for k, v in sorted(chains.items()) if len(v) == 4]
        self.paraphrase_aug = paraphrase_aug
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.chains)

    def oracle_order(self, cs, level):
        gains = {}
        for it in cs["items"]:
            base = np.mean([norm_utility(it[a], self.registry[a])
                            for a in self.registry if it.get(a) is not None])
            gains[it["item_id"]] = 0.5 * base + LEVEL_W[level] * norm_utility(
                it[cs["target_attribute"]], self.registry[cs["target_attribute"]])
        return sorted(gains, key=lambda x: -gains[x]), gains

    def margin(self, cs, alpha=1.0, m_max=0.5):
        """§6.3 attribute-advantage margin for the focal pair."""
        spec = self.registry[cs["target_attribute"]]
        items = {it["item_id"]: it for it in cs["items"]}
        fi, fj = cs["focal_pair"]
        du = norm_utility(items[fi][cs["target_attribute"]], spec) - \
            norm_utility(items[fj][cs["target_attribute"]], spec)
        return alpha * float(np.clip(du, 0, m_max))

    def encode_level(self, cs, spec):
        text = serialize(cs["items"], spec["base_context"], spec["surface_form"], self.units)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=1024,
                       return_offsets_mapping=True)
        ids = enc.input_ids[0]
        offsets = enc.offset_mapping[0].tolist()
        # locate each candidate's marker by character position, then map to the
        # token whose span contains the marker's final character
        positions, start = [], 0
        for _ in cs["items"]:
            c = text.find(MARKER, start)
            assert c != -1, "marker missing in text"
            last_char = c + len(MARKER) - 1
            tok_idx = next(i for i, (a, b) in enumerate(offsets)
                           if a <= last_char < b)
            positions.append(tok_idx)
            start = c + len(MARKER)
        pos = torch.tensor(positions, dtype=torch.long)
        assert pos.numel() == len(cs["items"]), "marker mismatch"
        return ids, enc.attention_mask[0], pos

    def get(self, idx):
        (set_id, fam), levels = self.chains[idx]
        cs = self.sets[set_id]
        if self.order_aug:
            # permutation augmentation (study7): fresh candidate listing order
            # per visit, shared across the chain's levels (matches the eval
            # permutation design where order is fixed within a chain)
            cs = dict(cs, items=self.rng.sample(list(cs["items"]), len(cs["items"])))
        item_ids = [it["item_id"] for it in cs["items"]]
        fi, fj = cs["focal_pair"]
        spec_a = self.registry[cs["target_attribute"]]
        near_tied = []
        for a in range(len(cs["items"])):
            for b in range(a + 1, len(cs["items"])):
                va, vb = cs["items"][a].get(cs["target_attribute"]), cs["items"][b].get(cs["target_attribute"])
                if va is not None and vb is not None and abs(va - vb) < spec_a["min_pair_gap"]:
                    near_tied.append((a, b))
        ex = {"set_id": set_id, "family": fam, "attr": cs["target_attribute"],
              "attr_idx": ATTRS.index(cs["target_attribute"]),
              "focal": (item_ids.index(fi), item_ids.index(fj)),
              "near_tied_pairs": near_tied,
              "margin": self.margin(cs), "levels": []}
        for level in self.levels:
            spec = dict(levels[level])
            if self.paraphrase_aug and self.rng.random() < 0.5:
                # swap in the other train family's wording (T1<->T2) as augmentation
                other = "T2" if fam == "T1" else "T1"
                from src.intervention_generation.generate_interventions import (
                    ATTR_NOUN, ATTR_PHRASES, TEMPLATES)
                tmpl = TEMPLATES[level][other]
                spec["surface_form"] = tmpl.format(
                    attr=ATTR_PHRASES[ex["attr"]],
                    noun=ATTR_NOUN[ex["attr"]].capitalize() if tmpl.startswith("{noun}") else ATTR_NOUN[ex["attr"]])
            ids, mask, pos = self.encode_level(cs, spec)
            order, gains = self.oracle_order(cs, level)
            ex["levels"].append({
                "input_ids": ids, "attention_mask": mask, "marker_positions": pos,
                "oracle_rank_idx": [item_ids.index(x) for x in order],
                "gains": [gains[x] for x in item_ids], "level": level,
            })
        return ex
