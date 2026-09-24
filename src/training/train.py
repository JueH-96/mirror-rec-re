"""Study 3 trainer (YAML-driven, seed-fixed, resumable).

Loss family (§6.3-§6.5), enabled per system config:
  L_rec        pairwise logistic over rule-oracle order (all systems)
  L_pairwise   extra weight on within-request focal pair (S4)
  L_consist    non-directional consistency: |Delta^{k+1} - Delta^k| (S5)
  L_ICR        adjacent-transition hinge with attribute margin (S7, S8)
  L_chain      all-pair transition hinge (S7, S8)
  L_resp       responsiveness hinge, material pairs only (S8)
  L_offtarget  off-target stability on near-tied pairs (S8)

Usage: python src/training/train.py --config configs/study3/<system>.yaml
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.data_module import ChainDataset  # noqa: E402
from src.training.scorer_model import RerankScorer  # noqa: E402


def trainable_state(model):
    """Only LoRA adapters, score heads, and monotone params — never the 16GB backbone."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_" in k or k.startswith(("head_base", "head_util", "head_resid",
                                             "level_emb", "g0", "eta"))}


def pairwise_logistic(scores, oracle_rank_idx):
    loss, n = 0.0, 0
    for a in range(len(oracle_rank_idx)):
        for b in range(a + 1, len(oracle_rank_idx)):
            hi, lo = oracle_rank_idx[a], oracle_rank_idx[b]
            loss = loss + torch.nn.functional.softplus(scores[lo] - scores[hi])
            n += 1
    return loss / n


def chain_losses(deltas, margin, cfg, levels=(1, 2, 3, 4)):
    """deltas: per-level focal Delta tensors, aligned with `levels`.

    With the default 4-level chain this reproduces the Study 3 losses exactly;
    l_chain scales its margin by the actual strength-level distance so that
    level-subset training (Study 4c, levels {1,2,4}) demands the same slope
    per level step as full-chain training.
    """
    n = len(deltas)
    out = {}
    if cfg.get("l_icr", 0):
        h = sum(torch.clamp(margin - (deltas[k + 1] - deltas[k]), min=0) for k in range(n - 1)) / (n - 1)
        out["icr"] = cfg["l_icr"] * h
    if cfg.get("l_chain", 0):
        terms = [torch.clamp(margin * (levels[s] - levels[r]) / 3 - (deltas[s] - deltas[r]), min=0)
                 for r in range(n) for s in range(r + 1, n)]
        out["chain"] = cfg["l_chain"] * sum(terms) / len(terms)
    if cfg.get("l_resp", 0):
        rho = cfg.get("rho_target", 0.05)
        h = sum(torch.clamp(rho - (deltas[k + 1] - deltas[k]), min=0) for k in range(n - 1)) / (n - 1)
        out["resp"] = cfg["l_resp"] * h
    if cfg.get("l_consist", 0):
        h = sum(torch.abs(deltas[k + 1] - deltas[k]) for k in range(n - 1)) / (n - 1)
        out["consist"] = cfg["l_consist"] * h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_dir = ROOT / "experiments" / cfg.get("exp_dir", "study3") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))
    log_path = run_dir / "train_log.jsonl"

    device = "cuda"
    model = RerankScorer(cfg["model_name"], mode=cfg.get("mode", "base"),
                         lora_r=cfg.get("lora_r", 16),
                         g_const=cfg.get("g_const", False),
                         g_shared=cfg.get("g_shared", False)).to(device)
    levels = tuple(cfg.get("levels", (1, 2, 3, 4)))
    ds = ChainDataset("train", model.tokenizer,
                      paraphrase_aug=cfg.get("paraphrase_aug", False), seed=seed,
                      levels=levels, order_aug=cfg.get("order_aug", False))
    val = ChainDataset("val", model.tokenizer, levels=levels)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.get("lr", 1e-4))
    max_steps = args.max_steps or cfg.get("max_steps", 3000)
    accum = cfg.get("grad_accum", 4)

    start_step = 0
    ckpt = run_dir / "last.pt"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            opt.load_state_dict(state["opt"])
            start_step = state["step"]
            print(f"resumed from step {start_step}")
        except Exception as e:  # noqa: BLE001 - corrupt checkpoint (e.g. ENOSPC truncation)
            bad = ckpt.with_suffix(".pt.corrupt")
            ckpt.rename(bad)
            print(f"WARN: checkpoint unreadable ({e}); moved to {bad.name}, training from scratch")

    order = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(order)
    t0 = time.time()
    model.train()
    for step in range(start_step, max_steps):
        opt.zero_grad()
        acc = {}
        for g in range(accum):
            ex = ds.get(order[(step * accum + g) % len(order)])
            fi, fj = ex["focal"]
            deltas, all_scores, l_rec = [], [], 0.0
            for lv in ex["levels"]:
                s = model(lv["input_ids"].to(device), lv["attention_mask"].to(device),
                          lv["marker_positions"].to(device), ex["attr_idx"], lv["level"])
                l_rec = l_rec + pairwise_logistic(s, lv["oracle_rank_idx"])
                if cfg.get("l_pairwise", 0):
                    oi = lv["oracle_rank_idx"].index(fi) < lv["oracle_rank_idx"].index(fj)
                    d = s[fi] - s[fj] if oi else s[fj] - s[fi]
                    l_rec = l_rec + cfg["l_pairwise"] * torch.nn.functional.softplus(-d)
                deltas.append(s[fi] - s[fj])
                all_scores.append(s)
            n_lv = len(ex["levels"])
            loss = cfg.get("l_rec", 1.0) * l_rec / n_lv
            if cfg.get("l_offtarget", 0) and ex["near_tied_pairs"]:
                ot = 0.0
                for (a, b) in ex["near_tied_pairs"]:
                    for k in range(n_lv - 1):
                        d1 = all_scores[k][a] - all_scores[k][b]
                        d2 = all_scores[k + 1][a] - all_scores[k + 1][b]
                        ot = ot + torch.abs(d2 - d1)
                ot = ot / ((n_lv - 1) * len(ex["near_tied_pairs"]))
                loss = loss + cfg["l_offtarget"] * ot
                acc["offtarget"] = acc.get("offtarget", 0) + float(ot)
            for name, v in chain_losses(deltas, ex["margin"], cfg, levels).items():
                loss = loss + v
                acc[name] = acc.get(name, 0) + float(v)
            acc["rec"] = acc.get("rec", 0) + float(l_rec) / n_lv
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "elapsed_s": round(time.time() - t0, 1),
                   **{k: round(v / accum / 25, 5) for k, v in acc.items()}}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            acc = {}
        if (step + 1) % cfg.get("ckpt_every", 250) == 0 or step + 1 == max_steps:
            torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                        "step": step + 1}, ckpt)
    torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                "step": max_steps}, run_dir / "final.pt")
    ckpt.unlink(missing_ok=True)  # keep only final.pt; disk budget (fat-ckpt incident)
    print(f"done: {cfg['run_name']} at step {max_steps}")


if __name__ == "__main__":
    main()
