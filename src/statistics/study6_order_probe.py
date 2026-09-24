"""Study 6 post-hoc GPU probe (NOT preregistered): attribute the order
sensitivity mechanism at the representation/head level.

For S2/S7/S8_llama_seed1042 on the five order splits it measures, per
matched item across permutations:
  - order-drift of the marker hidden state h_i: mean pairwise cosine
    similarity across the 5 permutations (same set/family/level/item), and
    relative L2 dispersion ||h_p - h_bar|| / ||h_bar||;
  - level-drift of h_i within a permutation (cos between adjacent levels);
and for S8 (mirror mode) the focal-margin transition increment decomposed
into its Sec. 6.2 components:
  delta = dB + d(gU) + 0.1*dR, where B_l = b_l(i)-b_l(j), U_l = u_l(i)-u_l(j),
  R_l = r_l(i)-r_l(j), gU_l = g(l)*U_l;
reporting cross-permutation mean/std of each component per transition.

Output: results/study6/order_probe.json.
Runs on GPU 1 only (CUDA_VISIBLE_DEVICES set by launcher).
"""
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.data_module import ChainDataset  # noqa: E402
from src.training.scorer_model import RerankScorer  # noqa: E402

PERMS = [f"test_order_p{k}" for k in range(1, 6)]
RUNS = ["S2_llama_seed1042", "S7_llama_seed1042", "S8_llama_seed1042"]


def pstd(t):
    return float(torch.std(t, unbiased=False))


@torch.no_grad()
def probe_run(run):
    cfg = yaml.safe_load(open(ROOT / f"configs/study3/{run}.yaml"))
    device = "cuda"
    model = RerankScorer(cfg["model_name"], mode=cfg.get("mode", "base"),
                         lora_r=cfg.get("lora_r", 16)).to(device)
    state = torch.load(ROOT / f"experiments/study3/{run}/final.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    mirror = cfg.get("mode") == "mirror"

    dss = [ChainDataset(s, model.tokenizer) for s in PERMS]
    keymaps = [{(ds.chains[i][0]): i for i in range(len(ds))} for ds in dss]
    common = sorted(set.intersection(*(set(m) for m in keymaps)))

    order_cos, order_rel_l2, level_cos = [], [], []
    s8_rows = []  # per (set,fam,transition): component means/stds across perms
    score_check = []  # (probe_score, disk unavailable here) sanity vs raw later
    for n, key in enumerate(common):
        set_id, fam = key
        # per perm: level -> dict(item_id -> h, plus components)
        per_perm = []
        for ds, km in zip(dss, keymaps):
            ex = ds.get(km[key])
            cs = ds.sets[set_id]
            item_ids = [it["item_id"] for it in cs["items"]]
            levels = {}
            for lv in ex["levels"]:
                out = model.backbone(
                    input_ids=lv["input_ids"].unsqueeze(0).to(device),
                    attention_mask=lv["attention_mask"].unsqueeze(0).to(device),
                    output_hidden_states=True)
                hi = out.hidden_states[-1][0][lv["marker_positions"].to(device)]
                rec = {"h": hi.float().cpu(), "items": item_ids}
                if mirror:
                    rec["b"] = model.head_base(hi).squeeze(-1).float().cpu()
                    rec["u"] = model.head_util(hi).squeeze(-1).float().cpu()
                    rec["r"] = model.head_resid(hi).squeeze(-1).float().cpu()
                    rec["g"] = float(model.g_value(ex["attr_idx"], lv["level"]))
                else:
                    rec["s"] = model.head_base(hi).squeeze(-1).float().cpu()
                levels[lv["level"]] = rec
            per_perm.append((ex, levels))
        # ---- h order-drift and level-drift
        lvls = sorted(per_perm[0][1])
        items0 = per_perm[0][1][lvls[0]]["items"]
        for lv in lvls:
            for it in items0:
                hs = []
                for ex, levels in per_perm:
                    idx = levels[lv]["items"].index(it)
                    hs.append(levels[lv]["h"][idx])
                H = torch.stack(hs)  # (5, hid)
                Hn = torch.nn.functional.normalize(H, dim=-1)
                cosm = (Hn @ Hn.T)
                iu = torch.triu_indices(len(hs), len(hs), offset=1)
                order_cos.append(float(cosm[iu[0], iu[1]].mean()))
                hbar = H.mean(0)
                order_rel_l2.append(float(
                    (H - hbar).norm(dim=-1).mean() / (hbar.norm() + 1e-8)))
        for ex, levels in per_perm:
            for a, b in zip(lvls[:-1], lvls[1:]):
                for it in items0:
                    ia = levels[a]["items"].index(it)
                    ib = levels[b]["items"].index(it)
                    level_cos.append(float(torch.nn.functional.cosine_similarity(
                        levels[a]["h"][ia], levels[b]["h"][ib], dim=0)))
        # ---- S8 component decomposition of focal-margin increments
        if mirror:
            cs0 = dss[0].sets[set_id]
            fi, fj = cs0["focal_pair"]
            for a, b in zip(lvls[:-1], lvls[1:]):
                comp = {"dB": [], "dgU": [], "dR": [], "total": []}
                for ex, levels in per_perm:
                    ra, rb = levels[a], levels[b]
                    ii_a, ij_a = ra["items"].index(fi), ra["items"].index(fj)
                    ii_b, ij_b = rb["items"].index(fi), rb["items"].index(fj)
                    Ba = float(ra["b"][ii_a] - ra["b"][ij_a])
                    Bb = float(rb["b"][ii_b] - rb["b"][ij_b])
                    Ua = float(ra["u"][ii_a] - ra["u"][ij_a])
                    Ub = float(rb["u"][ii_b] - rb["u"][ij_b])
                    Ra = float(ra["r"][ii_a] - ra["r"][ij_a])
                    Rb = float(rb["r"][ii_b] - rb["r"][ij_b])
                    dB = Bb - Ba
                    dgU = rb["g"] * Ub - ra["g"] * Ua
                    dR = 0.1 * (Rb - Ra)
                    comp["dB"].append(dB)
                    comp["dgU"].append(dgU)
                    comp["dR"].append(dR)
                    comp["total"].append(dB + dgU + dR)
                row = {"set_id": set_id, "family": fam, "transition": f"{a}->{b}"}
                for k, v in comp.items():
                    t = torch.tensor(v)
                    row[f"{k}_mean"] = float(t.mean())
                    row[f"{k}_std"] = pstd(t)
                s8_rows.append(row)
        if (n + 1) % 20 == 0:
            print(f"{run}: {n + 1}/{len(common)} chains", flush=True)

    res = {
        "n_chains": len(common),
        "order_cos_mean": sum(order_cos) / len(order_cos),
        "order_cos_p10": sorted(order_cos)[int(len(order_cos) * 0.1)],
        "order_rel_l2_mean": sum(order_rel_l2) / len(order_rel_l2),
        "level_cos_mean": sum(level_cos) / len(level_cos),
    }
    if mirror:
        agg = {}
        for k in ("dB", "dgU", "dR", "total"):
            agg[k] = {
                "mean": sum(r[f"{k}_mean"] for r in s8_rows) / len(s8_rows),
                "cross_perm_std": sum(r[f"{k}_std"] for r in s8_rows) / len(s8_rows),
            }
        res["s8_increment_components"] = agg
        res["s8_component_rows"] = s8_rows
    del model
    torch.cuda.empty_cache()
    return res


def main():
    out_path = ROOT / "results/study6/order_probe.json"
    out = {"_note": "post-hoc GPU probe, not preregistered"}
    if out_path.exists():
        out.update(json.load(open(out_path)))
    for run in RUNS:
        if run in out:
            print(f"[skip] {run} already probed")
            continue
        print(f"=== probing {run}", flush=True)
        out[run] = probe_run(run)
        json.dump(out, open(out_path, "w"), indent=1)
        print(json.dumps({k: v for k, v in out[run].items()
                          if k != "s8_component_rows"}, indent=1), flush=True)
    print(f"PROBE_DONE -> {out_path}")


if __name__ == "__main__":
    main()
