"""Study 7 permutation-augmentation minimal-validation memo (EXPLORATORY;
user-authorized 2026-08-11; screening only — results do not enter any
conclusion; purpose is to judge whether the 9-run Stage B is worth running).

Compares S7perm/S8perm_llama_seed1042 (order_aug training) against their
non-augmented seed-1042 counterparts on:
  - order-axis increment DVR across the 5 permutation splits (same machinery
    as the study6 order diagnostic, applied to results/study7/raw);
  - test_main in-distribution DVR (single-'perm' degenerate case);
  - cross-permutation score noise / increment noise.

Output: results/study7/minval_memo.json + reports/study7_minval_memo.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.statistics.study6_order_diagnostic import (  # noqa: E402
    analyze, load_focal_pairs, load_positions, load_run, pstd)

RAW7 = ROOT / "results/study7/raw"
PERM_RUNS = ["S7perm_llama_seed1042", "S8perm_llama_seed1042"]
BASELINE = {"S7perm_llama_seed1042": "S7_llama_seed1042",
            "S8perm_llama_seed1042": "S8_llama_seed1042"}


def test_main_dvr(run, raw_root, focal_pairs, eps=0.01):
    """Plain focal-increment DVR on the single test_main rendering."""
    data = load_run(run, raw_root=raw_root, splits=["test_main"])
    if data is None:
        return None
    m = data["test_main"]
    margins = {}
    for (set_id, level, fam), scores in m.items():
        fi, fj = focal_pairs.get(set_id, (None, None))
        if fi in scores and fj in scores:
            margins[(set_id, fam)] = margins.get((set_id, fam), {})
            margins[(set_id, fam)][level] = scores[fi] - scores[fj]
    viol = n = 0
    for ch in margins.values():
        lv = sorted(ch)
        for a, b in zip(lv[:-1], lv[1:]):
            viol += (ch[b] - ch[a]) < -eps
            n += 1
    return viol / n if n else None


def main():
    positions = load_positions()
    focal_pairs = load_focal_pairs()
    base = json.load(open(ROOT / "results/study6/order_diagnostic.json"))["runs"]
    out = {"_note": "exploratory minval (order_aug screening); not evidence",
           "runs": {}}
    for run in PERM_RUNS:
        res = analyze(run, positions, focal_pairs, raw_root=RAW7)
        if res is None:
            print(f"MEMO_INCOMPLETE: {run} order raws missing")
            sys.exit(1)
        res["test_main_dvr"] = test_main_dvr(run, RAW7, focal_pairs)
        out["runs"][run] = res

    L = ["# Study 7 排列增强最小验证备忘（探索性，2026-08-11 授权；不作为证据）",
         "",
         "- 目的：判断是否值得启动 Stage B（3-seed 9-run 预注册验证）。",
         "- 训练：S7perm/S8perm_llama_seed1042 = study3 同配置 + order_aug"
         "（每次访问随机打乱候选列表顺序，链内各级共享该顺序）。",
         "- 对照数字取自 results/study6/order_diagnostic.json（非增强 seed1042）。",
         "",
         "| 指标 | S7 (原) | S7perm | S8 (原) | S8perm |",
         "|---|---|---|---|---|"]
    rows = [
        ("顺序轴增量 DVR", "increment_violation_rate", "{:.4f}"),
        ("K=5 聚合后 DVR", "aggregated_k5_violation_rate", "{:.4f}"),
        ("逐物品去共模噪声/组内分差", "noise_ratio_centered", "{:.3f}"),
        ("增量均值 μ_δ", "increment_mean", "{:.3f}"),
        ("增量跨排列 std σ_δ", "increment_noise_std", "{:.3f}"),
        ("focal 同级符号翻转率", "focal_sign_flip_rate", "{:.4f}"),
    ]
    b7, b8 = base["S7_llama_seed1042"], base["S8_llama_seed1042"]
    p7 = out["runs"]["S7perm_llama_seed1042"]
    p8 = out["runs"]["S8perm_llama_seed1042"]
    for label, key, fmt in rows:
        L.append(f"| {label} | {fmt.format(b7[key])} | {fmt.format(p7[key])} "
                 f"| {fmt.format(b8[key])} | {fmt.format(p8[key])} |")
    tm7, tm8 = p7["test_main_dvr"], p8["test_main_dvr"]
    L += ["",
          f"- test_main 分布内 DVR（护栏观察，限 order 公共子样本 100 组，与"
          f" study6 §4.3 无扰动基线同口径）：S7perm {tm7:.4f}、S8perm {tm8:.4f}"
          "（原 S7/S8 seed1042 均为 0）。",
          "",
          "判断口径（写死于本备忘，非预注册检验）：顺序轴增量 DVR 相对非增强"
          "对应系统下降 ≥5pp 且 test_main DVR ≤0.5pp 视为'值得上 Stage B'。"]
    d7 = b7["increment_violation_rate"] - p7["increment_violation_rate"]
    d8 = b8["increment_violation_rate"] - p8["increment_violation_rate"]
    ok7 = d7 >= 0.05 and (tm7 is not None and tm7 <= 0.005)
    ok8 = d8 >= 0.05 and (tm8 is not None and tm8 <= 0.005)
    L += [f"- S7perm：顺序轴改善 {d7*100:.1f}pp，护栏 {'通过' if tm7 <= 0.005 else '未过'}"
          f" → {'建议进入 Stage B' if ok7 else '未达筛选线'}。",
          f"- S8perm：顺序轴改善 {d8*100:.1f}pp，护栏 {'通过' if tm8 <= 0.005 else '未过'}"
          f" → {'建议进入 Stage B' if ok8 else '未达筛选线'}。",
          "",
          "该建议仅供用户审 study7_plan 时参考；Stage B 是否执行由用户裁定。"]
    out["screening"] = {"S7perm_improvement_pp": d7 * 100, "S8perm_improvement_pp": d8 * 100,
                        "S7perm_recommend_stageB": ok7, "S8perm_recommend_stageB": ok8}

    json.dump(out, open(ROOT / "results/study7/minval_memo.json", "w"), indent=1)
    (ROOT / "reports/study7_minval_memo.md").write_text("\n".join(L) + "\n")
    print("memo written; screening:", json.dumps(out["screening"]))


if __name__ == "__main__":
    main()
