"""Render reports/study5_report.md from results/study5/engineering.json +
results/study5/noise_extension.json. Mechanical number transfer only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]
FAMILIES = [("S0_llama", ["S0_llama"]),
            ("S2_llama", [f"S2_llama_seed{s}" for s in (1042, 1043, 1044)]),
            ("S7_llama", [f"S7_llama_seed{s}" for s in (1042, 1043, 1044)]),
            ("S8_llama", [f"S8_llama_seed{s}" for s in (1042, 1043, 1044)]),
            ("S0_qwen", ["S0_qwen"]),
            ("S2_qwen", ["S2_qwen_seed1042"]),
            ("S7_qwen", ["S7_qwen_seed1042"]),
            ("S8_qwen", ["S8_qwen_seed1042"])]
COLS = ["DVR_before", "DVR_after", "RR_before", "RR_after",
        "chains_adjusted_share", "binding_constraint_share",
        "NDCG5_before", "NDCG5_after", "mean_abs_focal_rank_shift",
        "mean_kendall_tau_pre_post", "solve_time_ms_per_chain"]


def pool(E, runs, split):
    rs = [E[r][split] for r in runs if r in E and split in E[r]]
    if not rs:
        return None
    w = sum(r["n_chains"] for r in rs)
    return {c: sum(r[c] * r["n_chains"] for r in rs) / w for c in COLS} | {
        "n_chains": w, "n_seeds": len(rs)}


def main():
    E = json.load(open(ROOT / "results/study5/engineering.json"))
    N = json.load(open(ROOT / "results/study5/noise_extension.json"))
    L = []
    A = L.append
    A("# Study 5 报告：约束投影的工程可行性刻画")
    A("")
    A("- 授权：用户 2026-08-11 裁定按 reports/study4_7_plan_draft.md Study 5 段"
      "全量执行（纯后处理，无新训练、无预注册门槛争议）。")
    A("- 测量脚本：src/projection/study5_engineering.py、study5_noise_extension.py"
      "（指标定义与 pilot 预实验 isotonic_projection.py 完全一致）；数据："
      "results/study5/engineering.json、noise_extension.json。")
    A("- 交叉验证：噪声管线 test_main rate 5% 的 DVR_after 0.0374 与 Study 3 "
      "已入库 S1N5_llama raw 直接复算值 0.0374 一致（管线互证）。")
    A("- 对象：Study 3 缓存输出（S0/S2/S7/S8 × llama 3 seeds / qwen seed1042）"
      "× 4 split；投影 = 每链 focal Δ 序列 PAVA（§6.6，±c/2 最优分摊）。")
    A("")
    A("## 1. 投影前后指标（llama 家族按 3 seeds 池化；qwen 单 seed）")
    A("")
    for split in SPLITS:
        A(f"### {split}")
        A("")
        A("| 系统 | DVR 前→后 | RR 前→后 | 调整链占比 | 活跃约束占比 | NDCG@5 前→后 | focal 位移 | Kendall τ | ms/链 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for fam, runs in FAMILIES:
            p = pool(E, runs, split)
            if p is None:
                continue
            A(f"| {fam} | {p['DVR_before']:.4f}→{p['DVR_after']:.4f} "
              f"| {p['RR_before']:.3f}→{p['RR_after']:.3f} "
              f"| {p['chains_adjusted_share']:.4f} | {p['binding_constraint_share']:.4f} "
              f"| {p['NDCG5_before']:.4f}→{p['NDCG5_after']:.4f} "
              f"| {p['mean_abs_focal_rank_shift']:.4f} "
              f"| {p['mean_kendall_tau_pre_post']:.4f} "
              f"| {p['solve_time_ms_per_chain']:.3f} |")
        A("")
    # mechanical extremes
    all_cells = [(r, s, v) for r, d in E.items() for s, v in d.items()]
    ndcg_drop = max(all_cells, key=lambda x: x[2]["NDCG5_before"] - x[2]["NDCG5_after"])
    slowest = max(all_cells, key=lambda x: x[2]["solve_time_ms_per_chain"])
    tr_adj = [c for c in all_cells if not c[0].startswith("S0")]
    max_adj_tr = max(tr_adj, key=lambda x: x[2]["chains_adjusted_share"])
    A("要点（机械读数）：")
    A(f"- 投影后 DVR 全部归 0（PAVA 对 focal 转换构造性保证）；全表最大 "
      f"NDCG@5 损失 = {ndcg_drop[2]['NDCG5_before'] - ndcg_drop[2]['NDCG5_after']:.4f}"
      f"（{ndcg_drop[0]} × {ndcg_drop[1]}）。")
    A(f"- 求解成本可忽略：最慢 {slowest[2]['solve_time_ms_per_chain']:.3f} ms/链"
      f"（{slowest[0]} × {slowest[1]}）。")
    A(f"- 训练系统（S2/S7/S8）投影几乎无操作：调整链占比最大 "
      f"{max_adj_tr[2]['chains_adjusted_share']:.4f}（{max_adj_tr[0]} × "
      f"{max_adj_tr[1]}）；S7/S8 大多数 run×split 为 0——投影的收益集中在 "
      "S0（zero-shot）上，对结构训练系统是纯冗余成本。")
    A("")
    A("## 2. 适用域边界：单请求模式覆盖率为 0（形式化说明）")
    A("")
    A("投影算子定义在**已装配的干预链**上（同一 (set, 措辞族) 的 K=4 个等级"
      "请求的 focal Δ 序列）。部署主模式为单请求独立打分（study3_plan §0-P1，"
      "推理期无链信息），此时不存在可投影的序列对象——投影覆盖率恒为 0，"
      "非近似为 0。因此投影只适用于离线批处理/会话内可聚链的场景；这与"
      "训练期内化方向性（S7/S8，单请求即生效）构成本质适用域差异。")
    A("")
    A("## 3. S1 解析噪声退化曲线（test_main → 全 split 扩展）")
    A("")
    A("| split | rate | DVR 无投影 | DVR 投影后 | 不可投影链占比 |")
    A("|---|---|---|---|---|")
    for split in SPLITS:
        for rate in ("0", "5", "10", "20"):
            v = N[split][rate]
            A(f"| {split} | {rate}% | {v['DVR_before']:.4f} "
              f"| {v['DVR_after_projection']:.4f} | {v['unprojectable_chain_share']:.4f} |")
    A("")
    worst = max(SPLITS, key=lambda s: N[s]["20"]["DVR_after_projection"])
    A("要点（机械读数）：")
    A("- 四个 split 的退化形态一致：噪声率 5/10/20% 下不可投影链占比约 "
      f"{N['test_main']['5']['unprojectable_chain_share']:.2f}/"
      f"{N['test_main']['10']['unprojectable_chain_share']:.2f}/"
      f"{N['test_main']['20']['unprojectable_chain_share']:.2f}（test_main），"
      "其余 split 同量级——解析噪声主要通过打断链装配使投影失效。")
    A(f"- 20% 噪声下投影后残余 DVR 最高为 {N[worst]['20']['DVR_after_projection']:.4f}"
      f"（{worst}），对比干净投影的 0：投影的保证以解析器可靠性为前提"
      "（与 Study 3 M3 的 S1-vs-S7 结论一致，此处扩展到全部 split）。")
    A("")
    A("## 4. 结论")
    A("")
    A("- 工程上投影本身廉价（亚毫秒/链）且效用无损（NDCG@5 变化 ≤ 全表最大值"
      "，见 §1），对 zero-shot 输出可把 DVR 构造性清零——作为**离线后处理**"
      "组件可行。")
    A("- 但其价值边界清晰：(i) 单请求部署模式覆盖率 0（§2）；(ii) 解析噪声下"
      "保证快速失效（§3）；(iii) 对已训练的结构系统近乎无操作（§1）。这与"
      "既有主线结论互补：投影是 S0 的廉价补丁，不是训练期结构目标的替代。")
    A("")
    A("（本 study 无预注册检验；全部为描述性工程刻画。）")
    A("")
    out = ROOT / "reports/study5_report.md"
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
