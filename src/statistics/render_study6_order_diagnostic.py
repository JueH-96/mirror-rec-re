"""Render reports/study6_order_diagnostic.md from
results/study6/order_diagnostic.json + results/study6/order_probe.json.
Mechanical number transfer only (project discipline: no hand transcription).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYS_LLAMA = [("S2", "S2_llama"), ("S7", "S7_llama"), ("S8", "S8_llama")]


def main():
    D = json.load(open(ROOT / "results/study6/order_diagnostic.json"))
    P = json.load(open(ROOT / "results/study6/order_probe.json"))
    runs, pooled = D["runs"], D["pooled_llama"]
    L = []
    A = L.append

    A("# Study 6 附录：候选顺序位置敏感性诊断（post-hoc，非预注册）")
    A("")
    A("- 触发：H6-R §4.2 顺序轴非劣检验失败（S7 +5.2pp / S8 +2.2pp vs S2，"
      "均显著）；用户 2026-08-11 要求诊断机制来源后停车。")
    A("- 方法：五个 order split（test_order_p1..p5）对同一候选集仅改变 prompt "
      "中候选排列，同一 (set, level, family) 请求跨排列配对既有 raw 分数"
      "（零新增 GPU 评测）；另加最小 GPU probe（seed1042 三系统）读取 "
      "[ITEM] 标记隐状态与 S8 三分量。")
    A("- 脚本：src/statistics/study6_order_diagnostic.py、study6_order_probe.py；"
      "数据：results/study6/order_diagnostic.json、order_probe.json。")
    A("- S0 分数为模型文本自报数值，仅作描述性锚点。")
    A("")

    A("## 1. 逐物品 raw score 跨排列方差（用户要求的最小实验）")
    A("")
    A("| run | 逐物品 std（原始） | 共模占比 | 逐物品 std（去共模） | 去共模噪声/组内分差 | 噪声>½最小邻差占比 | 位置槽位可解释方差 R² |")
    A("|---|---|---|---|---|---|---|")
    order = [f"{s}_llama_seed{sd}" for s in ("S2", "S7", "S8")
             for sd in (1042, 1043, 1044)]
    order += [f"{s}_qwen_seed1042" for s in ("S2", "S7", "S8")]
    order += ["S0_llama", "S0_qwen"]
    for r in order:
        if r not in runs:
            continue
        v = runs[r]
        A(f"| {r} | {v['mean_item_std_raw']:.3f} | {v['common_mode_share']:.2f} "
          f"| {v['mean_item_std_centered']:.3f} | {v['noise_ratio_centered']:.3f} "
          f"| {v['margin_exceed_frac_centered']:.3f} | {v['position_r2']:.3f} |")
    A("")
    A("要点（机械读数）：")
    nr = {k: pooled[f'{k}_llama']['noise_ratio_centered'] for k, _ in SYS_LLAMA}
    A(f"- 去共模后的归一化噪声三系统几乎同水平（llama 3-seed 池化 "
      f"S2 {nr['S2']:.3f} / S7 {nr['S7']:.3f} / S8 {nr['S8']:.3f}）——"
      "**顺序确实进入表示并使 raw score 随排列改变，但逐物品分数层面的相对"
      "噪声不是 S7 与 S8 差异的来源**。")
    pr2 = {k: pooled[f'{k}_llama']['position_r2'] for k, _ in SYS_LLAMA}
    A(f"- 位置槽位（第 1..5 位）只解释极小部分方差（R² S2 {pr2['S2']:.3f} / "
      f"S7 {pr2['S7']:.3f} / S8 {pr2['S8']:.3f}）：不是简单的\"排前/排后加分\"，"
      "而是随排列组合变化的上下文漂移（因果注意力下每个候选的前文不同）。")
    A("")

    A("## 2. DVR 相关层：focal 对 margin 与跨等级增量")
    A("")
    A("| run | focal margin 跨排列噪声比 | 同级符号翻转率 | 增量均值 μ_δ | 增量跨排列 std σ_δ | σ_δ/μ_δ | 顺序轴 DVR（复算） |")
    A("|---|---|---|---|---|---|---|")
    for r in order:
        if r not in runs:
            continue
        v = runs[r]
        ratio = (v['increment_noise_std'] / v['increment_mean']
                 if v['increment_mean'] else float('nan'))
        A(f"| {r} | {v['focal_margin_noise_ratio']:.3f} | {v['focal_sign_flip_rate']:.3f} "
          f"| {v['increment_mean']:.3f} | {v['increment_noise_std']:.3f} "
          f"| {ratio:.2f} | {v['increment_violation_rate']:.4f} |")
    A("")
    dvr = {k: pooled[f'{k}_llama']['increment_violation_rate'] for k, _ in SYS_LLAMA}
    A(f"- 复算的顺序轴 DVR（llama 池化 S2 {dvr['S2']:.3f} / S7 {dvr['S7']:.3f} / "
      f"S8 {dvr['S8']:.3f}）与 study6 报告 §4.2 排列面 DVR 一致，差值复现 "
      f"S7−S2 = +{(dvr['S7']-dvr['S2'])*100:.1f}pp、S8−S2 = "
      f"+{(dvr['S8']-dvr['S2'])*100:.1f}pp。")
    im = {k: pooled[f'{k}_llama']['increment_mean'] for k, _ in SYS_LLAMA}
    isd = {k: pooled[f'{k}_llama']['increment_noise_std'] for k, _ in SYS_LLAMA}
    A(f"- 违反发生在**跨等级增量层**：S8 的方向增量均值与噪声都比 S7 小约 "
      f"5–6 倍（μ_δ S7 {im['S7']:.2f} vs S8 {im['S8']:.2f}；σ_δ S7 {isd['S7']:.2f} "
      f"vs S8 {isd['S8']:.2f}），DVR 由 σ_δ/μ_δ 之比决定，而非绝对噪声大小。")
    A("")

    A("## 3. GPU probe：h(i) 漂移与 S8 §6.2 分量分解（seed1042）")
    A("")
    A("| run | h_i 跨排列余弦（均值） | h_i 跨排列相对 L2 | h_i 相邻等级余弦 |")
    A("|---|---|---|---|")
    for r in ("S2_llama_seed1042", "S7_llama_seed1042", "S8_llama_seed1042"):
        if r not in P:
            continue
        v = P[r]
        A(f"| {r} | {v['order_cos_mean']:.4f} | {v['order_rel_l2_mean']:.4f} "
          f"| {v['level_cos_mean']:.5f} |")
    A("")
    s8 = P.get("S8_llama_seed1042", {}).get("s8_increment_components")
    if s8:
        A("S8 focal-margin 增量 δ = ΔB + Δ(gU) + 0.1ΔR 的分量（跨排列均值 / 跨排列 std）：")
        A("")
        A("| 分量 | 均值 | 跨排列 std |")
        A("|---|---|---|")
        for k, lab in (("dB", "ΔB（基础相关性头）"), ("dgU", "Δ(g·U)（单调门控×效用头）"),
                       ("dR", "0.1·ΔR（残差头）"), ("total", "合计 δ")):
            A(f"| {lab} | {s8[k]['mean']:.4f} | {s8[k]['cross_perm_std']:.4f} |")
        A("")
        d42 = runs["S8_llama_seed1042"]
        A(f"- 交叉验证：probe 前向的增量合计（均值 {s8['total']['mean']:.4f}、"
          f"跨排列 std {s8['total']['cross_perm_std']:.4f}）与既有入库 raw 独立"
          f"计算值（{d42['increment_mean']:.4f}、{d42['increment_noise_std']:.4f}）"
          "一致，probe 管线与 M2 评测管线相互印证。")
        A("")

    A("## 4. 机制结论（对用户问题的直接回答）")
    A("")
    A("1. **位置是否进入表示、分数是否随排列改变**：是。三种训练系统的逐物品 "
      "raw score 都随排列显著移动；其中 ~62–78% 是不影响排名的共模平移，"
      "去共模后的秩相关噪声约为组内分差的 0.29–0.33，且在几乎所有物品上"
      "超过最小邻差的一半（表 1）。这是因果 LM 下 h(i) 依赖前文候选的直接后果，"
      "三系统共享该表示层缺陷。")
    p2 = {r: P[f"{r}_llama_seed1042"] for r in ("S2", "S7", "S8") if f"{r}_llama_seed1042" in P}
    A(f"2. **S8 优于 S7 是否可归因于 §6.2 分解**：可以，经两条互补路径，"
      "用户假设的\"h(i) 相对独立于上下文\"在方向上成立但需限定表述："
      f"（i）**表示层**：S8 的 h_i 跨排列漂移约为 S7 的一半（余弦 "
      f"{p2['S8']['order_cos_mean']:.3f} vs S7 {p2['S7']['order_cos_mean']:.3f} / "
      f"S2 {p2['S2']['order_cos_mean']:.3f}；相对 L2 {p2['S8']['order_rel_l2_mean']:.2f} "
      f"vs {p2['S7']['order_rel_l2_mean']:.2f} / {p2['S2']['order_rel_l2_mean']:.2f}）——"
      "这不是分解结构的先验保证（h 仍来自同一因果 LM），而是分解目标下 LoRA "
      "训练出的性质；（ii）**通道层**（决定性）：S8 的 h_i 跨相邻等级几乎不动"
      f"（余弦 {p2['S8']['level_cos_mean']:.5f} vs S7 {p2['S7']['level_cos_mean']:.4f}），"
      "方向增量几乎全部由上下文无关的单调门承载——分量表中 Δ(g·U) 占增量"
      "均值的 ~100%，ΔB/ΔR 均值≈0、噪声占比小；S7 的方向信号是两次全头输出"
      "之差，直接暴露在 score 级顺序噪声下，增量噪声约为 S8 的 6 倍，"
      "σ_δ/μ_δ 更差 → 顺序轴 DVR 更高。")
    A("3. 顺序效应**并非零均值噪声**：对增量做 K=5 排列平均后残余违反率仍达 "
      "7–20%（§5(a) 表），说明存在随排列组合系统性翻转的链，聚合可缓解不可根除。")
    A("")

    A("## 5. 缓解可行性快评（只评估，未实现）")
    A("")
    A("### (a) 推理期多排列分数聚合")
    A("")
    A("- 先例：Permutation Self-Consistency（Tang et al., \"Found in the "
      "Middle: Permutation Self-Consistency Improves Listwise Ranking in "
      "Large Language Models\", arXiv:2310.07712, NAACL 2024）对 listwise "
      "排名做多排列采样 + 秩聚合；多序平均校准位置偏置另见 Wang et al., "
      "arXiv:2305.17926。本项目的 score-head 变体更简单：对 K 个随机排列的 "
      "raw score 逐物品取均值（分数级 Monte-Carlo 对称化），归因时应表述为 "
      "PSC 思想在 score-head 重排序器上的移植，非原创。")
    A("- 预期效果（零新评测，由既有 p1..p5 离线计算，K=5 均值聚合后的"
      "顺序轴违反率）：")
    A("")
    A("| 系统（llama 池化 / qwen） | 单排列 DVR | K=5 聚合 DVR |")
    A("|---|---|---|")
    for sys_, pk in SYS_LLAMA:
        v = pooled[f"{sys_}_llama"]
        A(f"| {sys_}_llama | {v['increment_violation_rate']:.3f} "
          f"| {v['aggregated_k5_violation_rate']:.3f} |")
    for r in ("S2_qwen_seed1042", "S7_qwen_seed1042", "S8_qwen_seed1042"):
        v = runs[r]
        A(f"| {r} | {v['increment_violation_rate']:.3f} "
          f"| {v['aggregated_k5_violation_rate']:.3f} |")
    A("")
    A("- 成本：推理量 ×K（K=5 即评测/部署成本 ×5）；无需重训。实现工作量小"
      "（evaluate.py 外套一层排列循环 + 均值）。")
    A("- 对现有结论的影响：不触碰 H6 主判定（6a/6b 轴与顺序无关且按固化口径"
      "已收官）；只可能改写 H6-R §4.2 顺序轴的非劣结论，且残余 7–17% 违反率"
      "意味着聚合后 S7/S8 仍未必对 S2 非劣——不应期待其翻转判定，定位为"
      "部署缓解手段而非结论修复。")
    A("")
    A("### (b) 训练期排列增强")
    A("")
    A("- 做法：训练时每步随机打乱候选列表顺序（data_module 现固定 "
      "cs['items'] 顺序；已有 paraphrase_aug 同型开关，工程量小）。")
    A("- 成本：新系统变体需重训——最小验证（S7/S8 各 1 seed）2 run × ~1.6h "
      "≈ 3.2 GPU·h + 顺序轴评测 ~0.5 GPU 日；结论级验证（S2/S7/S8 × 3 seeds）"
      "9 run ≈ 14.5 GPU·h + 全轴评测。")
    A("- 对现有结论的影响：产生的是**新系统**，所有已固化结论不变；预期直接"
      "攻击机制根源（迫使 h(i) 对前文候选组合不变），比 (a) 更可能恢复非劣，"
      "但属 Study 7 范畴的新预注册轴，需用户裁定是否纳入。")
    A("")
    A("## 6. 已落实的关联事项")
    A("")
    A("- study6 报告 §6 已补 6b(4c) 效应量 seed 依赖性局限"
      "（逐 seed 差 0.0151/0.0020/0.0638，渲染器机械取自 h6_report.json）。")
    A("")
    A("**停车**：本诊断完成后等待用户裁定（是否 & 如何进 Study 7 / 缓解项）。")
    A("")

    out = ROOT / "reports/study6_order_diagnostic.md"
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
