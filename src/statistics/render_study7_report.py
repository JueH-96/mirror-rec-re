"""Render reports/study7_report.md from results/study7/stageA_selection.json,
results/study7_statistical_tests.json, results/study7/descriptive_tradeoff.json.
Mechanical number transfer only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABL_LABEL = {"S8c_ablg": "−强度表示 g(ℓ)→常数",
             "S8c_ablga": "−属性条件化（g 共享行）",
             "S8c_ablicr": "−L_ICR",
             "S8c_ablchain": "−L_chain",
             "S8c_ablresp": "−L_resp",
             "S8c_ablot": "−L_offtarget"}


def fmt_p(p):
    return f"{p:.4f}" if isinstance(p, float) else str(p)


def main():
    SEL = json.load(open(ROOT / "results/study7/stageA_selection.json"))
    ST = json.load(open(ROOT / "results/study7_statistical_tests.json"))
    DT = json.load(open(ROOT / "results/study7/descriptive_tradeoff.json"))
    L = []
    A = L.append
    A("# Study 7 报告：S8 组件消融 + 排列增强（H7）")
    A("")
    A("- 计划：experiments/study7_plan.md（2026-08-12 用户批准，检验口径 §5 "
      "批准时刻固化）。统计：results/study7_statistical_tests.json；"
      "筛选审计：results/study7/stageA_selection.json；"
      "权衡分析：results/study7/descriptive_tradeoff.json。")
    A("- **透明性声明（如实）**：" + ST["transparency"])
    A("- 覆盖范围：仅 Llama 家族（2026-08-03 裁定 d，记为局限）。")
    A("")

    # ---- 1. Stage A ---------------------------------------------------------
    A("## 1. Stage A 消融筛查（探索性，seed 1042，点估计，无显著性检验）")
    A("")
    ref = SEL["reference"]
    A(f"参照 S8c（完整，seed1042）：4c 未见转换合并违反率 {ref['rate_4c']:.4f}，"
      f"RR {ref['RR']:.3f}，NDCG@5 {ref['NDCG5']:.4f}（S0_llama RR "
      f"{ref['RR_S0_llama']:.3f}）。入选规则（§5 固化，机械执行）："
      "4c 升幅 ≥1pp 或 RR/NDCG 护栏触发。")
    A("")
    A("| 消融 | 4c 违反率 | Δ vs S8c | RR | NDCG@5 | 护栏触发 | 入选 Stage B |")
    A("|---|---|---|---|---|---|---|")
    for abl, v in SEL["ablations"].items():
        trips = []
        if v["RR_guardrail_trip"]:
            trips.append("RR")
        if v["NDCG_guardrail_trip"]:
            trips.append("NDCG")
        A(f"| {ABL_LABEL.get(abl, abl)} | {v['rate_4c']:.4f} "
          f"| {v['d4c_vs_S8c']:+.4f} | {v['RR']:.3f} | {v['NDCG5']:.4f} "
          f"| {'/'.join(trips) if trips else '无'} "
          f"| {'是' if v['selected'] else '否'} |")
    A("")
    A(f"入选：{SEL['selected'] if SEL['selected'] else '无'}。")
    A("")

    # ---- 2. H7-A ------------------------------------------------------------
    A("## 2. H7-A 组件必要性（预注册·确认性；入选消融 × 3 seeds）")
    A("")
    h7a = ST["h7a"]
    if not h7a["selected_ablations"]:
        A("- " + h7a["verdict"])
        A("- 机械含义：在 Stage A 的 1-seed 筛查精度内，没有单一组件的移除"
          "使 4c 未见转换违反率上升 ≥1pp——S8c 对该面的零违反不依赖任何"
          "单个被测组件；该观察为探索性（1 seed），不构成预注册结论。")
    else:
        A("| 组件 | 未见转换差 (pp) | 逐 seed | p_raw | p_holm | 效应门(≥1pp) "
          "| 判定 |")
        A("|---|---|---|---|---|---|---|")
        for abl in h7a["selected_ablations"]:
            v = h7a.get(abl)
            if not isinstance(v, dict):
                A(f"| {ABL_LABEL.get(abl, abl)} | missing | | | | | |")
                continue
            ps = "/".join(f"{x:+.4f}" for x in v["per_seed"].values())
            A(f"| {ABL_LABEL.get(abl, abl)} | {v['unseen_transition_diff_pp']:+.4f} "
              f"| {ps} | {fmt_p(v['p_raw'])} | {fmt_p(v.get('p_holm'))} "
              f"| {'过' if v['effect_pass'] else '未过'} "
              f"| {'组件必要' if v.get('component_necessary') else '不支持'} |")
        A("")
        A("护栏（单独如实报告，不并入判定）：")
        for abl in h7a["selected_ablations"]:
            v = h7a.get(abl)
            if isinstance(v, dict):
                g = v["guardrails"]
                A(f"- {ABL_LABEL.get(abl, abl)}：RR {g['RR_3seed']}（Δ vs S0 "
                  f"{g['RR_delta_vs_S0']}，collapse={g['responsiveness_collapse']}）；"
                  f"NDCG@5 {g['NDCG5_3seed']}（降幅 vs S8c "
                  f"{g['NDCG5_drop_vs_S8c']}，flag={g['ndcg_flag']}）")
    A("")

    # ---- 3. H7-P ------------------------------------------------------------
    A("## 3. H7-P 排列增强有效性（预注册·确认性重测；3 seeds × 5 排列）")
    A("")
    h7p = ST["h7p"]
    A("| 系统 | 顺序轴 DVR 非增强 | 增强后 | 差 (pp) | p_raw | p_holm "
      "| 效应门(≥2pp) | 分布内护栏 | 判定 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for sysname in ("S7", "S8"):
        v = h7p.get(sysname)
        if not isinstance(v, dict):
            A(f"| {sysname} | missing | | | | | | | |")
            continue
        g = v["guardrail_test_main"]
        A(f"| {sysname} | {v['order_DVR_nonaug']:.4f} | {v['order_DVR_perm']:.4f} "
          f"| {v['order_diff_pp']:+.4f} | {fmt_p(v['p_raw'])} "
          f"| {fmt_p(v.get('p_holm'))} | {'过' if v['effect_pass'] else '未过'} "
          f"| Δ {g['delta_pp']:+.4f}"
          f"{'（触发）' if g['in_distribution_cost_flag'] else '（未触发）'} "
          f"| {'支持' if v.get('supported') else '不支持'} |")
    A("")
    sec = h7p.get("secondary_noninferiority_vs_S2perm", {})
    A(f"次级（描述性，不并入判定）：S2perm 顺序轴 DVR "
      f"{sec.get('S2perm_order_DVR')}（原 S2：{sec.get('S2_order_DVR')}）。")
    for sysname in ("S7", "S8"):
        v = sec.get(sysname)
        if isinstance(v, dict):
            A(f"- {sysname}perm − S2perm = "
              f"{v['order_DVR_sysperm_minus_S2perm']:+.4f}（增强前 "
              f"{sysname} − S2 = {v['order_DVR_sys_minus_S2_original']:+.4f}）"
              f"——非劣{'恢复' if v['order_DVR_sysperm_minus_S2perm'] <= 0 else '未恢复（仍高于 S2perm）'}。")
    A("")

    # ---- 4. Sec 5b trade-off ------------------------------------------------
    A("## 4. 权衡分析：排列增强是否牺牲强度语义分布外能力（§5b，描述性，"
      "非预注册）")
    A("")
    A("### 4.1 T4 措辞分布外腿（链级 DVR 退化 = DVR_t4 − DVR_main）")
    A("")
    t4 = DT["t4_leg"]
    A("| 系统 | 逐 seed 退化 | 均值退化 | perm − 非增强 |")
    A("|---|---|---|---|")
    for sysname in ("S2", "S7", "S8"):
        for variant in (sysname, f"{sysname}perm"):
            v = t4[variant]
            ps = "/".join(
                (f"{r['degradation']:+.4f}" if isinstance(r, dict) else "缺")
                for r in v["per_seed"].values())
            delta = (t4.get(f"{sysname}_delta_perm_minus_nonaug")
                     if variant.endswith("perm") else "")
            A(f"| {variant} | {ps} | {v['mean_degradation']} "
              f"| {delta if delta is not None else '缺'} |")
    A("")
    A("### 4.2 4c 未见强度转换腿（c-变体实例化：levels {1,2,4} ± order_aug）")
    A("")
    c4 = DT["c4_leg"]
    A("| 系统 | 逐 seed 未见转换违反率 | 均值 | perm − 非增强 |")
    A("|---|---|---|---|")
    for sysname in ("S7c", "S8c"):
        for variant in (sysname, f"{sysname}perm"):
            v = c4[variant]
            ps = "/".join(
                (f"{r['unseen_transition_rate']:.4f}" if isinstance(r, dict) else "缺")
                for r in v["per_seed"].values())
            delta = (c4.get(f"{sysname}_delta_perm_minus_nonaug")
                     if variant.endswith("perm") else "")
            A(f"| {variant} | {ps} | {v['mean_unseen_rate']} "
              f"| {delta if delta is not None else '缺'} |")
    A("")
    d7 = c4.get("S7c_delta_perm_minus_nonaug")
    d8 = c4.get("S8c_delta_perm_minus_nonaug")
    t7 = t4.get("S7_delta_perm_minus_nonaug")
    t8 = t4.get("S8_delta_perm_minus_nonaug")
    A("要点（机械读数）：")
    if None not in (t7, t8):
        A(f"- T4 腿：排列增强使 T4 退化变化 S7 {t7:+.4f} / S8 {t8:+.4f}"
          "（正 = 增强后退化更大）。")
    if None not in (d7, d8):
        A(f"- 4c 腿：排列增强使未见转换违反率变化 S7c {d7:+.4f} / S8c "
          f"{d8:+.4f}（正 = 增强后更差）。")
    A("- 该节无显著性检验；逐 seed 数字并列呈现以显示 4c 面已知的 seed "
      "依赖性（study6 报告 §6 局限）。")
    A("")

    # ---- 5. conclusions -----------------------------------------------------
    A("## 5. 结论（机械汇总，逐条对应上表）")
    A("")
    if not h7a["selected_ablations"]:
        A("- **H7-A**：无组件入选 Stage B（全部单组件移除 4c 升幅 <1pp 且无"
          "护栏触发）；H7-A 空判定，如实报告——不得表述为'组件均不必要'的"
          "确证结论（Stage A 仅 1 seed）。")
    else:
        nec = [a for a in h7a["selected_ablations"]
               if isinstance(h7a.get(a), dict) and h7a[a].get("component_necessary")]
        A(f"- **H7-A**：入选 {len(h7a['selected_ablations'])} 项，判定'组件"
          f"必要'：{nec if nec else '无'}。")
    for sysname in ("S7", "S8"):
        v = h7p.get(sysname)
        if isinstance(v, dict):
            A(f"- **H7-P/{sysname}**：{'支持' if v.get('supported') else '不支持'}"
              f"（差 {v['order_diff_pp']:+.4f}，p_holm {fmt_p(v.get('p_holm'))}，"
              f"护栏{'触发' if v['guardrail_test_main']['in_distribution_cost_flag'] else '未触发'}）。")
    A("")

    A("## 6. 局限与透明性")
    A("")
    A("- H7 全部为确认性重测（见文首声明），非盲预注册。")
    A("- 仅 Llama；Stage A 为 1-seed 筛查；§4 权衡分析为描述性。")
    A("- S7perm/S8perm seed1042 checkpoint 复用自最小验证（同配置同 seed，"
      "重训为确定性重复；experiment_manifest.yaml study7.minval_reuse）。")
    A("")
    out = ROOT / "reports/study7_report.md"
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
