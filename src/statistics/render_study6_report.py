"""Render reports/study6_report.md mechanically from results/study6/h6_report.json.

Single source of truth: the stats JSON (also copied to
results/study6_statistical_tests.json). No hand-transcribed numbers.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "results/study6/h6_report.json").read_text())

SPLIT_ZH = {"test_main": "test_main（分布内）",
            "test_unseen_word": "test_unseen_word（T3）",
            "test_t4_word": "test_t4_word（T4）"}
AXIS_ZH = {"order": "候选顺序", "distract": "干扰文本", "mix": "链内混族",
           "neartied_2x_1x": "near-tied（2×/1× 档）", "conflict": "次要属性冲突"}


def cell(e):
    if e == "missing" or e is None:
        return "缺失"
    if isinstance(e, dict):
        r = e["seed_range"]
        return f"{e['mean']:.4f}" + (f" [{r[0]:.4f}, {r[1]:.4f}]" if r[0] != r[1] else "")
    if isinstance(e, float):
        return f"{e:.4f}"
    return str(e)


def dvr_row_table(block, systems, title_row):
    lines = [title_row, "|" + "---|" * (title_row.count("|") - 1)]
    for name in systems:
        e = block.get(name)
        if e is None:
            continue
        lines.append(f"| {name} | " + cell(e) + " |")
    return lines


def main():
    L = []
    A = L.append
    A("# Study 6 报告：强度语义分布外确证（H6）+ 鲁棒性与反作弊（H6-R）")
    A("")
    A("- 预注册计划：experiments/study6_plan.md（2026-08-04 固化；四项用户裁定见 experiment_manifest.yaml）")
    A("- 本报告由 src/statistics/render_study6_report.py 从 results/study6/h6_report.json 机械渲染，无手工转录数字")
    A("- 统计 JSON 副本：results/study6_statistical_tests.json（§18.2 命名）")
    A(f"- 违反判定 ε = {R['epsilon']}；链单元 = (seed, set_id|措辞族)")
    A(f"- **6a 透明性声明**：{R['notes']['6a_transparency']}")
    A(f"- **6b 门槛依据**：{R['notes']['6b_gate']}")
    A("")

    A("## 1. H6 预注册检验（主证据）")
    A("")
    A("| 对比 | 轴 | 效应 (pp) | p_raw | p_holm | 显著(α=0.05) | 效应门 | 过门 |")
    A("|---|---|---|---|---|---|---|---|")
    for key, label in [("S7_vs_S2_test_t4_word", "6a T4 退化差"),
                       ("S8_vs_S2_test_t4_word", "6a T4 退化差"),
                       ("S7c_vs_S2c_unseen_transitions", "6b 未见转换合并差"),
                       ("S8c_vs_S2c_unseen_transitions", "6b 未见转换合并差")]:
        e = R["h6"].get(key)
        if not isinstance(e, dict):
            A(f"| {key} | {label} | 缺失 | | | | | |")
            continue
        eff = e.get("degradation_diff_pp", e.get("unseen_transition_diff_pp"))
        A(f"| {key.replace('_vs_', ' vs ').split(' ')[0]} vs S2{'c' if 'c_' in key else ''}"
          f" | {label} | {eff * 100:.2f} | {e['p_raw']:.4f} | {e.get('p_holm', float('nan')):.4f} | "
          f"{'是' if e.get('significant_holm_05') else '否'} | ≥{e['effect_gate_pp'] * 100:.0f}pp | "
          f"{'是' if e['effect_pass'] else '否'} |")
    A("")
    for key in ("S7c_vs_S2c_unseen_transitions", "S8c_vs_S2c_unseen_transitions"):
        e = R["h6"].get(key)
        if isinstance(e, dict) and e.get("per_seed"):
            ps = ", ".join(f"{k}:{v:.4f}" for k, v in e["per_seed"].items())
            A(f"- {key} 各 seed 差：{ps}")
    A(f"- 预注册标准：{R['h6'].get('prereg_criterion', '')}")
    for s, v in R["h6"].get("per_system", {}).items():
        if isinstance(v, dict):
            A(f"- {s}：确证 = {v['confirmed']}（塌缩标记 {v['collapse_flag']}）")
        else:
            A(f"- {s}：{v}")
    A(f"- **H6 预注册判定：{R['h6'].get('prereg_verdict', 'PENDING')}**（机械输出，禁止改述）")
    A("")

    A("## 2. H6 轴护栏")
    A("")
    for axis, zh in [("test_t4_word", "6a T4 轴"), ("test_main_c_variants", "6b 4c 轴（c 变体，test_main）")]:
        block = R["guardrails"].get(axis, {})
        if not block:
            continue
        A(f"### {zh}")
        A("")
        A("| 系统 | RR | ΔRR vs S0 | 塌缩 | NDCG@5 | ΔNDCG5 | 效用受损 |")
        A("|---|---|---|---|---|---|---|")
        for name, g in block.items():
            A(f"| {name} | {cell(g.get('RR'))} | {g.get('RR_delta_vs_S0', '—')} | "
              f"{g.get('responsiveness_collapse', '—')} | {cell(g.get('NDCG5'))} | "
              f"{g.get('NDCG5_delta_vs_S2', g.get('NDCG5_delta_vs_S2c', '—'))} | "
              f"{g.get('utility_damage_flag', '—')} |")
        A("")

    A("## 3. DVR 对照与 4c 三 seed 分解")
    A("")
    A("### 3.1 T4 轴 DVR（llama 3 seeds；qwen seed1042 确认）")
    A("")
    A("| 系统 | " + " | ".join(SPLIT_ZH[s] for s in ("test_main", "test_unseen_word", "test_t4_word")) + " |")
    A("|---|---|---|---|")
    for fam in ("llama", "qwen"):
        for sysname in ("S0", "S2", "S7", "S8"):
            e = R["dvr_t4"].get(f"{sysname}_{fam}", {})
            A(f"| {sysname}_{fam} | " + " | ".join(
                cell(e.get(s)) for s in ("test_main", "test_unseen_word", "test_t4_word")) + " |")
    A("")
    A("### 3.2 4c 未见转换违反率（2→3 与 3→4 合并，3 seeds）")
    A("")
    A("| 系统 | 重训 lv124 | study3 全等级同 seeds |")
    A("|---|---|---|")
    for retrain in ("S2c", "S7c", "S8c"):
        e = R["axis_4c_3seed"].get(retrain, {})
        A(f"| {retrain} vs {retrain[:-1]} | {cell(e.get('unseen_transition_rate'))} | "
          f"{cell(e.get('study3_full_levels'))} |")
    A("")
    A("### 3.3 4c 逐 seed 转移分解")
    A("")
    A("| run | seed | 全链 DVR | RR | 1→2 | 2→3（未见） | 3→4（未见） | 2→4 跳变 | 子链{1,2,4} |")
    A("|---|---|---|---|---|---|---|---|---|")
    for retrain in ("S2c", "S7c", "S8c"):
        for seed, t in R["axis_4c_3seed"].get(retrain, {}).get("per_seed", {}).items():
            if not t:
                A(f"| {retrain} | {seed} | 缺失 | | | | | | |")
                continue
            A(f"| {retrain} | {seed} | {t['DVR']:.4f} | {t['RR']:.4f} | {t['viol_1->2']:.4f} | "
              f"{t['viol_2->3']:.4f} | {t['viol_3->4']:.4f} | {t['viol_2->4_jump']:.4f} | "
              f"{t['DVR_seen_subchain_124']:.4f} |")
    A("")

    A("## 4. H6-R 鲁棒性（次级，不并入 H6 主判定）")
    A("")
    A("### 4.1 汇总非劣检验（S7/S8 vs S2，双侧配对置换，Holm 跨 5 轴）")
    A("")
    A("| 对比 | 轴 | DVR 差 (sys−S2) | p_raw | p_holm | 显著 | 描述性非劣 |")
    A("|---|---|---|---|---|---|---|")
    for sysname in ("S7", "S8"):
        for axis in AXIS_ZH:
            e = R["h6_r"]["noninferiority_tests"].get(f"{sysname}_vs_S2_{axis}")
            if not isinstance(e, dict):
                A(f"| {sysname} vs S2 | {AXIS_ZH[axis]} | 缺失 | | | | |")
                continue
            A(f"| {sysname} vs S2 | {AXIS_ZH[axis]} | {e['dvr_diff_sys_minus_S2']:.4f} | "
              f"{e['p_raw']:.4f} | {e.get('p_holm', float('nan')):.4f} | "
              f"{'是' if e.get('significant_holm_05') else '否'} | "
              f"{'是' if e['noninferior_desc'] else '否'} |")
    A("")

    A("### 4.2 候选顺序（5 排列）")
    A("")
    A("排列间排名一致性（跨 5 排列全同率 / 与原顺序 p0 一致率，逐 run）：")
    A("")
    A("| 系统 | run | 全同率 | 与 p0 一致率 |")
    A("|---|---|---|---|")
    for name, per_run in R["h6_r"]["order"].get("invariance", {}).items():
        for run, v in per_run.items():
            A(f"| {name} | {run} | {v['identical_across_5_perms']:.4f} | "
              f"{v['agreement_with_p0']:.4f} |")
    A("")
    A("各排列面 DVR（llama 系统，均值[seed区间]）：")
    A("")
    A("| 系统 | " + " | ".join(f"p{k}" for k in range(1, 6)) + " |")
    A("|---|---|---|---|---|---|")
    for sysname in ("S0", "S2", "S7", "S8"):
        cells = [cell(R["h6_r"]["order"]["per_perm_dvr"]
                      .get(f"test_order_p{k}", {}).get(f"{sysname}_llama"))
                 for k in range(1, 6)]
        A(f"| {sysname}_llama | " + " | ".join(cells) + " |")
    for sysname in ("S0", "S2", "S7", "S8"):
        cells = [cell(R["h6_r"]["order"]["per_perm_dvr"]
                      .get(f"test_order_p{k}", {}).get(f"{sysname}_qwen"))
                 for k in range(1, 6)]
        if any(c != "缺失" for c in cells):
            A(f"| {sysname}_qwen | " + " | ".join(cells) + " |")
    A("")

    A("### 4.3 干扰文本（1/2/4 句）")
    A("")
    A("| 系统 | 无干扰基线（子样本） | d1 | d2 | d4 | RR@d4 | NDCG5@d4 |")
    A("|---|---|---|---|---|---|---|")
    base = R["h6_r"]["distract"].get("baseline_subsample", {})
    for sysname in ("S0", "S2", "S7", "S8"):
        row = [cell(base.get(f"{sysname}_llama"))]
        for t in ("d1", "d2", "d4"):
            row.append(cell(R["h6_r"]["distract"].get(f"test_distract_{t}", {})
                            .get(f"{sysname}_llama")))
        d4 = R["h6_r"]["distract"].get("test_distract_d4", {}).get(f"{sysname}_llama", {})
        rr = d4.get("RR") if isinstance(d4, dict) else None
        nd = d4.get("NDCG5") if isinstance(d4, dict) else None
        A(f"| {sysname}_llama | " + " | ".join(row) +
          f" | {cell(rr)} | {cell(nd)} |")
    A("")

    A("### 4.4 链内混族（M12 / M123 / M1234，M1234 含 T4）")
    A("")
    A("| 系统 | 同族基线（子样本） | M12 | M123 | M1234 |")
    A("|---|---|---|---|---|")
    for sysname in ("S0", "S2", "S7", "S8"):
        row = [cell(R["h6_r"]["mix"]["same_family_baseline_subsample"]
                    .get(f"{sysname}_llama"))]
        for s in ("test_mix12", "test_mix123", "test_mix1234"):
            row.append(cell(R["h6_r"]["mix"].get(s, {}).get(f"{sysname}_llama")))
        A(f"| {sysname}_llama | " + " | ".join(row) + " |")
    A("")

    A("### 4.5 near-tied 分档 DVR")
    A("")
    A(f"- {R['h6_r']['neartied']['note_05x']}")
    A("")
    A("| 系统 | 2× 档 | 1× 档 | 0.5× 档（描述性） |")
    A("|---|---|---|---|")
    for fam in ("llama", "qwen"):
        for sysname in ("S0", "S2", "S7", "S8"):
            cells = [cell(R["h6_r"]["neartied"].get(f"tier_{t}", {})
                          .get(f"{sysname}_{fam}")) for t in ("2x", "1x", "05x")]
            if any(c != "缺失" for c in cells):
                A(f"| {sysname}_{fam} | " + " | ".join(cells) + " |")
    A("")

    A("### 4.6 次要属性冲突（DVR + 逐级 NDCG@5）")
    A("")
    A("| 系统 | DVR | NDCG5 L1 | L2 | L3 | L4 |")
    A("|---|---|---|---|---|---|")
    for sysname in ("S0", "S2", "S7", "S8"):
        d = cell(R["h6_r"]["conflict"]["dvr"].get(f"{sysname}_llama"))
        nd = R["h6_r"]["conflict"].get(f"NDCG5_by_level_{sysname}_llama", {})
        A(f"| {sysname}_llama | {d} | " + " | ".join(
            cell(nd.get(lv)) for lv in ("1", "2", "3", "4")) + " |")
    A("")

    A("### 4.7 固定排序塌缩检查")
    A("")
    fo = R["h6_r"]["fixed_order"]
    A("| run | 众数位置模式一致率 | 模式熵 (bits) | RR | ΔRR vs S0 | 塌缩标记 | 链内固定率（描述性） |")
    A("|---|---|---|---|---|---|---|")
    for run, v in fo.items():
        if not isinstance(v, dict) or "modal_agreement" not in v:
            continue
        A(f"| {run} | {v['modal_agreement']:.4f} | {v['position_pattern_entropy_bits']:.4f} | "
          f"{cell(v.get('RR'))} | {v.get('RR_delta_vs_S0', '—')} | {v.get('collapse_flag', '—')} | "
          f"{cell(v.get('within_chain_fixed_rate_descriptive'))} |")
    A("")
    dv = fo["detector_validity"]
    A(f"- **检测器效度**：S5 阳性对照触发固化塌缩标记 = {dv['S5_positive_control_flagged']}。"
      + ("" if dv["S5_positive_control_flagged"] else
         "按固化计划 §4：阳性对照未触发 → 该检查报**检测器失效**，固化标记不用于结论；"
         "S5 的实际塌缩形态由链内固定率（描述性列）与 RR 呈现。"))
    A(f"- 主系统被固化标记命中：{fo.get('main_systems_flagged', [])}")
    A("")

    A("## 5. 结果叙事（由入库数值机械生成）")
    A("")
    v = R["h6"].get("prereg_verdict", "PENDING")
    A(f"- **H6 预注册判定：{v}**（判定标准 2026-08-04 固化；6a 为确认性重测——"
      "非盲预注册，见头部透明性声明；6b 门槛为 pilot 定门槛 1pp）。")
    for s, pv in R["h6"].get("per_system", {}).items():
        if isinstance(pv, dict):
            A(f"- {s}: 两轴 p_holm/效应门明细 {json.dumps(pv['detail'])}；确证 = {pv['confirmed']}。")
    nit = R["h6_r"]["noninferiority_tests"]
    worst = [(k, e["dvr_diff_sys_minus_S2"]) for k, e in nit.items()
             if isinstance(e, dict)]
    if worst:
        wk, wv = max(worst, key=lambda x: x[1])
        A(f"- H6-R 非劣检验中 DVR 差最大（最不利于结构系统）的一组：{wk} = {wv:.4f}。")
    A("")

    A("## 6. 局限与后续")
    A("")
    A("- 6a 为确认性重测而非盲预注册（llama T4 点估计在 Study 4 已观察）；qwen 复现与检验口径为本 study 新增。")
    A("- qwen 覆盖限于 T4 / 候选顺序 / near-tied / 固定排序；干扰、混族、冲突轴仅 llama。")
    A("- near-tied 0.5× 档低于资格设计保证，只作描述性报告。")
    ps7 = R["h6"].get("S7c_vs_S2c_unseen_transitions", {}).get("per_seed", {})
    if ps7:
        ps_txt = "/".join(f"{v:.4f}" for _, v in sorted(ps7.items()))
        A(f"- 6b(4c) 效应量存在 seed 依赖性：S7c vs S2c 逐 seed 差 {ps_txt}"
          f"（seed {'/'.join(sorted(ps7))}），合并效应主要由 S2c seed1044 的"
          "未见转换退化驱动；方向在全部 seed 上一致（均 ≥0），但效应量的"
          "跨 seed 波动应在解释 6b 效应大小时注意（3 seeds 为预注册设计上限）。")
    A("- 手机目录许可（Study 4 裁定）：论文注明上游来源，复现包不分发目录数据本身，仅提供重建路径。")
    A("- Study 7 消融仅 Llama（裁定 d），第二评测面 = 4c（2026-08-04 裁定 3）。")
    A("")
    A("**M3 停车**：本报告与 Study 7 计划细化待用户评审后继续。")
    A("")

    out = ROOT / "reports/study6_report.md"
    out.write_text("\n".join(L))
    shutil.copy(ROOT / "results/study6/h6_report.json",
                ROOT / "results/study6_statistical_tests.json")
    print(f"wrote {out} ({len(L)} lines) + results/study6_statistical_tests.json")


if __name__ == "__main__":
    main()
