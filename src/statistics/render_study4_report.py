"""Render reports/study4_report.md mechanically from results/study4/h4_report.json.

Single source of truth: the stats JSON (which is also copied to
results/study4_statistical_tests.json). No hand-transcribed numbers.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "results/study4/h4_report.json").read_text())
CAL = json.loads((ROOT / "results/study4/t4_calibration.json").read_text())
META = json.loads((ROOT / "data_processed/phone_catalog_meta.json").read_text())

SPLIT_ZH = {"test_main": "test_main（分布内）",
            "test_unseen_word": "test_unseen_word（T3 未见措辞）",
            "test_unseen_item": "test_unseen_item（未见物品）",
            "test_domain_phone": "test_domain_phone（第二领域·手机）",
            "test_t4_word": "test_t4_word（T4 探索）"}
AXES = ["test_unseen_word", "test_unseen_item", "test_domain_phone"]


def cell(e):
    if e == "missing" or e is None:
        return "缺失"
    if isinstance(e, dict):
        r = e["seed_range"]
        return f"{e['mean']:.4f}" + (f" [{r[0]:.4f}, {r[1]:.4f}]" if r[0] != r[1] else "")
    return str(e)


def dvr_table(family, systems, splits):
    lines = ["| 系统 | " + " | ".join(SPLIT_ZH[s] for s in splits) + " |",
             "|" + "---|" * (len(splits) + 1)]
    for sysname in systems:
        e = R["dvr_table"].get(f"{sysname}_{family}")
        if not e:
            continue
        lines.append(f"| {sysname} | " +
                     " | ".join(cell(e.get(s, "—")) for s in splits) + " |")
    return "\n".join(lines)


def main():
    L = []
    A = L.append
    A("# Study 4 报告：结构单调性是否提升泛化（H4）")
    A("")
    A("- 预注册计划：experiments/study4_plan.md（2026-08-03 固化；四项用户裁定见 experiment_manifest.yaml）")
    A("- 本报告由 src/statistics/render_study4_report.py 从 results/study4/h4_report.json 机械渲染，无手工转录数字")
    A("- 统计 JSON 副本：results/study4_statistical_tests.json（§18.2 命名）")
    A(f"- 违反判定 ε = {R['epsilon']}；链单元 = (seed, set_id|措辞族)；"
      "test_main / test_unseen_word 数值来自 Study 3 已入库 metrics")
    A(f"- 第二领域目录：{META['rows']} 行真实手机数据（{META['upstream'].split(';')[0]}），"
      f"资格门 {'通过' if META['qualification_passed'] else '未通过'}；许可状态见 manifest（待用户复核）")
    A("- S1/S9 投影在本 study 使用生成器真值解析（oracle 形态）；噪声解析形态已在 Study 3 M3 刻画")
    A("")

    A("## 1. DVR 对照表（链级方向违反率，均值 [跨 seed 区间]）")
    A("")
    A("### 1.1 Llama（S2/S7/S8 三 seed；S0/S1 单 run；S9 三 seed）")
    A("")
    A(dvr_table("llama", ["S0", "S1", "S2", "S7", "S8", "S9"],
                ["test_main"] + AXES))
    A("")
    A("### 1.2 Qwen（确认组，seed 1042）")
    A("")
    A(dvr_table("qwen", ["S0", "S1", "S2", "S7", "S8"],
                ["test_main", "test_unseen_word", "test_unseen_item", "test_domain_phone"]))
    A("")

    A("## 2. H4 预注册检验（退化差）")
    A("")
    A("退化定义 Δ_sys,axis = DVR_sys,axis − DVR_sys,test_main；检验统计量 = 配对"
      "链单元上的退化差 Δ_S2 − Δ_{S7|S8}（正值 = S2 退化更多）；单侧符号翻转置换"
      "（B=5000），Holm 校正跨 6 组。")
    A("")
    A("| 对比 | 轴 | 退化差 (pp) | p_raw | p_holm | 显著(α=0.05) | 效应≥2pp | 各seed退化差 |")
    A("|---|---|---|---|---|---|---|---|")
    for sysname in ["S7", "S8"]:
        for axis in AXES:
            e = R["h4"].get(f"{sysname}_vs_S2_{axis}")
            if not isinstance(e, dict):
                A(f"| S2 vs {sysname} | {SPLIT_ZH[axis]} | 缺失 | | | | | |")
                continue
            ps = ", ".join(f"{k}:{v:.4f}" for k, v in e["per_seed"].items())
            A(f"| S2 vs {sysname} | {SPLIT_ZH[axis]} | {e['degradation_diff_pp']*100:.2f} | "
              f"{e['p_raw']:.4f} | {e['p_holm']:.4f} | "
              f"{'是' if e['significant_holm_05'] else '否'} | "
              f"{'是' if e['effect_ge_2pp'] else '否'} | {ps} |")
    A("")
    A(f"- 预注册标准：{R['h4'].get('prereg_criterion', '')}")
    A(f"- 护栏冲突（S7/S8 在 H4 轴被标记响应塌缩）：{R['h4'].get('h4_axis_collapse_flagged')}")
    A(f"- **预注册判定：{R['h4'].get('prereg_verdict', 'PENDING')}**")
    A("")

    A("## 3. 护栏扫描（各轴 RR / NDCG@5）")
    A("")
    for axis in AXES:
        block = R["guardrails"].get(axis, {})
        if not block:
            continue
        A(f"### {SPLIT_ZH[axis]}")
        A("")
        A("| 系统 | RR 均值 [seed区间] | ΔRR vs S0 | 塌缩标记 | NDCG@5 | ΔNDCG5 vs S2 | 效用受损 |")
        A("|---|---|---|---|---|---|---|")
        for name, g in block.items():
            rr = g.get("RR")
            rrs = (f"{rr['mean']:.4f} [{rr['seed_range'][0]:.4f}, {rr['seed_range'][1]:.4f}]"
                   if rr else "—")
            A(f"| {name} | {rrs} | {g.get('RR_delta_vs_S0', '—')} | "
              f"{g.get('responsiveness_collapse', '—')} | {g.get('NDCG5', '—')} | "
              f"{g.get('NDCG5_delta_vs_S2', '—')} | {g.get('utility_damage_flag', '—')} |")
        A("")

    A("## 4. 4c 未见强度转换（描述性，1 seed）")
    A("")
    A(f"- {R['axis_4c']['note']}")
    A(f"- 训练所见等级：{R['axis_4c']['levels_seen_in_training']}；评测为 test_main 全 4 级渲染，"
      "等级 3 的两个转换（2→3、3→4）对重训模型是未见强度")
    A("")
    A("| 系统 | 全链 DVR | RR | viol 1→2 | viol 2→3（未见） | viol 3→4（未见） | viol 2→4 跳变（已见） | 已见子链{1,2,4} DVR | NDCG@5 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for retrain in ["S2c", "S7c", "S8c"]:
        e = R["axis_4c"].get(retrain, {})
        for tag, t in [(retrain + "（重训 lv124）", e.get("retrained_lv124")),
                       (retrain[:-1] + "（study3 全等级，同 seed）", e.get("study3_full_levels_same_seed"))]:
            if not t:
                A(f"| {tag} | 缺失 | | | | | | | |")
                continue
            A(f"| {tag} | {t['DVR']:.4f} | {t['RR']:.4f} | {t['viol_1->2']:.4f} | "
              f"{t['viol_2->3']:.4f} | {t['viol_3->4']:.4f} | {t['viol_2->4_jump']:.4f} | "
              f"{t['DVR_seen_subchain_124']:.4f} | {t.get('NDCG5', '—')} |")
    A("")

    A("## 5. T4 强化未见措辞族（探索性，非预注册，不进主证据链）")
    A("")
    A(f"- 状态：{R['t4_exploratory']['status']}")
    A("- 校准门（机械三项）：round-trip 全对 = "
      f"{CAL['checks']['roundtrip_all_correct']}；未见性证书（原 T1–T3 解析器零命中）= "
      f"{CAL['checks']['unseen_certificate']}；跨族零碰撞 = "
      f"{CAL['checks']['no_cross_family_collision']}")
    A("- **等级单调性人检（待用户目检确认）**，4 级阶梯（以 price 为例）：")
    for lv in ("1", "2", "3", "4"):
        A(f"  {lv}. {CAL['templates'][lv].format(attr='a low price', noun='price')}")
    A("")
    A("| 系统 | test_main | test_unseen_word (T3) | test_t4_word (T4) | RR@T4 |")
    A("|---|---|---|---|---|")
    for sysname in ["S0", "S2", "S7", "S8"]:
        row = R["t4_exploratory"].get(f"{sysname}_llama", {})
        A(f"| {sysname} | {cell(row.get('test_main'))} | {cell(row.get('test_unseen_word'))} | "
          f"{cell(row.get('test_t4_word'))} | {row.get('RR_test_t4_word', '—')} |")
    A("")

    # 6. narrative ------------------------------------------------------------
    A("## 6. 结果叙事（由入库数值机械生成）")
    A("")
    d = R["dvr_table"]

    def m(sysname, split, family="llama"):
        e = d.get(f"{sysname}_{family}", {}).get(split)
        return e["mean"] if isinstance(e, dict) else None

    for axis in AXES:
        parts = []
        for sysname in ["S0", "S1", "S2", "S7", "S8"]:
            v = m(sysname, axis)
            if v is not None:
                base = m(sysname, "test_main")
                parts.append(f"{sysname} {v:.4f}（较 test_main {v - base:+.4f}）"
                             if base is not None else f"{sysname} {v:.4f}")
        A(f"- **{SPLIT_ZH[axis]}**：DVR " + "；".join(parts) + "。")
    A("")
    for sysname in ["S7", "S8"]:
        rows = [R["h4"].get(f"{sysname}_vs_S2_{a}") for a in AXES]
        rows = [r for r in rows if isinstance(r, dict)]
        if rows:
            sig = sum(r["significant_holm_05"] for r in rows)
            eff = sum(r["effect_ge_2pp"] for r in rows)
            A(f"- {sysname} vs S2 退化差：3 轴中 {sig} 轴显著（Holm α=0.05）、"
              f"{eff} 轴效应 ≥ 2pp。")
    A(f"- **H4 预注册判定：{R['h4'].get('prereg_verdict', 'PENDING')}**（机械输出，"
      "判定标准与门槛为 2026-08-03 固化，效应量采用退化差口径——Study 3 H3 地板效应教训的直接应用）。")
    A("")

    A("## 7. 局限与后续")
    A("")
    A("- 4c 为 1-seed 描述性轴（裁定 a）；如被选为 Study 6/7 第二评测面，须补齐 3 seeds 后方可进入结论。")
    A("- T4 为探索性轴，校准通过但等级阶梯需用户人检确认；其结果不进入主证据链。")
    A("- 手机目录许可状态存在不确定性（无显式数据集许可，上游为 GSMArena 抓取数据），"
      "已如实记录于 manifest，仅用于研究评测。")
    A("- S1/S9 本 study 为 oracle 解析形态；解析噪声下的投影退化见 Study 3 M3。")
    A("")
    A("**M3 停车**：本报告与 Study 6 计划细化待用户评审后继续。")
    A("")

    out = ROOT / "reports/study4_report.md"
    out.write_text("\n".join(L))
    shutil.copy(ROOT / "results/study4/h4_report.json",
                ROOT / "results/study4_statistical_tests.json")
    print(f"wrote {out} ({len(L)} lines) + results/study4_statistical_tests.json")


if __name__ == "__main__":
    main()
