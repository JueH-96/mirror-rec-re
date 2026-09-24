"""Render reports/study3_report.md (plan §6 deliverable) from the committed
main-matrix report JSON. Raw numbers + preregistered tests only; interpretation
is deliberately absent (M3 scope).

Also copies the stats JSON to results/study3_statistical_tests.json (§6 name).
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/study3/main_matrix_report.json"
SPLITS = ["val", "test_main", "test_unseen_word", "test_unseen_attr"]


def f4(x):
    return f"{x:.4f}"


def dvr_cell(e):
    return "missing" if e == "missing" else f4(e["mean"])


def seed_range(e):
    lo, hi = e["seed_range"]
    return f"[{f4(lo)}, {f4(hi)}]"


def ci(e):
    lo, hi = e["ci95_seedavg_chains"]
    return f"[{f4(lo)}, {f4(hi)}]"


def dvr_table(dvr, family, systems):
    lines = ["| 系统 | val | test_main（主终点） | 95% CI | 跨 seed 区间 | T3 unseen_word | unseen_attr |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        e = dvr.get(f"{s}_{family}")
        if not e:
            continue
        tm = e["test_main"]
        single = "single" in tm["per_seed"] or len(tm["per_seed"]) == 1
        lines.append("| {} | {} | **{}** | {} | {} | {} | {} |".format(
            s, dvr_cell(e["val"]), dvr_cell(tm), ci(tm),
            "—" if single else seed_range(tm),
            dvr_cell(e["test_unseen_word"]), dvr_cell(e["test_unseen_attr"])))
    return "\n".join(lines)


def main():
    r = json.loads(SRC.read_text())
    dvr, h3, guard, cons = (r["dvr_table"], r["h3"], r["guardrails"],
                            r["seed1043_consistency"])

    md = []
    md.append("# Study 3 M2 主矩阵停车报告\n")
    md.append("- 日期：2026-07-31；里程碑：M2（主矩阵训练+评测）完成后停车，等待用户审阅后进入 M3")
    md.append("- 完成判定：`experiments/verify_m2_complete.py` → M2_VERIFY_PASS（24/24 run）；"
              "S0/S1/S9 评测链（`experiments/run_s0_s1_s9_chain.sh`）CHAIN_DONE 无失败")
    md.append("- 数据来源：`results/study3/main_matrix_report.json`"
              "（同步落盘为 `results/study3_statistical_tests.json`，计划 §6 命名）；"
              "本报告只列原始数据与预注册检验，解读留待 M3")
    md.append("- 系统定义补充：" + "；".join(
        f"{k} = {v}" for k, v in r["system_definitions"].items()) + "\n")

    md.append("## 1. 完整对照表：DVR（链级违反率）\n")
    md.append("### Llama-3.1-8B（S2–S9 为 3 seeds 均值；S0/S1 单次确定性推理）\n")
    md.append(dvr_table(dvr, "llama",
                        ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]))
    md.append("\n### Qwen2.5-14B（确认组，seed 1042）\n")
    md.append(dvr_table(dvr, "qwen", ["S0", "S1", "S2", "S7", "S8", "S9"]))
    n_ll = dvr["S0_llama"]["test_main"]["n_chains"]
    n_qw = dvr["S0_qwen"]["test_main"]["n_chains"]
    n_tr = dvr["S2_llama"]["test_main"]["n_chains"]
    md.append(f"""
事实性备注：
- S1/S9 的 DVR=0 是等渗投影的构造性质（focal delta 序列经 PAVA 后单调 ⇒ 无方向违反）。
- S0/S1 的 test_main 链数少于训练系统（llama {n_ll} / qwen {n_qw}，训练系统 {n_tr}），
  因零样本存在解析失败导致部分链不完整。
- 投影调整量（见 `experiments/logs/s0s1s9_chain.log`）：S1_llama test_main 1169/3040 行、
  S1_qwen 644/3040 行被调整；S9 各 run 近乎零调整（S7/S8 输出本身已单调，
  仅 S9_llama_seed1043 unseen_attr 12 行、S9_qwen unseen_attr 4 行）。
""")

    md.append("## 2. H3 预注册判定（S7 vs S2–S6，test_main，配对置换 B=5000，Holm 校正）\n")
    md.append("| 配对 | 效应量（DVR 差，seed 均值） | 逐 seed 差 (1042/1043/1044) | p_holm | 显著 (α=.05) | 效应≥2pp |")
    md.append("|---|---|---|---|---|---|")
    for s in ["S2", "S3", "S4", "S5", "S6"]:
        e = h3[f"S7_vs_{s}"]
        ps = e["per_seed_diff"]
        md.append("| S7 vs {} | {} | {} / {} / {} | {:.4f} | {} | {} |".format(
            s, f4(e["dvr_diff_seedavg"]),
            f4(ps["1042"]), f4(ps["1043"]), f4(ps["1044"]),
            e["p_holm"],
            "是" if e["significant_holm_05"] else "否",
            "是" if e["effect_ge_2pp"] else "否"))
    n_sig = sum(h3[f"S7_vs_{s}"]["significant_holm_05"] for s in ["S2", "S3", "S4", "S5", "S6"])
    eff = [s for s in ["S2", "S3", "S4", "S5", "S6"] if h3[f"S7_vs_{s}"]["effect_ge_2pp"]]
    md.append(f"""
预注册标准（{h3['prereg_criterion']}）的机械套用结果：
- p_holm<0.05：{n_sig}/5 组满足；
- 绝对 DVR 降幅 ≥2pp：仅 {len(eff)}/5 组满足（{', '.join('S7 vs ' + s for s in eff)}）；
- S7 responsiveness_collapse：未触发。

**预注册判定（用户 2026-07-31 审阅裁决）：H3 不成立**（`{h3['prereg_verdict']}`）。
地板效应解释仅可在 M3 讨论中以明确标注的 post-hoc 形式出现。
""")

    md.append("## 3. 护栏扫描（test_main）\n")
    md.append("| 系统 | RR 均值 | ΔRR vs S0 | responsiveness_collapse (>5pp降) | NDCG@5 | Δ vs S2 | utility_damage |")
    md.append("|---|---|---|---|---|---|---|")
    order = [f"{s}_llama" for s in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]] + \
            [f"{s}_qwen" for s in ["S0", "S1", "S2", "S7", "S8", "S9"]]
    for k in order:
        g = guard.get(k)
        if not g:
            continue
        base = k.startswith("S0_")
        md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            k, f4(g["RR_test_main"]["mean"]),
            "—" if base else f"{g['RR_delta_vs_S0']:+.4f}",
            "—" if base else ("**是**" if g["responsiveness_collapse"] else "否"),
            f4(g["NDCG5_test_main"]), f"{g['NDCG5_delta_vs_S2']:+.4f}",
            "是" if g["utility_damage_flag"] else "否"))
    collapsed = [k for k in order
                 if guard.get(k, {}).get("responsiveness_collapse") is True]
    md.append(f"""
触发的护栏：{', '.join(collapsed) if collapsed else '无'}"""
              + (f"（按预注册规则其 DVR 数字不计入 H3 支持证据）。" if collapsed else "。")
              + " utility_damage 全系统零触发。\n")

    md.append("## 4. S2_seed1043 补训一致性核查\n")
    md.append("| split | 1042 DVR/RR | 1043 DVR/RR（补训） | 1044 DVR/RR | 在同胞区间内 | 偏离同胞均值 |")
    md.append("|---|---|---|---|---|---|")
    for split in SPLITS:
        row = cons[split]
        md.append("| {} | {} / {} | {} / {} | {} / {} | {} | {:+.4f} |".format(
            split,
            f4(row["1042"]["DVR"]), f4(row["1042"]["RR"]),
            f4(row["1043"]["DVR"]), f4(row["1043"]["RR"]),
            f4(row["1044"]["DVR"]), f4(row["1044"]["RR"]),
            "是" if row["seed1043_within_sibling_range"] else "否",
            row["seed1043_dev_from_sibling_mean"]))
    md.append("""
说明（事实性）：seed1043 为磁盘耗尽事故后从零补训（未复用任何中断状态），
事故时间线、S2_seed1042 checkpoint 逐位校验与 checkpoint 策略变更已记录于
`experiment_manifest.yaml` → study3_m2.deviations（复现性附录素材）。
test_main 上 seed1043 DVR=0.0000 落在同胞区间 [0.0009, 0.0053] 下方，
偏离 -0.0031（方向为更低）。
""")

    # --- Section 5: M2-review supplementary checks (user ruling item 4) ----
    bm_path = ROOT / "results/study3/bclass_margin_report.json"
    if bm_path.exists():
        bm = json.loads(bm_path.read_text())
        sysd = bm["systems"]
        order = [f"{s}_llama" for s in
                 ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]] + \
                [f"{s}_qwen" for s in ["S0", "S1", "S2", "S7", "S8", "S9"]]
        md.append("## 5. M2 审阅补充核查（用户裁决第 4 项，test_main）\n")
        md.append("### 5a. DVR 按候选集类型分解（B = 真实商品对，仅评测从未训练）\n")
        md.append("| 系统 | A 类 DVR [seed区间] | B 类 DVR [seed区间] | C 类 DVR [seed区间] |")
        md.append("|---|---|---|---|")
        for k in order:
            e = sysd.get(k)
            if not e:
                continue
            cells = []
            for t in ["A", "B", "C"]:
                v = e["dvr_by_set_type"][t]
                cells.append("{} [{}, {}]".format(
                    f4(v["mean"]), f4(v["seed_range"][0]), f4(v["seed_range"][1])))
            md.append("| " + " | ".join([k] + cells) + " |")
        nb = sysd["S2_llama"]["dvr_by_set_type"]["B"]["n_chains_per_seed"]
        nb0 = sysd["S0_llama"]["dvr_by_set_type"]["B"]["n_chains_per_seed"]
        md.append(f"\n每 seed B 类链数：训练系统 {nb}（100 个合格 B 类集 × 2 措辞族）；"
                  f"S0/S1 为 {nb0}（解析失败致部分链不完整）。\n")
        md.append("### 5b. 得分 margin 分布（step = δ_{k+1} − δ_k；违反 ⇔ step < −ε，ε=0.01）\n")
        md.append("| 系统 | min-step/链 p5 | p50 | 链违反占比(≥1转移) | 链贴边占比(|min-step|≤ε) | 归一化 min-step p5 | p50 |")
        md.append("|---|---|---|---|---|---|---|")
        for k in order:
            e = sysd.get(k)
            if not e:
                continue
            m = e["margin"]
            md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                k, f4(m["min_step_pctl"]["p5"]), f4(m["min_step_pctl"]["p50"]),
                f4(m["frac_chains_violating"]), f4(m["frac_chains_near_boundary"]),
                f4(m["norm_min_step_pctl"]["p5"]), f4(m["norm_min_step_pctl"]["p50"])))
        md.append("""
事实性备注：
- 链单位 = (seed, 链)；"链违反占比"为至少 1 个转移违反的链比例
  （与第 1 节转移加权 DVR 口径不同，两者一致性已核对：如 S2_llama
  0.0061 ≈ 3×0.0021）。
- 得分为各系统自身量纲（S0/S1 为 LLM 文本分数，训练系统为 head 输出）；
  "归一化 min-step" = min-step / max|δ|（链内），量纲无关。
- S1 的链贴边占比 ~0.98 是投影构造性质：PAVA 将违反段拉平为相等
  （step 恰为 0），其零违反天然贴边；训练系统 S7/S8/S9 与 S2_qwen/S7_qwen
  的贴边占比为 0，min-step 分布整体远离边界。
""")

    # --- Section 6: parser-noise axis (plan §0-P2, user-ruled M3 axis) -----
    pn_path = ROOT / "results/study3/parser_noise_report.json"
    pn = json.loads(pn_path.read_text()) if pn_path.exists() else None
    if pn:
        md.append("## 6. 解析器噪声轴（§0-P2，test_main，Llama）\n")
        md.append("损坏模型：逐请求以率 p 损坏解析结果——50% 强度 off-by-one"
                  "（±1 截断到 [1,4]）、50% 属性混淆（均匀换成其余 3 属性）；"
                  "损坏表跨系统共享（配对比较）。违反始终按**真实**链结构度量。\n")
        md.append("| 噪声率 | S1（投影，解析依赖） DVR / RR | S8（mirror，g_a(ℓ) 条件化） DVR / RR | S7（base，无解析输入） DVR / RR |")
        md.append("|---|---|---|---|")
        s1r = pn["systems"]["S1_llama"]["by_rate"]
        s8r = pn["systems"]["S8_llama"]["by_rate"]
        s7a = pn["systems"]["S7_llama"]["all_rates"]
        for r in ["0", "5", "10", "20"]:
            md.append("| {}% | {} / {} | {} / {} | {} / {}{} |".format(
                r, f4(s1r[r]["DVR"]), f4(s1r[r]["RR"]),
                f4(s8r[r]["DVR_mean"]), f4(s8r[r]["RR_mean"]),
                f4(s7a["DVR_mean"]), f4(s7a["RR_mean"]),
                "" if r == "0" else "（恒等）"))
        md.append("""
事实性备注：
- S7（及所有 base 模式系统 S2–S5）的 forward 不消费解析出的 (属性, 强度)，
  输入在解析噪声下不变 ⇒ 输出恒等。表中 S7 各行为结构恒等性陈述，
  **未重跑**（如实标注，非伪造数据）。
- S1 的退化机制：单请求单措辞的部署形态下，任一请求解析损坏即令整链
  无法装配（漏加约束）——5%/10%/20% 请求级噪声分别击破 184/303/455
  条链（共 760）。
- S8 的退化机制：g_a(ℓ) 以损坏的 (a,ℓ) 条件化，仅影响被损坏请求的
  得分刻度。
- S8 逐 seed 数值见 `results/study3/parser_noise_report.json`。
""")

    # --- Section 7: results narrative (M3, user-ruled three lines) ---------
    def dvr_of(sysfam, split):
        e = dvr.get(sysfam, {}).get(split)
        return e["mean"] if isinstance(e, dict) else None

    if pn and bm_path.exists():
        s0b = sysd["S0_llama"]["dvr_by_set_type"]["B"]["mean"]
        s0bq = sysd["S0_qwen"]["dvr_by_set_type"]["B"]["mean"]
        m7 = sysd["S7_llama"]["margin"]
        md.append(f"""## 7. 结果叙事（M3，按用户裁决框架）

**主线 (a)：零样本方向违反严重，且 prompt 无法修复。**
未经训练的零样本重排序在 test_main 上的链级 DVR 为
{f4(dvr_of('S0_llama','test_main'))}（Llama）/ {f4(dvr_of('S0_qwen','test_main'))}（Qwen），
且在真实商品对（B 类）上更差（{f4(s0b)} / {f4(s0bq)}，第 5a 节）——方向违反
不是合成数据的伪象。Pilot 已确认 prompt 工程（structured /
explain-then-rank / direction-constraint 变体）无法将其压到接近零
（Go/No-Go 条件 3，reports/gonogo_report.md）。

**主线 (b)：关系型 oracle 监督近乎消除分布内违反；方向性目标在分布外
保持零违反，非方向监督则退化且跨 seed 不稳。**
除 S5（无方向信息，见主线 c）外，所有训练系统在 test_main 上
DVR ≤ {f4(dvr_of('S3_llama','test_main'))}；但在未见属性
（storage，训练全程 held-out）上，S2（普通监督）退化到
{f4(dvr_of('S2_llama','test_unseen_attr'))} 且跨 seed 区间宽达
[{f4(dvr['S2_llama']['test_unseen_attr']['seed_range'][0])}, {f4(dvr['S2_llama']['test_unseen_attr']['seed_range'][1])}]，
而 S7 三 seed 全零、S8 均值 {f4(dvr_of('S8_llama','test_unseen_attr'))}。
margin 分析（第 5b 节）表明两类"零违反"成色不同：S7 的最差转移裕度
p5 分位为 +{f4(m7['min_step_pctl']['p5'])}（贴边链占比 0），是**带裕度的内化单调性**；
S1（投影）的零违反 97.6% 贴在违反边界上（PAVA 拉平的构造性质）。
解析器噪声轴（第 6 节）进一步区分二者：S1 随解析噪声退化
（20% 噪声下 DVR 回升至 {f4(s1r['20']['DVR'])}），S7 结构性免疫，
S8 仅轻度退化（{f4(s8r['20']['DVR_mean'])}）。

**主线 (c)：非方向一致性正则适得其反——方向性是不可替代的目标属性。**
S5（一致性正则，无方向信息）不仅未降低违反
（test_main DVR {f4(dvr_of('S5_llama','test_main'))}，与 S0 同量级），
还是全矩阵唯一触发 responsiveness_collapse 护栏的系统
（RR {f4(guard['S5_llama']['RR_test_main']['mean'])}，较 S0 降
{abs(guard['S5_llama']['RR_delta_vs_S0']):.4f}）：它靠压平对干预的响应来买
一致性。同为单调性正则但带方向的 S6 则无此代价
（DVR {f4(dvr_of('S6_llama','test_main'))}，RR {f4(guard['S6_llama']['RR_test_main']['mean'])}）。

## 8. 讨论（M3）

**H3 预注册判定：不成立。** 五组配对 p_holm 全部 <0.05，但绝对 DVR 降幅
≥2pp 仅 1/5 组（S7 vs S5）满足，合取条件不成立（第 2 节）。
*Post-hoc 解释（非预注册，标注为事后分析）*：效应量条件在本矩阵上
存在地板效应——S2 等强基线已把 test_main DVR 压到
{f4(dvr_of('S2_llama','test_main'))}（0.21%），距零不足 2pp，预注册时设定的
绝对降幅门槛在该地板上不可能达成；这属于门槛设计与地板位置的错配，
不构成对方向性目标价值的证伪。方向性目标的价值证据由预注册检验之外的
三条线承载：未见属性零违反（主线 b）、margin 裕度与解析噪声免疫
（第 5b/6 节）、以及 S5 的反例（主线 c）。后续 study 的效应量门槛
改用"退化差"度量（见 Study 4–7 计划草案）。

**探索性轴状态。** "强化未见措辞族"（T4）已按裁决标注为探索性
（非预注册，不入主证据链）；因新措辞族须先通过 M0 同款强度校准才能产生
可用数据，安排在 Study 4a 随未见措辞轴一并执行（见计划草案 4a 与
待裁定项 c）。

**局限（事实性）。** 单领域（笔记本目录）；训练措辞族仅 T1/T2；
Qwen 确认组单 seed；S9 的可组合性证据在本矩阵上是平凡的
（S7/S8 输出已单调，投影近乎零调整）；解析器噪声轴目前仅覆盖
test_main 与 Llama 家族。

---

**M3 停车。** 完整报告与统计叙事已落盘；Study 4–7 计划草案见
`reports/study4_7_plan_draft.md`，等待用户审阅。
""")

    out = ROOT / "reports/study3_report.md"
    out.write_text("\n".join(md))
    shutil.copy(SRC, ROOT / "results/study3_statistical_tests.json")
    print("wrote", out)
    print("copied ->", ROOT / "results/study3_statistical_tests.json")


if __name__ == "__main__":
    main()
