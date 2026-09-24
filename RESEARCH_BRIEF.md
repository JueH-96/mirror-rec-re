# 研究规范（预注册执行契约）

## 面向有序自然语言偏好干预的 LLM 推荐单调响应研究

**研究对象：** MIRROR-Rec：面向 LLM 推荐的单调偏好干预响应排序方法


---

## 0. 文档用途

本文档不是面向导师评审、学位申请或个人时间管理的 Research Proposal，而是供执行方直接执行本项目的研究规范。文档中的定义、数据约束、方法组件、指标、决策门槛和交付文件应被视为可执行契约。

执行方不得自行扩展到其他本项目方向，也不得在未重新进行文献撞车检查的情况下增加新的研究问题。任何自动生成的结论必须来自可复现的实验结果，不得依赖人工判断或主观筛选。

## 1. 执行摘要

大语言模型（LLM）使用户能够通过自然语言直接表达推荐需求。现有研究已经覆盖指令式推荐、自然语言可控推荐、序数偏好强度、偏好优化和变形测试。然而，现有方法通常只关注系统是否“响应”了用户偏好，并未显式要求推荐排序变化的**方向**必须与用户偏好强度的变化一致。

例如，假设两台笔记本电脑 \(i\) 和 \(j\) 的其他条件相近，并且 \(i\) 的电池续航明显优于 \(j\)。当用户将需求从“续航比较重要”修改为“续航是我最重要的考虑因素”时，\(i\) 相对于 \(j\) 的排序优势不应下降。在固定其他需求、候选集合和物品属性的前提下，这是一条可自动验证的逻辑关系。然而，LLM 重排序模型以概率化文本处理方式完成决策，可能产生与偏好变化方向相反的排序响应。

本研究将这种现象定义为：

> **偏好干预—响应不一致（Preference Intervention–Response Inconsistency）**

核心研究问题为：

> **RQ1：干预对比排序优化能否减少 LLM 重排序模型在有序、单属性自然语言偏好干预下产生的方向不一致响应？**

本研究规范要求完成以下研究产物：

1. 形式化定义自然语言偏好干预下的成对方向一致性；
2. 构建完全自动生成、完全自动验证的测试基准；
3. 提出方向违反、严格反转、响应性、非目标变化和推荐效用等指标；
4. 提出新方法 **MIRROR-Rec**；
5. 研究可在推理阶段强制满足方向约束的排序投影方法；
6. 建立不依赖人工标注、人工审查或人工结果修复的全自动实验流程；
7. 为后续扩展提供机器可读研究规范、数据格式和执行契约。

本项目的有效研究范围严格限定为：

- 固定候选集合重排序；
- 单属性软偏好；
- 客观可排序的数值属性；
- 有序自然语言偏好强度；
- 自动关系型 ground truth；
- 自动化训练、评测和统计分析。

本项目暂不处理多轮对话、长期记忆、动态偏好漂移、多属性同时修改和真人用户实验，以控制范围并减少混杂因素。

---

## 2. 研究背景与动机

### 2.1 自然语言正在成为推荐系统的新交互接口

传统推荐系统主要通过用户点击、评分、购买和浏览历史推断偏好。LLM 推荐系统提供了另一种交互方式：用户可以直接用自然语言表达需求，例如：

- “我更在意续航。”
- “价格比品牌重要。”
- “我希望设备更轻一些。”
- “我不介意多花一点钱，但一定要有更大的存储空间。”

指令式推荐和自然语言可控推荐已经证明，LLM 能够把这些请求转换为推荐信号。然而，现有研究通常只评价：

\[
\text{用户输入发生变化}
\Rightarrow
\text{推荐结果发生变化}
\]

但真正可靠的系统还应满足：

\[
\text{用户偏好沿某个方向变化}
\Rightarrow
\text{推荐结果沿正确方向变化}
\]

### 2.2 偏好强度建模不等于方向一致性

现有工作已经利用：

- 一星至五星评分；
- 浏览、收藏和购买等行为等级；
- 正向与负向反馈；
- 不同交互强度；
- 序数语义锚点；

建模用户偏好强度。

本研究关注不同的问题。输入不是历史评分等级，而是同一自然语言需求的有序干预：

\[
q_a^{-}
\prec_a
q_a^{0}
\prec_a
q_a^{+}
\prec_a
q_a^{++}
\]

例如：

| 等级 | 自然语言请求 |
|---|---|
| \(-1\) | 续航对我不重要 |
| \(0\) | 我没有特别的续航要求 |
| \(+1\) | 我比较重视续航 |
| \(+2\) | 续航是我最重要的考虑因素 |

研究目标不是简单判断“强偏好对应更高权重”，而是研究**同一个物品对在不同偏好干预等级下的相对排序差值是否按正确方向变化**。

### 2.3 传统推荐指标不足以发现该问题

NDCG、Recall 和 Hit Rate 等指标用于衡量相关物品是否排在前面，但无法判断模型是否正确响应了偏好修改。

一个模型可能具有较高 NDCG，却仍出现：

1. 用户加强对续航的偏好；
2. 候选 \(i\) 的续航明显优于 \(j\)；
3. 模型却降低 \(i\) 相对于 \(j\) 的排序优势。

因此，本研究采用关系型评测：

\[
\text{不要求人工给出唯一正确排名}
\]

而是要求多次模型执行之间满足一个可计算关系：

\[
\text{输入干预关系}
\Rightarrow
\text{输出排序关系}
\]

---

## 3. 文献定位与研究缺口

### 3.1 指令式推荐与自然语言可控推荐

现有研究已经证明：

- 自然语言可以表达用户偏好；
- 用户可以使用自然语言控制推荐；
- 自然语言请求可以作为检索或排序信号；
- 系统可以根据 critique 调整推荐。

这些工作主要回答：

> 系统是否能够按照自然语言指令改变推荐？

本研究回答：

> 系统的变化方向是否与偏好变化的逻辑方向一致？

设 \(q_a^r\) 与 \(q_a^s\) 是同一属性 \(a\) 上的两个偏好强度，且：

\[
q_a^r \preceq_a q_a^s
\]

如果物品 \(i\) 在属性 \(a\) 上优于物品 \(j\)，即：

\[
x_{i,a}>x_{j,a}
\]

则希望满足：

\[
\Delta_{ij}(q_a^s)
\geq
\Delta_{ij}(q_a^r)
\]

其中：

\[
\Delta_{ij}(q)
=
s(i\mid q)-s(j\mid q)
\]

### 3.2 序数偏好强度研究

现有序数推荐方法主要从评分、交互等级或结构化反馈中学习偏好强度。它们解决的是：

\[
\text{更强历史反馈}
\Rightarrow
\text{更强偏好表示}
\]

本研究解决的是：

\[
\text{更强自然语言偏好干预}
\Rightarrow
\text{同一物品对的排序优势沿正确方向变化}
\]

### 3.3 单调推荐研究

推荐系统中使用单调函数并非新概念。传统研究可能要求：

\[
x_{i,a}\uparrow
\Rightarrow
s(i)\uparrow
\]

本研究的单调关系是二维关系：

\[
\text{PreferenceStrength}_a(q)\uparrow
\times
(x_{i,a}-x_{j,a})>0
\]

应导致：

\[
[s(i\mid q)-s(j\mid q)]\uparrow
\]

因此，本研究约束的是：

> 用户侧偏好强度变化与物品侧属性优势之间的交互响应。

### 3.4 变形测试与推荐一致性

已有研究使用变形测试检查：

- 同义表达是否产生相似推荐；
- 无关输入变化是否影响结果；
- 提示词变化是否改变排序；
- 候选顺序是否影响输出。

这类关系通常是**不变性**：

\[
q\equiv q'
\Rightarrow
R(q)\approx R(q')
\]

本研究不是不变性，而是**方向等变性**：

\[
q_a^r\prec_a q_a^s
\Rightarrow
R(q_a^s)
\text{应沿目标属性对应方向变化}
\]

### 3.5 精确研究缺口

目前未发现一项工作同时覆盖以下六项：

1. 对同一需求构造有序自然语言偏好干预链；
2. 每次只改变一个目标属性；
3. 使用结构化物品属性确定客观的成对属性优势；
4. 比较同一物品对跨干预等级的排序差值；
5. 把反向响应定义为独立错误；
6. 提出显式优化或约束投影方法修复该错误。

本研究不会声称以下内容本身是首次提出：

- 自然语言推荐；
- 偏好强度；
- 单调推荐；
- 变形测试；
- hinge loss；
- 单调神经网络；
- isotonic projection。

创新点是它们围绕一个此前未被系统解决的**偏好干预—排序响应关系**形成统一问题、基准和方法。

---

## 4. 研究问题、假设与范围

### 4.1 研究问题

> **RQ1：干预对比排序优化能否减少 LLM 重排序模型在有序、单属性自然语言偏好干预下产生的方向不一致响应？**

### 4.2 研究假设

**H1：现有模型确实存在该错误。**  
通用 LLM 和推荐专用 LLM 在受控偏好强度变化下存在非零且稳定的方向违反率。

**H2：提示工程不足以彻底解决。**  
结构化提示、解释后排序或 chain-of-thought 只能减少部分违反，无法跨模型、属性、措辞和候选集合稳定消除。

**H3：跨干预监督有效。**  
干预对比排序目标比普通监督重排序、数据增强和单请求 pairwise loss 更能降低方向违反。

**H4：结构单调约束提升泛化。**  
属性条件化的单调参数化能够提高对未见措辞和未见强度组合的泛化。

**H5：推理投影可提供强保证。**  
在满足资格条件的物品对上，约束投影可强制满足方向一致性，同时保持较低排序效用损失。

**H6：改进不会依赖固定排序作弊。**  
模型在降低违反率的同时仍保持足够响应性和标准推荐性能。

### 4.3 研究范围

第一阶段包括：

- 固定候选集重排序；
- 单属性自然语言软偏好；
- 有序偏好强度和正负方向；
- 客观数值属性；
- LLM 打分或 pairwise 排序；
- 自动生成的干预链；
- 自动关系型验证。

第一阶段不包括：

- 多轮对话；
- 用户长期记忆；
- 隐式偏好漂移；
- 多属性同时干预；
- 主观属性；
- 理想点偏好；
- 工具调用；
- 推荐解释；
- 在线真人用户实验。

---

## 5. 形式化问题定义

### 5.1 输入

设：

- \(C=\{i_1,\ldots,i_n\}\) 为固定候选集合；
- \(X_i=(x_{i,1},\ldots,x_{i,d})\) 为物品 \(i\) 的结构化属性；
- \(a\) 为目标属性；
- \(q_a^k\) 为第 \(k\) 个自然语言偏好等级；
- \(f_\theta(q,C)\) 为 LLM 重排序模型；
- \(s_\theta(i\mid q,C)\) 为模型对物品 \(i\) 的得分。

对于“越大越好”的属性，若：

\[
x_{i,a}>x_{j,a}
\]

则记：

\[
i\succ_a j
\]

对于价格、重量等“越小越好”的属性，定义效用归一化函数 \(u_a(x)\)，统一转换为：

\[
u_a(x_{i,a})>u_a(x_{j,a})
\]

### 5.2 有序自然语言偏好干预

定义干预链：

\[
Q_a=(q_a^1,q_a^2,\ldots,q_a^K)
\]

且：

\[
q_a^1\prec_a q_a^2\prec_a\cdots\prec_a q_a^K
\]

每次干预只能改变属性 \(a\) 的偏好强度或极性。以下内容必须保持不变：

- 其他用户需求；
- 候选集合；
- 候选顺序；
- 物品描述；
- 固定约束；
- 非目标属性。

### 5.3 成对方向一致性

定义成对排序差：

\[
\Delta_{ij}^{k}
=
s_\theta(i\mid q_a^k,C)
-
s_\theta(j\mid q_a^k,C)
\]

对于 \(i\succ_a j\)，方向一致性要求：

\[
\Delta_{ij}^1
\leq
\Delta_{ij}^2
\leq
\cdots
\leq
\Delta_{ij}^K
\]

若：

\[
\Delta_{ij}^{k+1}
<
\Delta_{ij}^{k}-\epsilon
\]

则发生局部方向违反。

若：

\[
\Delta_{ij}^{k}\geq0
\quad\text{且}\quad
\Delta_{ij}^{k+1}<0
\]

则发生严格反转。

### 5.4 自动资格条件

仅当以下条件全部成立时，物品对才进入主评测：

1. 两个物品均具有目标属性；
2. 属性方向在注册表中明确；
3. 属性差异超过最小阈值；
4. 两个物品均不违反固定硬约束；
5. 干预链各版本候选集合完全一致；
6. 干预生成器没有修改非目标信息；
7. 模型输出能够被结构化解析；
8. 物品 ID 均来自候选集合。

所有条件均由程序自动检查。

---

## 6. 新方法：MIRROR-Rec

MIRROR-Rec 全称为：

> **Monotonic Intervention-Responsive Ranking Optimization for Recommendation**

中文可译为：

> **面向推荐的单调偏好干预响应排序优化**

方法包含四个核心模块：

1. 有序偏好表示；
2. 属性条件化单调得分；
3. 干预对比排序目标；
4. 可选的推理阶段约束投影。

### 6.1 有序偏好表示

每个请求表示为：

\[
z(q)
=
(z_{\text{base}},a,p_a,\ell_a)
\]

其中：

- \(z_{\text{base}}\)：非目标需求表示；
- \(a\)：目标属性；
- \(p_a\in\{-1,0,+1\}\)：偏好极性；
- \(\ell_a\)：序数强度等级。

在 benchmark 中，这些结构化字段由生成器直接提供，不需要人工标注。

面向部署实验时，可训练自动解析器把自然语言转换为：

```json
{
  "attribute": "battery_life",
  "polarity": "positive",
  "strength": 3
}
```

解析器的 ground truth 同样来自自动生成规格。

### 6.2 属性条件化得分分解

物品得分定义为：

\[
s_\theta(i\mid q,C)
=
b_\theta(i\mid z_{\text{base}},C)
+
g_{\phi,a}(\ell_a,p_a)h_{\psi,a}(i)
+
r_\theta(i,q,C)
\]

其中：

- \(b_\theta\)：基础相关性；
- \(h_{\psi,a}(i)\)：物品在属性 \(a\) 上的效用；
- \(g_{\phi,a}\)：偏好强度映射；
- \(r_\theta\)：残差交互项。

对正向偏好，要求 \(g_{\phi,a}\) 单调不下降：

\[
g_{\phi,a}(k)
=
g_{\phi,a}(0)
+
\sum_{t=1}^{k}
\operatorname{softplus}(\eta_{a,t})
\]

对负向偏好使用对应方向的单调约束。

### 6.3 干预对比排序目标

对于 \(i\succ_a j\)，定义：

\[
L_{\mathrm{ICR}}
=
\max
\left(
0,
m_{ij,a}
-
[
\Delta_{ij}^{k+1}
-
\Delta_{ij}^{k}
]
\right)
\]

其中 margin 根据属性优势调整：

\[
m_{ij,a}
=
\alpha
\cdot
\operatorname{clip}
\left(
u_a(x_{i,a})-u_a(x_{j,a}),
0,
m_{\max}
\right)
\]

对于完整干预链，定义：

\[
L_{\mathrm{chain}}
=
\sum_{r<s}
\max
\left(
0,
m_{r,s}
-
[
\Delta_{ij}^{s}
-
\Delta_{ij}^{r}
]
\right)
\]

完整目标：

\[
L
=
L_{\mathrm{rec}}
+
\lambda_1L_{\mathrm{ICR}}
+
\lambda_2L_{\mathrm{chain}}
+
\lambda_3L_{\mathrm{responsive}}
+
\lambda_4L_{\mathrm{off-target}}
\]

### 6.4 避免固定排序作弊

如果模型永远输出相同排序，它可能在弱指标下表现为“没有下降”。

因此定义响应目标：

\[
L_{\mathrm{responsive}}
=
\max
\left(
0,
\rho_{ij,a}
-
[
\Delta_{ij}^{k+1}
-
\Delta_{ij}^{k}
]
\right)
\]

该目标只对目标属性差异足够大的样本启用。

同时保留 \(L_{\mathrm{rec}}\)，确保标准推荐效用不显著下降。

### 6.5 非目标稳定性

当两个物品在目标属性上相近时，用户加强该属性偏好不应导致任意的大幅相对变化：

\[
L_{\mathrm{off-target}}
=
\sum_{(i,j)\in\mathcal{N}_a}
\left|
\Delta_{ij}^{k+1}
-
\Delta_{ij}^{k}
\right|
\]

该项用于减少副作用，不作为独立 RQ。

### 6.6 推理阶段约束投影

对于冻结模型或黑盒模型，研究后处理投影：

\[
\min_{\tilde{s}}
\sum_{k=1}^{K}
\sum_{i\in C}
(\tilde{s}_i^k-s_i^k)^2
\]

满足：

\[
(\tilde{s}_i^{k+1}-\tilde{s}_j^{k+1})
\geq
(\tilde{s}_i^{k}-\tilde{s}_j^{k})
\]

仅对自动判定为 eligible 的物品对施加约束。

该模块可在不重新训练模型的情况下提供强制方向一致性。

---

## 7. 完全自动化的数据集构造

### 7.1 无人工验证原则

本研究不使用以下人工环节：

- 人工标注用户偏好等级；
- 人工判断物品属性优劣；
- 人工检查推荐结果；
- 人工修复模型输出；
- 人工筛选不利结果；
- 人工判断自然语言是否保持原意；
- LLM-as-a-judge 作为主 oracle。

所有 ground truth 来自：

- 结构化物品属性；
- 机器可读属性注册表；
- 确定性生成规则；
- round-trip 解析；
- 可执行关系检查；
- 多解析器一致性过滤。

### 7.2 领域选择

第一领域建议为：

> 笔记本电脑或智能手机

可用属性包括：

- 价格；
- 重量；
- 续航；
- 存储；
- 内存；
- 屏幕尺寸；
- 摄像头像素。

第二领域可考虑电影，但只使用客观属性：

- 时长；
- 年份；
-评分；
- 流行度。

主观属性暂不进入主 benchmark。

### 7.3 属性注册表

示例：

```yaml
battery_life:
  type: numeric
  unit: hours
  preference_direction: higher
  min_pair_gap: 3
  valid_range: [1, 30]

price:
  type: numeric
  unit: currency
  preference_direction: lower
  min_pair_gap: 100
  valid_range: [100, 10000]

weight:
  type: numeric
  unit: kg
  preference_direction: lower
  min_pair_gap: 0.2
  valid_range: [0.2, 10]
```

注册表是所有自动 ground truth 的唯一来源。

### 7.4 候选集合构造

#### A. 完全控制合成对

只改变目标属性，其他属性完全相同。

优点：

- 内部有效性最强；
- 可明确判断反向响应；
- 不需要人工标注。

#### B. 匹配真实物品对

使用真实商品数据，并最小化非目标属性差异：

\[
d_{-a}(i,j)
=
\sum_{b\neq a}
w_b
\left|
\hat{x}_{i,b}
-
\hat{x}_{j,b}
\right|
\]

同时要求：

\[
|x_{i,a}-x_{j,a}|
\geq
\delta_a
\]

#### C. 权衡型候选集合

目标属性更优的物品在一个次要属性上略差，用于测试真实权衡场景。

### 7.5 干预规格优先

先生成结构化规格：

```json
{
  "target_attribute": "battery_life",
  "polarity": "positive",
  "strength_level": 3,
  "base_context_id": "CTX-001",
  "fixed_constraints": {
    "max_price": 1500
  }
}
```

再由模板或 LLM 生成自然语言表面形式。

### 7.6 自动 paraphrase 验证

LLM 可用于生成多样化措辞，但不能定义 ground truth。

每个 paraphrase 必须通过：

1. 多个解析器回译；
2. 属性一致性检查；
3. 极性一致性检查；
4. 强度一致性检查；
5. 固定约束一致性检查；
6. 禁止属性词检测；
7. round-trip canonicalization；
8. 自动丢弃不一致样本。

宁可保守过滤，也不进行人工修正。

### 7.7 数据泄漏控制

训练、验证和测试需要隔离：

- 物品 ID；
- base context；
- 模板；
- paraphrase 生成种子；
- 干预链；
- 属性组合。

额外设置：

- 未见模板测试；
- 未见物品测试；
- 未见属性强度转换测试；
- 跨领域测试。

---

## 8. 全自动实验流水线

### 8.1 流程阶段

```text
1. ingest_catalog
2. validate_schema
3. normalise_attributes
4. construct_candidate_sets
5. generate_intervention_specs
6. render_templates
7. generate_paraphrases
8. roundtrip_validate_language
9. run_models
10. parse_rankings
11. validate_outputs
12. compute_relational_metrics
13. train_mirror_rec
14. run_ablations
15. run_statistical_tests
16. build_tables_and_figures
17. generate_reproducibility_report
```

### 8.2 输出验证

模型必须返回 JSON：

```json
{
  "ranking": [
    {"item_id": "A", "score": 0.82},
    {"item_id": "B", "score": 0.64}
  ]
}
```

系统自动检查：

- JSON 是否有效；
- item ID 是否来自候选集合；
- 是否有重复；
- 是否遗漏候选；
- 得分是否可解析；
- 排名是否完整。

失败输出只允许按照固定策略自动重试。重试失败后记为格式错误，不进行人工修复。

### 8.3 自动 oracle

主 oracle 只需要：

- 偏好强度顺序；
- 目标属性；
- 物品属性优势；
- 模型得分或排名。

不需要人工提供唯一正确 Top-\(K\)。

### 8.4 多次执行

每次模型调用记录：

- 模型名称与版本；
- prompt 版本；
- temperature；
- seed；
- 时间戳；
- 原始响应；
- 解析状态；
- token 数；
- 延迟；
- 成本。

### 8.5 自动质量门槛

出现以下情况时自动停止或标记：

- 属性值超出注册范围；
- round-trip 失败；
- 干预链候选集合发生变化；
- 属性差异低于阈值；
- 模型输出无效物品；
- 指标样本数不足；
- 数据切分泄漏；
- 配置文件不完整。

---

## 9. 评测指标

### 9.1 方向违反率

\[
DVR
=
\frac{
\sum
\mathbb{I}
[
\Delta_{ij}^{k+1}
<
\Delta_{ij}^{k}
-
\epsilon
]
}{
N_{\text{eligible transitions}}
}
\]

### 9.2 严格反转率

\[
SRR
=
\frac{
\sum
\mathbb{I}
[
\Delta_{ij}^{k}\geq0
\land
\Delta_{ij}^{k+1}<0
]
}{
N_{\text{eligible transitions}}
}
\]

### 9.3 链一致性

对：

\[
\Delta^{-},
\Delta^{0},
\Delta^{+},
\Delta^{++}
\]

计算：

- 非递减比例；
- Kendall \(\tau\)；
- Spearman 相关系数；
- 链级完全一致率。

### 9.4 响应率

\[
RR
=
\frac{
\sum
\mathbb{I}
[
\Delta_{ij}^{k+1}
>
\Delta_{ij}^{k}
+
\rho
]
}{
N_{\text{material transitions}}
}
\]

### 9.5 非目标变化

使用：

- pairwise flip rate；
- Kendall distance；
- Rank-Biased Overlap；
- 与目标属性近似相同物品对的排序变化。

### 9.6 标准推荐性能

- NDCG@\(K\)；
- Recall@\(K\)；
- Hit Rate@\(K\)；
- pairwise accuracy；
- constraint satisfaction；
- candidate coverage。

### 9.7 输出可靠性

- 有效输出率；
- 目录内物品率；
- 重复物品率；
- 得分完整率。

结果报告应同时展示：

\[
\text{方向一致性}
\quad
\text{响应性}
\quad
\text{普通推荐效用}
\]

而不是把它们合并成一个难以解释的单一分数。

---

## 10. 对比基线

至少包括：

1. 结构化属性规则模型；
2. Zero-shot LLM 重排序；
3. 提示词方向一致性约束；
4. 结构化偏好提示；
5. reason-then-rank；
6. 普通监督重排序；
7. paraphrase augmentation；
8. 单请求 pairwise ranking loss；
9. 非方向 consistency regularization；
10. ordinal strength encoding；
11. 自然语言可控推荐方法；
12. post-hoc isotonic projection；
13. MIRROR-Rec 仅损失；
14. MIRROR-Rec 仅结构；
15. 完整 MIRROR-Rec。

具体模型需在实验开始时重新检索和确认版本。

---

## 11. 实验设计

### Study 1：问题是否真实存在

- 比较不同 LLM；
- 比较不同属性；
- 比较不同候选构造；
- 比较不同自然语言措辞；
- 比较不同随机种子。

### Study 2：简单提示能否解决

比较：

- 普通 prompt；
- 结构化 prompt；
- 解释后排序；
- 明确方向约束 prompt。

如果 prompt 已经能够稳定解决，则无需发展复杂方法。

### Study 3：干预对比训练是否有效

比较：

- 普通监督；
- pairwise loss；
- ordinal encoding；
- intervention-contrastive loss；
- 完整 MIRROR-Rec。

### Study 4：结构单调性是否提升泛化

测试：

- 未见措辞；
- 未见物品；
- 未见强度转换；
- 第二领域。

### Study 5：约束投影是否可行

测量：

- 投影前后违反率；
- NDCG 损失；
- 排名位移；
- 求解时间；
- 活跃约束数量。

### Study 6：鲁棒性和反作弊

检查：

- 候选顺序变化；
- 干扰文本；
- paraphrase；
- 固定排序塌缩；
- 小属性差距；
- 次要属性冲突。

### Study 7：消融实验

移除：

- 强度表示；
- 属性条件化模块；
- 干预对比损失；
- 链损失；
- 响应性项；
- 非目标稳定项；
- 推理投影。

---

## 12. 统计分析

所有分析均由脚本完成，包括：

- paired bootstrap 置信区间；
- paired permutation test；
- 非参数效应量；
- Holm 多重比较校正；
- 按模型和属性分层；
- mixed-effects model；
- 对 \(\epsilon\)、\(\rho\)、属性差距阈值的敏感性分析。

不依赖人工挑选样本或人工解释异常值。

---

## 13. Pilot 与 Go/No-Go 标准

### 13.1 Pilot 配置

- 领域：笔记本电脑；
- 4 个属性：价格、重量、续航、存储；
- 200 个基础候选集合；
- 每组 5 个候选；
- 4 个强度等级；
- 每级 3 个自动验证措辞；
- 至少 3 种 LLM 重排序模型；
- 1 个确定性规则基线。

### 13.2 Go 条件

只有满足以下条件才进入完整方法研究：

1. 至少两个模型家族出现非平凡 DVR；
2. 问题在未见措辞中仍然存在；
3. prompt-only 无法将 DVR 降至接近零；
4. 错误不是仅由格式失败导致；
5. 初步跨干预方法能降低 DVR；
6. 响应性不发生明显塌缩；
7. 推荐效用损失在预设范围内；
8. 真实商品对中仍存在该问题。

### 13.3 No-Go 或重设计条件

若出现以下情况，则停止或重设计：

- 强模型几乎不违反；
- 简单结构化 prompt 已完全解决；
- 语言强度无法可靠自动验证；
- 改进严重损害推荐效用；
- 问题只存在于完全合成数据；
- 方法只通过固定排序获得低违反率。

---

## 14. 可复现性与软件架构

### 14.1 仓库结构

```text
mirror-rec/
  configs/
  data_raw/
  data_processed/
  attribute_registry/
  intervention_specs/
  prompts/
  src/
    ingestion/
    validation/
    candidate_generation/
    intervention_generation/
    model_adapters/
    training/
    projection/
    metrics/
    statistics/
  experiments/
  results/
  reports/
  tests/
  containers/
```

### 14.2 可复现性要求

- 数据集使用 checksum；
- 每个样本具有确定 ID；
- prompt 全部版本控制；
- 模型调用全部缓存；
- 随机性尽量固定；
- 环境使用容器；
- 实验由 YAML 配置；
- 图表直接从结果生成；
- 不人工修改 CSV；
- 不删除失败运行；
- 不人工修正输出；
- 所有失败均保留并分类。

### 14.3 自动测试

单元测试和 property-based testing 覆盖：

- 属性方向归一化；
- 干预顺序；
- 候选集合不变性；
- eligible pair；
- 指标实现；
- 投影可行性；
- 数据泄漏；
- 输出解析；
- 固定 seed 可复现性。

---

## 15. 伦理与数据治理

第一阶段只使用：

- 公共物品元数据；
- 合成或模板化用户请求；
- 非敏感属性；
- 非安全关键领域。

不需要：

- 私有用户日志；
- 受保护属性；
- 真人受试者；
- 医疗、金融或法律推荐。

研究关注的是算法方向一致性，不判断用户偏好是否“正确”。

---

## 16. 风险与缓解措施

### 风险 1：简单提示即可解决

通过 Pilot 和 Go/No-Go 标准尽早验证。

### 风险 2：自然语言强度存在歧义

先生成结构化规格，再生成语言，并使用多解析器 round-trip 自动验证。

### 风险 3：并非所有现实决策都应全局单调

只对受控、单属性、eligible pair 施加关系，不扩展为全局规范性结论。

### 风险 4：模型通过固定排序作弊

同时报告响应率、标准推荐性能和真实权衡集合表现。

### 风险 5：合成数据缺乏现实性

同时使用：

- 完全控制合成对；
- 匹配真实物品对；
- 权衡候选集合；
- 第二领域验证。

### 风险 6：方法组件本身已有

不把 hinge loss、单调参数化或投影本身当成创新。创新聚焦于：

\[
\text{有序自然语言干预}
+
\text{跨干预物品对排序差}
+
\text{方向优化}
\]

### 风险 7：研究过程中出现高度相似新论文

建立自动月度文献检查，覆盖：

- arXiv；
- ACM Digital Library；
- DBLP；
- OpenReview；
- ACL Anthology。

---

## 17. 要求产物

1. **概念贡献：** 定义偏好干预—响应一致性；
2. **形式化贡献：** 定义成对和链级方向关系；
3. **Benchmark 贡献：** 构建完全自动验证的数据集；
4. **方法贡献：** 提出干预对比排序优化；
5. **结构贡献：** 属性条件化单调得分；
6. **推理贡献：** 约束排序投影；
7. **实证贡献：** 系统分析不同模型、属性和措辞；
8. **工程贡献：** 提供无人工验证的可复现实验系统。

---



## 18. 预注册执行规范

本节用于让执行方在无需人工验证的条件下执行、复现并迭代完善本项目。

### 18.1 不可修改的研究契约

执行方必须保持：

1. eligible intervention chain 的定义；
2. 干预链内候选集合不变；
3. ground truth 来自结构化属性；
4. 指标实现确定；
5. 训练、验证和测试隔离；
6. 失败输出完整保留；
7. 主 oracle 不使用 LLM-as-a-judge；
8. 不人工修复标签或输出；
9. 不选择性删除负结果；
10. 所有方法修改先进行撞车检查。

### 18.2 必须输出的机器可读文件

```text
literature_matrix.csv
attribute_registry.yaml
catalog_manifest.json
candidate_sets.jsonl
intervention_specs.jsonl
rendered_requests.jsonl
validation_report.json
model_runs.jsonl
metrics_by_instance.parquet
aggregate_results.csv
statistical_tests.json
experiment_manifest.yaml
reproducibility_report.md
```

### 18.3 自动撞车检查协议

对每个方法修改，搜索以下组合：

- LLM recommendation；
- controllable recommendation；
- natural-language preference；
- preference strength；
- ordinal preference；
- monotonic ranking；
- intervention response；
- metamorphic recommendation；
- counterfactual preference；
- constrained reranking；
- isotonic recommendation；
- directional consistency；
- pairwise margin intervention。

每篇候选论文进入矩阵：

```text
paper
year
problem
input_intervention
item_information
output_relation
training_objective
inference_constraint
dataset
manual_annotation
overlap_with_proposal
```

只有当核心机制未被最接近工作直接覆盖，或方案具有明确实质提升时，才允许继续。

### 18.4 自动决策规则

执行方只有在全部 Go 条件通过后，才能从 Pilot 进入完整实验。

若触发 No-Go 条件，必须生成：

```text
redesign_report.md
```

而不是自动扩大数据量掩盖问题。

### 18.5 自动实验规则

每个实验必须包含：

- 数据版本；
- 模型版本；
- 干预等级；
- 属性阈值；
- 候选集合类型；
- prompt 版本；
- 随机种子；
- 指标；
- 统计方法；
- 预注册假设。

缺失完整配置的实验不进入最终结果。

### 18.6 自动报告规则

所有表格、图和数字必须直接来自 instance-level 输出。

自动生成的文字报告中，每个数字必须可追溯到：

- 某个表格单元格；
- 某个 SQL/数据查询；
- 某个结果文件；
- 某个实验配置。

不得人工修改最终数字。

---

## 19. 终止条件与最终交付

执行方在以下条件全部满足后，才可将本项目标记为完成：

1. 已建立可复现的结构化数据集与属性注册表；
2. 已生成并验证有序自然语言偏好干预链；
3. 已在多个模型家族上完成问题存在性实验；
4. 已验证简单提示方法不能稳定解决问题；
5. 已实现并评估 MIRROR-Rec 的核心组件；
6. 已完成强基线、消融、鲁棒性和统计显著性分析；
7. 已证明降低方向违反不是通过固定排序或牺牲普通推荐效用实现；
8. 已完成至少一个真实物品数据场景；
9. 所有结论均可由 instance-level 结果自动追溯；
10. 已生成完整复现包和自动化研究报告。

最终必须交付：

```text
dataset/
attribute_registry.yaml
candidate_sets.jsonl
intervention_specs.jsonl
validated_requests.jsonl
model_runs.jsonl
metrics_by_instance.parquet
aggregate_results.csv
statistical_tests.json
ablation_results.csv
robustness_results.csv
literature_matrix.csv
experiment_manifest.yaml
reproducibility_report.md
final_research_report.md
```

若任一核心 Go 条件未通过，执行方不得把扩大数据规模视为默认补救措施，而应输出 `redesign_report.md`，明确指出：

- 哪一项假设未成立；
- 哪个实验排除了原始方案；
- 哪些修改仍符合研究边界；
- 修改后的方案是否需要重新进行文献撞车检查。

---

## 参考文献

1. Carroll, M., Foote, A., Feng, K., Williams, M., Dragan, A., Knox, W. B., and Milli, S. (2025). *CTRL-Rec: Controlling Recommender Systems With Natural Language*. arXiv:2510.12742.
2. Jeong, J., Han, D., Hong, S., Kang, W., and Yi, M. Y. (2026). *Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders*. arXiv:2605.10323.
3. Khirbat, M., Ren, Y., Castells, P., and Sanderson, M. (2024). *Metamorphic Evaluation of ChatGPT as a Recommender System*. arXiv:2411.12121.
4. Ouyang, Z., Wen, Q., Zhang, C., Ye, Y., and Vosoughi, S. (2025). *Towards Human-like Preference Profiling in Sequential Recommendation*. arXiv:2506.02261.
5. Wan, M., and McAuley, J. (2018). *Item Recommendation on Monotonic Behavior Chains*. Proceedings of the 12th ACM Conference on Recommender Systems, 86–94.
6. Zhang, J., Xie, R., Hou, Y., Zhao, W. X., Lin, L., and Wen, J.-R. (2023). *Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach*. arXiv:2305.07001.
7. Zhang, R., Yu, T., Shen, Y., Jin, H., Chen, C., and Carin, L. (2020). *Reward Constrained Interactive Recommendation with Natural Language Feedback*. arXiv:2005.01618.
8. Peng, Q. et al. (2025). *A Survey on LLM-powered Agents for Recommender Systems*. arXiv:2502.10050.
9. Zhang, Y. et al. (2025). *A Survey of Large Language Model Empowered Agents for Recommendation and Search*. arXiv:2503.05659.
