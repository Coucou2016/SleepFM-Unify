# SleepFM-Unify 研究报告（完整版）

## 封面说明

本报告对应仓库 `E:\Projects\20260522-SleepFM` 中的 **SleepFM-Unify** 工作：在 SleepFM 多模态睡眠基础模型编码器之上，加入 **共享–私有（Shared–Private）因子分解**、混合对比损失、模态丢弃与可选整夜时序头，并配套导出校验、标签门控与论文实验套件。

- **报告性质：** 工程实现 + 论文框架 + 合成演示验证；**不是**已完成真实 CinC/SHHS 数值的终稿论文。
- **公开仓库：** [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify)（`main`；不含大体积 `data/` / `outputs/`）。
- **图表风格：** SciencePlots + Times New Roman；中文用 SimHei 渲染。
- **诚实约束：** 合成 AUROC≈0.5 仅标为 demo；缺失真实指标一律写 **待补充**；不编造 CinC/SHHS 结果。
- **ChatGPT：** 本轮浏览器 MCP 仍无法稳定 navigate（`tabs new` 后 `viewId` 失效 / `navigate` 报无 tab，共 ≥3 次尝试）；**五轮迭代以 WebSearch / 本地审计 / 文稿修订 / 出图 / 验收替代**，并留好人工粘贴 brief。文献由 WebSearch 独立核验 + nature-skills。

<!--FIGURES-->

---

## 摘要

**研究问题：** SleepFM 的 leave-one-out（LOO，留一模态对比学习：用其余模态预测被留出模态的嵌入）能对齐跨模态共享因素，但可能挤压模态私有信息，并在导联缺失时表现不稳定。

**方法：** SleepFM-Unify 在每模态骨干向量上增加共享头与私有头，默认各 256 维并拼接为 512 维以兼容下游；对比损失只作用在共享空间；辅以正交约束、模态丢弃与缺失重建项，可选夜级时序编码。

**证据状态：** 代码、配置、测试与合成演示已跑通；**PhysioNet CinC 2018 / NSRR SHHS / MESA 的定量结果待补充**（需用户完成数据使用协议下载；本机无 PhysioNet/NSRR 凭证）。

**边界：** 本报告不主张 Unify 在真实 PSG 上已超越 LOO；夜级 `ahi` 使用 **apnea_epoch_rate（呼吸暂停阳性 epoch 率）** 占位，**不是**临床 AASM AHI。

---

## 背景与相关工作

### 睡眠多模态基础模型

睡眠分期（Wake / N1 / N2 / N3 / REM）与睡眠呼吸障碍（SDB）评估依赖多导睡眠图（PSG）。SleepFM（ICML 2024；PMLR 235；arXiv:2405.17766）在 BAS、ECG、呼吸三路上做对比预训练，显示 LOO 优于简单 pairwise。后续 Nature Medicine 工作（doi:10.1038/s41591-025-04133-4；约 58.5 万小时 / 6.5 万受试者量级）将同类 LOO-CL 扩展到疾病风险预测与 SHHS 迁移——**不可把该文 C-Index 直接填进本仓库 CinC/SHHS 表**。Omni-Sleep（arXiv:2607.07720）引入中枢/自主神经（CNS/ANS）层次先验；CIMSleepNet（NeurIPS 2024）针对任意模态缺失做想象补全与对比校准。

### 共享–私有思想与本工作定位

多模态学习中常把表示拆成“跨模态共享”与“模态私有”两支，避免对齐损失抹掉特异信息。Unify 把该思想落到 SleepFM 编码器接口内部：共享支承载 LOO/pairwise，私有支退出对比、仅经正交与下游拼接使用。相对 Omni-Sleep 的生理拓扑与 CIMSleepNet 的缺失想象，Unify 更轻量，便于与官方 LOO 做消融对照。

### 拟模仿的论文结构

按 nature-skills（methods + Nature Machine Intelligence 写作契约）与 SleepFM 实验脊柱：摘要 → 引言 → 相关工作 → 方法 → 实验（基线/消融/缺失/少样本/检索）→ 讨论 → 可复现方法细节。主文数字预算对齐 NMI 风格（引言+结果+讨论约 3500 词量级），本仓库交付 Markdown/HTML/PDF 草稿。

---

## 数据与方法

### 数据（本机盘点）

| 数据 | 含义 | 本机状态 |
|------|------|----------|
| `data/synthetic` | 合成 epoch，CI/演示 | 已有 |
| `data/cinc2018_fixture` | 无 DUA 的 schema 夹具 | 已有 |
| `data/raw/cinc2018` 等 | 真实原始 PSG | **无** |
| CinC 2018 / SHHS / MESA 导出 | 论文预训练评测 | **待补充** |
| PhysioNet / NSRR 凭证 | 环境变量 / netrc | **未设置** |

完整下载步骤见 `docs/DATA_ACCESS.md`。协议自检：`python scripts/protocol_checklist.py`。

`check_data_ready` 区分：

- **raw_ready：** 原始 EDF/WFDB 是否足以开始导出；
- **pretrain_ready / exported_ready：** 是否已有 `index.json`；
- 退出码由 `--stage raw|pretrain` 决定。

### 方法要点（教师讲解版）

1. **编码器：** 每模态 1D EffNet → 512 维骨干。
2. **Unify 头：** `z_shared`、`z_private`；下游默认拼接。
3. **损失：** LOO / pairwise（只看共享）；正交；模态丢弃 + `L_miss`（尊重 `present_mask`）；可选 temporal。
4. **检索：** 共享嵌入；`--max-gallery` 默认 **RNG 子采样**。
5. **诚实门控：** 通道元数据 5/1/3 vs 10/2/7；CinC 标签覆盖；AHI 措辞。

---

## 研究过程（来龙去脉）

### A. 仓库基线

- 已推送公开仓库（见封面 GitHub URL）。
- 文档：`README.md`、`docs/UNIFY.md`、既有 `docs/reports/2026-08-15-*.md`。
- 先前 P0/P1：GRU padding、`L_miss` mask、空 loader 守卫、paper suite temporal、通道/标签门控。

### B. 五轮 Cursor↔顾问协作（本轮）

因 ChatGPT 浏览器 MCP 无法附着标签页，**未伪造 ChatGPT 对话 URL**。改为五轮可审计替代循环（每轮：输入→判断→本地改动→测试计划），全文见 `docs/reports/rounds/`：

| Round | 主题 | 替代手段 | 产出 |
|-------|------|----------|------|
| 1 | 文献 + 架构 | WebSearch 核验四篇核心文献 | 相关工作修订；`DATA_ACCESS.md` |
| 2 | 创新叙事 + NMI 大纲 | nature-writing methods/NMI | 摘要≤150 词气质；贡献/非主张 |
| 3 | 代码/架构审计 | 本地树 ≡ GitHub | P0/P1 清单；`protocol_checklist.py` |
| 4 | 方法/结果章节 | 文稿精修 | 实验矩阵全 待补充；失败模式 |
| 5 | 风险诚实 + 图注 + 验收 | 出图 + 测试 + 推送 | fig 诚实标题；section-19 |

人工可在 ChatGPT 粘贴：`docs/reports/chatgpt-paste-brief-2026-08-16.md`（五段独立 Chat）。

### C. 出图与文稿

- `scripts/plot_unify_figures.py` → `docs/figures/*.png|svg`（Times New Roman + SimHei）。
- `docs/paper/paper.{md,html,pdf}`；本报告 `report.{md,html,pdf}`（HTML 自包含 Base64，无 CDN）。

---

## 结果

### 合成演示（非论文主张）

| 项目 | 结果 |
|------|------|
| 少 epoch Unify 合成预训练 | 损失可记录（图2）；波动大 |
| 下游 AUROC（合成标签） | 约 chance（≈0.5），图3/图5 示意 |
| Gram 诊断 | 图4 前向可视化 |
| CinC / SHHS 指标 | **待补充** |

### 工程验证

以本轮 `run_tests.py` / `smoke_test.py` / `protocol_checklist.py` 日志为准（写入 section-19）。

---

## 讨论

Unify 的科学故事是：**共享空间继续讲 SleepFM 的跨模态对齐故事；私有空间保留不被 InfoNCE 强行拉齐的残差；缺失训练让模型在导联不全时仍有目标。**  
与 Omni-Sleep 的 CNS/ANS 层次、CIMSleepNet 的缺失想象相比，Unify 更“轻量地”挂在 SleepFM 骨干上，便于对照 LOO 基线做消融。

真正决定能否投稿的，是 CinC/SHHS 上的公平对照与不确定度（多种子），而不是合成曲线。

---

## 结论

1. SleepFM-Unify 的方法与工程接口已在本仓库落地。  
2. 论文框架与 SciencePlots 图已再成熟一版；真实数据数字 **待补充**。  
3. 五轮协作已文档化（ChatGPT MCP 阻断 → 替代循环 + 粘贴 brief）。  
4. 评价链路强调不夸大：标签门控、通道门控、AHI 措辞、RNG gallery。  

---

## 局限性

- 无真实 CinC/SHHS/MESA 训练与评测数字。  
- ChatGPT 顾问环本轮仍因浏览器 MCP 中断（可人工粘贴 brief）。  
- 合成指标接近随机，不可写入摘要作为主结果。  
- 夜级 AHI 为占位定义。  
- 大体积数据与 checkpoint 未入库（故意排除）。  

---

## 术语表

| 术语 | 展开与深解释 |
|------|----------------|
| PSG | Polysomnography，多导睡眠图：同步记录脑电、眼电、肌电、心电、呼吸等。 |
| BAS | Brain Activity Signals，本仓库对脑电相关通道组的称呼。 |
| ECG | Electrocardiogram，心电图。 |
| LOO / leave-one-out contrastive | 留一模态对比：用其他模态嵌入对齐被留出模态。 |
| InfoNCE | 噪声对比估计损失：拉近正样本对、推开负样本对。 |
| Shared / Private | 共享/私有子空间：共享对齐跨模态，私有保留特异。 |
| Orthogonality loss | 正交损失：抑制共享与私有编码同一信息。 |
| Modality dropout | 训练时随机丢掉某一模态，模拟缺失。 |
| L_miss | 缺失项：用剩余模态预测被丢模态的共享嵌入。 |
| AUROC | ROC 曲线下面积；0.5≈随机，1.0=完美排序。 |
| AUPRC | Precision–Recall 曲线下面积。 |
| Recall@k | 检索：真实配对是否进入前 k。 |
| apnea_epoch_rate | 呼吸暂停阳性 epoch 数 / 记录小时；**非**临床 AHI。 |
| AHI | Apnea–Hypopnea Index；本仓库夜级探针不用真 AHI。 |
| DUA | Data Use Agreement。 |
| CinC 2018 | PhysioNet Challenge 2018，以觉醒为主，分期标签可能不全。 |
| SHHS / MESA | 大型睡眠队列，NSRR 分发。 |
| SciencePlots | Matplotlib 科学绘图样式包。 |

---

## 十九、双代理收尾报告（摘要）

| 项 | 内容 |
|----|------|
| GitHub | https://github.com/Coucou2016/SleepFM-Unify （public；code/docs，非部署） |
| ChatGPT | **MCP-blocked substitutes**（非真实 ChatGPT URL）；五轮见 `docs/reports/rounds/` |
| Paste brief | `chatgpt-paste-brief-2026-08-16.md`（5 chats） |
| nature-skills | writing + figure 路由已用 |
| SciencePlots | fig01–fig06 刷新 |
| 论文 / 报告 | `docs/paper/paper.*`；本文件 + html/pdf |
| 数据 | synthetic + fixture；真实 CinC/SHHS **待补充** |
| 测试 | section-19 记录 |

完整条目：`docs/reports/section-19-final-20260816.md`。
