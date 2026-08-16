# SleepFM-Unify 研究报告（完整版）

## 封面说明

本报告对应仓库 `E:\Projects\20260522-SleepFM` 中的 **SleepFM-Unify** 工作：在 SleepFM 多模态睡眠基础模型编码器之上，加入 **共享–私有（Shared–Private）因子分解**、混合对比损失、模态丢弃与可选整夜时序头，并配套导出校验、标签门控与论文实验套件。

- **报告性质：** 工程实现 + 论文框架 + 合成演示验证；**不是**已完成真实 CinC/SHHS 数值的终稿论文。
- **图表风格：** SciencePlots + Times New Roman；中文用 SimHei 渲染。
- **诚实约束：** 合成 AUROC≈0.5 仅标为 demo；缺失真实指标一律写 **待补充**；不编造 CinC/SHHS 结果。
- **ChatGPT：** 浏览器 MCP 无法导航，无会话 URL；文献架构由 WebSearch + nature-skills 补齐（见 `docs/reports/chatgpt-consultation-2026-08-16.md`）。

<!--FIGURES-->

---

## 摘要

**研究问题：** SleepFM 的 leave-one-out（LOO，留一模态对比学习：用其余模态预测被留出模态的嵌入）能对齐跨模态共享因素，但可能挤压模态私有信息，并在导联缺失时表现不稳定。

**方法：** SleepFM-Unify 在每模态骨干向量上增加共享头与私有头，默认各 256 维并拼接为 512 维以兼容下游；对比损失只作用在共享空间；辅以正交约束、模态丢弃与缺失重建项，可选夜级时序编码。

**证据状态：** 代码、配置、测试与合成演示已跑通；**PhysioNet CinC 2018 / NSRR SHHS / MESA 的定量结果待补充**（需用户完成数据使用协议下载）。

**边界：** 本报告不主张 Unify 在真实 PSG 上已超越 LOO；夜级 `ahi` 使用 **apnea_epoch_rate（呼吸暂停阳性 epoch 率）** 占位，**不是**临床 AASM AHI。

---

## 背景与相关工作

### 睡眠多模态基础模型

睡眠分期（Wake / N1 / N2 / N3 / REM）与睡眠呼吸障碍（SDB）评估依赖多导睡眠图（PSG）。SleepFM（ICML 2024）在 BAS、ECG、呼吸三路上做对比预训练，显示 LOO 优于简单 pairwise。后续 Nature Medicine 工作将同类表征扩展到疾病风险预测。Omni-Sleep 引入中枢/自主神经（CNS/ANS）层次先验；CIMSleepNet 针对任意模态缺失做想象补全与对比校准。

### 共享–私有思想

多模态学习中常把表示拆成“跨模态共享”与“模态私有”两支，避免对齐损失抹掉特异信息。Unify 把该思想落到 SleepFM 编码器接口内部：共享支承载 LOO/pairwise，私有支退出对比、仅经正交与下游拼接使用。

### 拟模仿的论文结构

按 nature-skills（methods + Nature Machine Intelligence 写作契约）与 SleepFM 实验脊柱：摘要 → 引言 → 相关工作 → 方法 → 实验（基线/消融/缺失/少样本/检索）→ 讨论 → 可复现方法细节。主文数字预算对齐 NMI 风格（引言+结果+讨论约 3500 词量级），本仓库先交付 Markdown/HTML/PDF 草稿。

---

## 数据与方法

### 数据

| 数据 | 含义 | 本机状态 |
|------|------|----------|
| `data/synthetic` | 合成 epoch，CI/演示 | 已有 |
| `data/cinc2018_fixture` | 无 DUA 的 schema 夹具 | 已有 |
| CinC 2018 真数据 | PhysioNet | **待补充** |
| SHHS / MESA | NSRR DUA | **待补充** |

`check_data_ready` 现区分：

- **raw_ready：** 原始 EDF/WFDB 文件是否在，足以开始导出；
- **pretrain_ready / exported_ready：** 是否已有 `index.json`，足以 validate/pretrain；
- 退出码由 `--stage raw|pretrain` 决定，避免“exit 0 = 可以宣称论文指标”的误解。

### 方法要点（教师讲解版）

1. **编码器：** 每模态 1D EffNet → 512 维骨干。
2. **Unify 头：** `z_shared`、`z_private`；下游默认拼接。
3. **损失：**  
   - LOO / pairwise InfoNCE（只看共享）；  
   - 正交：共享与私有 Gram 平方均值；  
   - 模态丢弃 + `L_miss`（剩余共享均值 vs 被丢模态共享，且尊重 `present_mask`）；  
   - 可选 temporal。
4. **检索：** 用共享嵌入；`--max-gallery` 默认 **RNG 子采样**（可复现 seed），避免 loader 前缀偏差。
5. **诚实门控：** 通道元数据 5/1/3 vs 10/2/7 默认失败；CinC 标签覆盖不足时不宣称分期/SDB。

---

## 研究过程

### A. 仓库基线

- 非 git 仓库（无 HEAD）。
- 文档：`README.md`、`docs/UNIFY.md`、既有 `docs/reports/2026-08-15-*.md`。
- 先前 P0/P1：GRU padding、`L_miss` mask、空 loader 守卫、paper suite temporal、通道/标签门控。

### B. 本轮 P2 清理

1. 规范化 `sleepfm/eval/retrieval.py`（去除双换行损坏）。
2. 完成 `limit_gallery(seed, mode="rng"|"prefix")`，并接入 `eval_retrieval.py` / `run_paper_suite.py`。
3. `check_data_ready` 明确 raw vs pretrain 命名与测试。
4. 夜级 AHI→apnea_epoch_rate 文案（既有）保持。

### C. 文献与写作技能

- nature-skills 已安装于 `~/.cursor/skills/nature-skills`。
- ChatGPT 导航失败 → WebSearch 文献笔记写入 consultation 文件。
- SciencePlots 已安装并可 `import scienceplots`。

### D. 出图与文稿

- `scripts/plot_unify_figures.py` → `docs/figures/*.png|svg`。
- `docs/paper/paper.md` + HTML/PDF；`docs/reports/report.*`。

---

## 结果

### 合成演示（非论文主张）

| 项目 | 结果 |
|------|------|
| 5-epoch Unify 合成预训练 | 损失可记录（见 图2）；波动大 |
| 下游 AUROC（合成标签） | 约 chance（≈0.5），图3/图5 示意 |
| Gram 诊断 | 图4 前向可视化 |
| CinC / SHHS 指标 | **待补充** |

### 工程验证（预期在测试节更新）

单元测试覆盖检索 RNG、`check_data_ready` 标志、既有 Unify/night/channel gates。具体通过数以本轮 `run_tests.py` / `smoke_test.py` 日志为准。

---

## 讨论

Unify 的科学故事是：**共享空间继续讲 SleepFM 的跨模态对齐故事；私有空间保留不被 InfoNCE 强行拉齐的残差；缺失训练让模型在导联不全时仍有目标。**  
与 Omni-Sleep 的 CNS/ANS 层次、CIMSleepNet 的缺失想象相比，Unify 更“轻量地”挂在 SleepFM 骨干上，便于对照 LOO 基线做消融。

真正决定能否投稿的，是 CinC/SHHS 上的公平对照与不确定度（多种子），而不是合成曲线。

---

## 结论

1. SleepFM-Unify 的方法与工程接口已在本仓库落地。  
2. 论文框架与 SciencePlots 图已生成；真实数据数字 **待补充**。  
3. 评价链路强调不夸大：标签门控、通道门控、AHI 措辞、RNG gallery。  

---

## 局限性

- 无真实 CinC/SHHS/MESA 训练与评测数字。  
- ChatGPT 顾问环断裂（浏览器 MCP）。  
- 合成指标接近随机，不可写入摘要作为主结果。  
- 夜级 AHI 为占位定义。  
- 非 git，版本以报告与 ZIP SHA 追溯。  

---

## 术语表

| 术语 | 展开与深解释 |
|------|----------------|
| PSG | Polysomnography，多导睡眠图：同步记录脑电、眼电、肌电、心电、呼吸等。 |
| BAS | Brain Activity Signals，本仓库对脑电相关通道组的称呼（SleepFM 原文亦用 sleep stages 通道组概念）。 |
| ECG | Electrocardiogram，心电图。 |
| LOO / leave-one-out contrastive | 留一模态对比：用其他模态嵌入对齐被留出模态，提升多模态鲁棒性。 |
| InfoNCE | 噪声对比估计损失：拉近正样本对、推开负样本对。 |
| Shared / Private | 共享/私有子空间：共享对齐跨模态，私有保留特异。 |
| Orthogonality loss | 正交损失：抑制共享与私有编码同一信息。 |
| Modality dropout | 训练时随机丢掉某一模态，模拟缺失。 |
| L_miss | 缺失项：用剩余模态预测被丢模态的共享嵌入。 |
| AUROC | ROC 曲线下面积；0.5≈随机，1.0=完美排序。 |
| AUPRC | Precision–Recall 曲线下面积，类别不平衡时更稳。 |
| Recall@k | 检索：真实配对是否进入前 k。 |
| apnea_epoch_rate | 呼吸暂停阳性 epoch 数 / 记录小时；**非**临床 AHI。 |
| AHI | Apnea–Hypopnea Index，临床事件/小时；本仓库夜级探针不用真 AHI。 |
| DUA | Data Use Agreement，数据使用协议（NSRR 等）。 |
| CinC 2018 | PhysioNet Challenge 2018，以觉醒为主，分期标签可能不全。 |
| SHHS / MESA | 大型睡眠队列，NSRR 分发。 |
| SciencePlots | Matplotlib 科学绘图样式包。 |

---

## 十九、双代理收尾报告

见文末专节（与交付清单同步更新）。完整条目亦写入同目录 `section-19-final.md`（若生成）及本文件下一节。

### 十九要点（摘要）

| 项 | 内容 |
|----|------|
| ChatGPT URL | 无（MCP navigate 失败） |
| nature-skills | 已找到并遵循写作/出图路由 |
| SciencePlots | 已安装；图已出（fig01–fig06） |
| 论文 | `docs/paper/paper.{md,html,pdf}` |
| 报告 | `docs/reports/report.{md,html,pdf}`（HTML 自包含 Base64） |
| 代码清理 | retrieval 规范化；gallery RNG；check_data_ready 阶段命名 |
| 测试 | **68 passed**；smoke **PASSED** |
| Git | 未 commit / push / PR |
| 真实指标 | 待补充 |

完整条目：`docs/reports/section-19-final.md`。
