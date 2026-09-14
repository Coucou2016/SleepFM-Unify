# SleepFM-Unify 研究报告（完整版）

## 封面说明

本报告对应公开仓库 [SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify) 中的工作：**异构 / 任意缺失 PSG 下的 SleepFM 兼容鲁棒预训练**（共享–私有头是实现工具，**不是**新颖性主张；FOCAL 等已有 shared/private）。配套混合对比损失、模态丢弃、导联 mask、导出校验、标签门控与论文实验套件。

- **报告性质：** 工程实现 + 论文框架 + **真实 CinC 2018（开放训练集子集）CPU-8 实测**；SHHS/MESA 仍缺 NSRR DUA。
- **公开仓库：** [https://github.com/Coucou2016/SleepFM-Unify](https://github.com/Coucou2016/SleepFM-Unify)（`main`；不含大体积 `data/` / `outputs/`）。
- **图表风格：** SciencePlots + Times New Roman；中文用 SimHei 渲染。
- **诚实约束：** 合成 AUROC≈0.5 仅标为 demo；CinC 表内数字来自 `docs/results/cinc2018_cpu8/` 实测 JSON；**不编造** SHHS/MESA。
- **本轮重点：** ChannelAwareMaskedPool 接线、CinC 开放下载/导出/重训、paper suite + 监督基线、文档填表。

<!--FIGURES-->

---

## 摘要

**研究问题：** 真实 PSG 常缺模态或缺导联；SleepFM 的 leave-one-out（LOO）对齐跨模态共享因素，但在不完全 montage 下行为欠定义。

**方法：** SleepFM-Unify 在 SleepFM 编码器上增加 **缺失/异构 PSG 训练栈**（样本级模态丢弃、`present_mask` 感知编码、按原始在场门控的 $\mathcal{L}_{\mathrm{miss}}$、可选 `channel_mask` 导联清零），并以共享–私有头作为工具（对比只看共享；下游可 probe concat/shared/private）。**不主张“发明了 shared-private”。**

**证据状态：** CinC 2018 开放训练子集已下载/导出；LOO+Unify 在 **P0 修复后**于 CPU-8 协议重训并跑通 paper suite（见 `docs/results/cinc2018_cpu8/`）。**SHHS/MESA 仍待 NSRR DUA**（本机无 `NSRR_TOKEN`）。分期 AUROC 在短 CPU 日程下接近随机——按实测如实填表，不当作 clinic SOTA。

**边界：** 本报告不主张 Unify 在真实 PSG 上已超越 LOO；夜级 `ahi` 使用 **apnea_epoch_rate** 占位，**不是**临床 AASM AHI。

---

## 背景与相关工作

### 睡眠多模态基础模型

睡眠分期与 SDB 评估依赖 PSG。SleepFM（ICML 2024；PMLR 235；arXiv:2405.17766）在 BAS、ECG、呼吸三路上做 LOO 对比预训练。后续 Nature Medicine SleepFM（doi:10.1038/s41591-025-04133-4）将 LOO-CL 扩展到疾病风险与 SHHS 迁移——**不可把该文 C-Index 填进本仓库 CinC/SHHS 表**。Omni-Sleep / PhysioOmni 类工作强调生理层次先验；CIMSleepNet（NeurIPS 2024）对缺失模态做想象补全；OSF/SleepBench 类工作强调异构评测标准化。

### 定位（相对 FOCAL / CIMSleepNet / PhysioOmni / SleepFM / SleepBench）

| 工作 | 我们如何相对定位 |
|------|------------------|
| FOCAL 等 shared/private | 已有共享–私有分解；Unify **复用该工具**，不宣称新颖分解理论 |
| CIMSleepNet | 想象缺失模态；Unify 保持 SleepFM LOO 接口 + dropout/mask，无想象解码器 |
| PhysioOmni / Omni-Sleep | 生理拓扑/层次；Unify 不引入新本体 |
| SleepFM ICML 2024 / Nat Med 2026 | 上游 LOO / 规模化疾病风险；Unify 是缺失鲁棒层 |
| OSF / SleepBench | 评测基准；Unify 提供可复现栈 + 诚实门控，不替代基准 |

**一句话：** 主张是 **异构/任意缺失 PSG 下的 SleepFM 风格鲁棒性**，不是“novel shared-private”。

### 拟模仿的论文结构

摘要 → 引言 → 相关工作 → 方法 → 实验（基线/消融/缺失/少样本 mean±95% CI/检索/space probe）→ 讨论 → 可复现细节。主文数字一律 **待补充** 直至真实导出。

---

## 数据与方法

### 数据（本机盘点，2026-09-15）

| 数据 | 含义 | 本机状态 |
|------|------|----------|
| `data/synthetic` | 合成 epoch，CI/演示 | 已有 |
| `data/cinc2018_fixture` | 无 DUA 的 schema 夹具 | 已有 |
| `data/raw/cinc2018` | 真实 CinC 开放训练子集 | **有**（24 例 S3） |
| `data/cinc2018` 导出 | CPU-8 实测协议 | **有**（670 epoch 索引 / 7304 全量导出） |
| SHHS / MESA | NSRR | **无**（需 DUA） |
| 实测指标 JSON | `docs/results/cinc2018_cpu8/` | **有** |

完整下载步骤见 `docs/DATA_ACCESS.md`。协议自检：`python scripts/protocol_checklist.py`。

### 方法要点

1. **编码器：** 每模态 1D EffNet → 512 维骨干。
2. **缺失栈：** sample-wise dropout；mask-aware BN skip；$\mathcal{L}_{\mathrm{miss}}$；`channel_mask` 导联清零（**Done**）；`ChannelAwareMaskedPool` 软注意力（Unify 默认开，**Done**）；轴长可变编码器仍为未来工作。
3. **Unify 头（工具）：** `z_shared` / `z_private`；下游默认拼接；CLI 可 probe shared/private。
4. **Few-shot：** 论文模式 ≥10 次受试者级重复，报告 mean±95% CI；本机 CinC 仅 1 个 train 受试者 → CI 退化为常数（如实记录）。
5. **诚实门控：** 通道元数据；CinC 标签覆盖；AHI 措辞；`assert_paper_isolation` 泄漏即 `RuntimeError`。

---

## 研究过程（来龙去脉）

### A. 仓库基线

- 已推送公开仓库（见封面 GitHub URL）。
- 文档：`README.md`、`docs/UNIFY.md`、既有 `docs/reports/2026-08-15-*.md`。
- 先前 P0/P1：GRU padding、`L_miss` mask、空 loader 守卫、paper suite temporal、通道/标签门控。

### B. 先前五轮文档（历史）

2026-08-16 的五轮替代循环见 `docs/reports/rounds/`（文献 / 大纲 / 审计 / 方法 / 诚实验收）。**本轮（2026-09-14）未再跑 ChatGPT**；聚焦代码正确性后的协议硬化与新颖性改写，并推送 GitHub。

### C. 出图与文稿

- `scripts/plot_unify_figures.py` → `docs/figures/*.png|svg`（Times New Roman + SimHei）。
- `docs/paper/paper.{md,html,pdf}`；本报告 `report.{md,html,pdf}`（HTML 自包含 Base64，无 CDN）。
- 新颖性主张已改为异构/缺失 PSG 鲁棒性（见 `docs/paper/paper.md` Related work）。

---

## 结果

### CinC 2018 CPU-8 实测（真实信号；非合成）

来源：`docs/results/cinc2018_cpu8/measured_compact.json`。设备 CPU；8 受试者；stride 子采样 670 epochs；预训练 5 epoch。

| 项目 | LOO | Unify | 备注 |
|------|-----|-------|------|
| Staging macro AUROC / AUPRC | 0.454 / 0.289 | 0.495 / 0.235 | n_test=60；近 chance |
| Apnea AUROC / AUPRC | **0.662** / 0.454 | 0.436 / 0.189 | 标签门控通过 |
| Retrieval R@10 macro | — | 0.0198≈rand 0.020 | co-presence |
| Few-shot k=1 staging | — | 0.4955±0.0000 | 仅 1 个 train 受试者 |
| Space probe concat/shared/private | — | 0.495 / 0.510 / 0.499 | staging AUROC |
| EffNet supervised | 0.393 | — | 同索引 |
| SeqStagingBaseline | 0.518 | — | CNN+GRU；非 U-Sleep |
| SHHS / MESA | — | — | **DUA 阻塞，空白** |

### 合成演示（工程冒烟，非论文主张）

| 项目 | 结果 |
|------|------|
| 少 epoch Unify 合成预训练 | 损失可记录；仅 CI |
| 下游 AUROC（合成标签） | 约 chance（≈0.5） |

### 工程验证

以本轮 `run_tests.py` / `smoke_test.py` 日志为准。

---

## 讨论

科学故事是：**在 SleepFM LOO 接口上把不完全 montage 训练与评测定义清楚**——mask-aware 编码、缺失目标、导联清零、严格 split isolation、少样本 mean±95% CI。共享–私有头只是让对比损失与下游容量分工的工具（FOCAL 等已有先例），不是核心 novelty。

真正决定能否投稿的，是 CinC/SHHS 上相对 LOO / CIMSleepNet 等强基线的公平对照，而不是合成曲线。

---

## 结论

1. SleepFM-Unify 方法与工程接口已落地；主张为 **异构/缺失 PSG 鲁棒性**。  
2. CinC 开放子集已完成下载→导出→P0 后重训→paper suite；数字见 `docs/results/cinc2018_cpu8/`。  
3. SHHS/MESA 仍缺用户 NSRR DUA；GPU/全量日程可扩展规模。  
4. 评价链路强调不夸大：标签门控、通道门控、AHI 措辞、RNG gallery。  

---

## 局限性

- CinC CPU-8 为小规模实测，分期近随机；不可当作 clinic SOTA。  
- SHHS/MESA 无 DUA，表内空白。  
- 强外部基线（CIMSleepNet、FOCAL port、SleepBench）仍为引用对照，未本地安装运行。  
- `ChannelAwareMaskedPool` 已接线；轴长可变编码器仍为未来工作。  
- 夜级 apnea-positive epoch rate 因每 split 夜数不足记为 N/A。  
- 大体积 raw / `.npy` / checkpoint 未入库（故意排除）。  

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
| 本轮 ChatGPT | **未跑**（聚焦代码/文档/GitHub） |
| 历史五轮 | `docs/reports/rounds/`（2026-08-16） |
| 新颖性 | 异构/缺失 PSG 鲁棒性；非 novel shared-private |
| 论文 / 报告 | `docs/paper/paper.*`；本文件 + html/pdf |
| 数据 | synthetic + fixture；真实 CinC/SHHS **待补充** |
| 协议 | few-shot ≥10 + 95% CI；strict isolation；space probe |

完整条目：`docs/reports/section-19-final-20260816.md`。
