#!/usr/bin/env python
"""Build self-contained paper/report HTML (+ PDF via reportlab) with Base64 figures."""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
PAPER = ROOT / "docs" / "paper"
REPORTS = ROOT / "docs" / "reports"

FIGURES = [
    ("fig01_architecture", "图1 / Fig. 1", "SleepFM-Unify 共享–私有架构示意"),
    ("fig02_loss_curves", "图2 / Fig. 2", "合成演示 Unify 预训练损失曲线"),
    ("fig03_ablation_schematic", "图3 / Fig. 3", "消融示意（合成≈随机，非论文主张）"),
    ("fig04_orthogonality", "图4 / Fig. 4", "共享×私有 Gram 诊断（合成前向）"),
    ("fig05_modality_dropout", "图5 / Fig. 5", "模态缺失鲁棒性示意（合成）"),
    ("fig06_pipeline", "图6 / Fig. 6", "实验流水线"),
]


def b64_png(stem: str) -> str:
    path = FIG / f"{stem}.png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def md_to_simple_html(text: str) -> str:
    """Minimal Markdown → HTML (headings, lists, tables, code, paragraphs)."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    code_lang = ""
    code_buf: list[str] = []
    in_ul = False
    in_table = False
    table_rows: list[list[str]] = []

    def flush_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append('<table class="tbl">')
        for ri, row in enumerate(table_rows):
            tag = "th" if ri == 0 else "td"
            if ri == 1 and all(re.match(r"^:?-+:?$", c.strip()) for c in row):
                continue
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in row) + "</tr>")
        out.append("</table>")
        in_table = False
        table_rows = []

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_ul()
            flush_table()
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_buf = []
            else:
                out.append(f'<pre class="code"><code>{html.escape(chr(10).join(code_buf))}</code></pre>')
                in_code = False
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if line.strip().startswith("|") and line.strip().endswith("|"):
            flush_ul()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            i += 1
            continue
        else:
            flush_table()

        if re.match(r"^#{1,6}\s", line):
            flush_ul()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            aid = re.sub(r"[^\w\u4e00-\u9fff]+", "-", title).strip("-").lower()
            out.append(f'<h{level} id="{aid}">{inline(title)}</h{level}>')
            i += 1
            continue
        if re.match(r"^[-*]\s+", line):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline(re.sub(r'^[-*]\\s+', '', line))}</li>")
            i += 1
            continue
        flush_ul()
        if line.strip() == "":
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr/>")
            i += 1
            continue
        out.append(f"<p>{inline(line)}</p>")
        i += 1
    flush_ul()
    flush_table()
    return "\n".join(out)


CSS = """
:root { --ink:#1a1a1a; --muted:#555; --line:#d0d5dd; --bg:#fafafa; --card:#fff; --accent:#1f4e79; }
* { box-sizing: border-box; }
body { margin:0; font-family: "Times New Roman", "SimHei", "Microsoft YaHei", serif;
  color:var(--ink); background:var(--bg); line-height:1.65; font-size:16px; }
.wrap { max-width:920px; margin:0 auto; padding:28px 22px 80px; background:var(--card);
  box-shadow:0 0 0 1px var(--line); }
.cover { text-align:center; padding:48px 12px 36px; border-bottom:2px solid var(--accent); margin-bottom:28px; }
.cover h1 { font-size:1.85rem; margin:0 0 12px; color:var(--accent); }
.cover .meta { color:var(--muted); font-size:0.95rem; }
nav.toc { background:#f3f6fa; border:1px solid var(--line); padding:16px 20px; margin:20px 0 32px; }
nav.toc h2 { margin-top:0; font-size:1.15rem; }
nav.toc ol { margin:0; padding-left:1.3rem; }
nav.toc a { color:var(--accent); text-decoration:none; }
nav.toc a:hover { text-decoration:underline; }
h1,h2,h3 { color:var(--accent); }
h2 { border-bottom:1px solid var(--line); padding-bottom:6px; margin-top:2.2em; }
figure { margin:28px 0; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; border:1px solid var(--line); }
figcaption { font-size:0.92rem; color:var(--muted); margin-top:10px; text-align:left; }
.explain { background:#f8fafc; border-left:4px solid var(--accent); padding:12px 16px; margin:10px 0 22px; }
.todo { background:#fff8e6; border:1px solid #e6c86a; padding:8px 12px; display:inline-block; }
table.tbl { border-collapse:collapse; width:100%; margin:16px 0; font-size:0.92rem; }
table.tbl th, table.tbl td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
table.tbl th { background:#eef3f8; }
pre.code { background:#111; color:#f5f5f5; padding:14px; overflow:auto; font-size:0.85rem; }
code { font-family: Consolas, "Courier New", monospace; font-size:0.9em; }
.footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:0.85rem; }
@media print { body { background:#fff; } .wrap { box-shadow:none; } }
"""


def figure_block(stem: str, label: str, title: str, long_explain: str) -> str:
    return f"""
<figure id="{stem}">
  <img src="{b64_png(stem)}" alt="{html.escape(title)}" />
  <figcaption><strong>{html.escape(label)}. {html.escape(title)}</strong></figcaption>
</figure>
<div class="explain">{long_explain}</div>
"""


EXPLAINS = {
    "fig01_architecture": """
<p><strong>来龙去脉（为什么画这张）：</strong>审稿人/合作者首先需要看清 Unify <em>挂在哪里</em>。
SleepFM 已有三路 1D EffNet；若直接改骨干，就无法与官方 LOO 基线做干净对照。
因此我们只在骨干向量后接线性共享/私有头，把“对齐什么 / 保留什么”显式拆开。</p>
<p><strong>图中元素：</strong>左列 BAS（脑电相关通道组）、ECG、Respiratory → 中列同结构编码器 →
共享头 <code>z<sup>shared</sup></code>（进 LOO/pairwise InfoNCE）与私有头 <code>z<sup>private</sup></code>（退出对比）→
右列三项目标：跨模态对齐、正交去相关、模态丢弃 + <code>L<sub>miss</sub></code>。</p>
<p><strong>如何阅读：</strong>沿箭头从左到右。基线对应 <code>unify.enabled=false</code>（无双头、仅 LOO）。
下游默认拼接 shared‖private；检索默认只用 shared，以保持与 SleepFM 检索语义可比。</p>
<p><strong>结论边界：</strong>架构示意，<strong>不含</strong> CinC/SHHS 准确率。<span class="todo">待补充：真实数据上的参数量/FLOPs 表</span>。</p>
""",
    "fig02_loss_curves": """
<p><strong>来龙去脉：</strong>混合损失是否“能训练、可分解记录”是工程验收点。我们在合成数据上跑 5 个 epoch 的 Unify 预训练，
把总损失与 LOO / pairwise / orth / miss 分项写入 <code>fig02_loss_history.json</code>，再画成曲线。</p>
<p><strong>如何阅读：</strong>左：train/val 总损失随 epoch；右：分项。合成标签噪声大，短程波动是预期现象，
<strong>不是</strong>临床收敛证据。若某分项恒为 0，应检查权重配置（如 <code>loss_weights.*</code>）。</p>
<p><strong>结论：</strong>证明损失接口可运行。<span class="todo">待补充：真实 PSG 上多种子稳定收敛曲线与学习率日程</span>。
禁止把合成损失下降写成方法优越性。</p>
""",
    "fig03_ablation_schematic": """
<p><strong>来龙去脉：</strong>投稿级主张需要消融：完整 Unify vs 去掉 orth / 去掉 miss / 加 temporal，以及相对 LOO 基线。
在 DUA 数据到位前，用合成下游 AUROC（≈0.5，chance 附近）占位版式，并画水平 chance 线，避免读者把柱高误读为真实增益。</p>
<p><strong>如何阅读每一柱：</strong>LOO · Unify-full · −orth · −miss · +temporal。带 * 的数值 = demo only。
柱高接近 0.5 表示<strong>当前没有可发表的排序能力</strong>，只说明评测脚本与图模板就绪。</p>
<p><strong>结论：</strong>不可宣称任何变体优于基线。<span class="todo">待补充：CinC/SHHS 消融表（均值±标准差，多种子）</span>。</p>
""",
    "fig04_orthogonality": """
<p><strong>来龙去脉：</strong>正交项的科学意图是：共享因子被 InfoNCE 拉齐后，私有头不应再重复同一组方向。
我们在合成 batch 上前向，取共享×私有交叉 Gram 的子块做诊断可视化（实现为行 L2 后的平方均值，细节见 <code>docs/UNIFY.md</code>）。</p>
<p><strong>子图解读：</strong>左热力图看方向耦合结构；右直方图看 Gram 元素分布是否被推向 0 附近。
这是<strong>训练诊断</strong>，不是下游 AUROC 排行榜；热图“好看”≠分期更准。</p>
<p><strong>结论：</strong>可用于检查去相关是否在起作用。<span class="todo">待补充：真实数据上 orth 统计随 epoch 曲线与消融对照</span>。</p>
""",
    "fig05_modality_dropout": """
<p><strong>来龙去脉：</strong>临床 PSG 常缺导联；CIMSleepNet 等用想象补全处理缺失，而 Unify 采用更轻的
<strong>训练期模态丢弃 + <code>L_miss</code></strong>（剩余共享均值 vs 被丢模态共享，且尊重 <code>present_mask</code>）。
本图用合成 AUROC 折线固定版式，数值接近 chance。</p>
<p><strong>如何阅读：</strong>横轴 = 保留/丢弃设定（如全模态、丢 ECG、丢呼吸等），纵轴 = 合成下游 AUROC。
折线贴近 0.5 时，只能说明脚本能在缺失设定下跑完评测。</p>
<p><strong>结论：</strong>版式就绪，非鲁棒性主张。<span class="todo">待补充：真实缺失模式表（可对齐 CIMSleepNet 的 missing patterns）</span>。</p>
""",
    "fig06_pipeline": """
<p><strong>来龙去脉：</strong>论文数字可信的前提是流水线门控正确：原始文件存在 ≠ 已导出可训；
标签覆盖不足时不能宣称分期/SDB。本图把 Raw→Export/Validate→Pretrain→Downstream/Retrieval→Ablation/Night→Paper suite
串成一条可审计路径，对应 <code>check_data_ready --stage raw|pretrain</code> 与 <code>run_paper_suite</code>。</p>
<p><strong>如何阅读：</strong>从左到右；任一菱形门控失败应停止宣称指标。真实 CinC/SHHS 下载仍是用户 DUA 动作，
仓库只提供 exporter / channel tables / fixture。</p>
<p><strong>结论：</strong>代码与脚本已公开（GitHub）；纸面 CinC/SHHS 数字仍 <span class="todo">待补充</span>。</p>
""",
}


def wrap_document(title: str, body: str, cover_html: str, toc: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{cover_html}
<nav class="toc"><h2>目录 / Table of Contents</h2>{toc}</nav>
{body}
<div class="footer">Generated locally for SleepFM-Unify. Figures embedded as Base64 PNG (SciencePlots). No CDN scripts. CinC/SHHS metrics marked 待补充 when missing.</div>
</div>
</body>
</html>
"""


def build_paper_html() -> Path:
    md = (PAPER / "paper.md").read_text(encoding="utf-8")
    body = md_to_simple_html(md)
    figs = "\n".join(
        figure_block(stem, lab, title, EXPLAINS[stem]) for stem, lab, title in FIGURES
    )
    body += "<h2 id='figures-embedded'>Embedded figures</h2>\n" + figs
    toc = """
<ol>
<li><a href="#abstract">Abstract</a></li>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#related-work">Related work</a></li>
<li><a href="#method">Method</a></li>
<li><a href="#experiments">Experiments</a></li>
<li><a href="#discussion">Discussion</a></li>
<li><a href="#methods-reproducibility">Methods</a></li>
<li><a href="#figures-embedded">Figures</a></li>
</ol>"""
    cover = """
<div class="cover">
  <h1>SleepFM-Unify</h1>
  <p>Shared–Private Factorization for Multimodal Sleep Foundation Models</p>
  <p class="meta">Draft paper · SciencePlots figures · synthetic demos only for metrics · CinC/SHHS 待补充</p>
</div>"""
    html_doc = wrap_document("SleepFM-Unify Paper Draft", body, cover, toc)
    out = PAPER / "paper.html"
    out.write_text(html_doc, encoding="utf-8")
    return out


def build_report_html(report_md: Path) -> Path:
    md = report_md.read_text(encoding="utf-8")
    body = md_to_simple_html(md)
    figs = "\n".join(
        figure_block(stem, lab, title, EXPLAINS[stem]) for stem, lab, title in FIGURES
    )
    # Insert figures section if placeholder exists
    if "<!--FIGURES-->" in body:
        body = body.replace("<!--FIGURES-->", figs)
    else:
        body += "<h2 id='figures-detail'>图表详解</h2>\n" + figs
    toc = """
<ol>
<li><a href="#封面说明">封面说明</a></li>
<li><a href="#摘要">摘要</a></li>
<li><a href="#背景与相关工作">背景与相关工作</a></li>
<li><a href="#数据与方法">数据与方法</a></li>
<li><a href="#研究过程">研究过程</a></li>
<li><a href="#结果">结果</a></li>
<li><a href="#讨论">讨论</a></li>
<li><a href="#结论">结论</a></li>
<li><a href="#局限性">局限性</a></li>
<li><a href="#figures-detail">图表详解</a></li>
<li><a href="#术语表">术语表</a></li>
<li><a href="#十九-双代理收尾报告">十九、双代理收尾</a></li>
</ol>"""
    cover = """
<div class="cover">
  <h1>SleepFM-Unify 研究报告</h1>
  <p>共享–私有统一预训练 · 工程实现与论文框架</p>
  <p class="meta">2026-08-16 · 自包含单文件 HTML（内联 CSS + Base64 图）· 教师可读学术语气</p>
</div>"""
    html_doc = wrap_document("SleepFM-Unify 研究报告", body, cover, toc)
    out = REPORTS / "report.html"
    out.write_text(html_doc, encoding="utf-8")
    return out


def md_to_pdf(md_path: Path, pdf_path: Path, title: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Prefer Chinese-capable fonts on Windows
    font_name = "Helvetica"
    for name, path in [
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
        ("YaHei", r"C:\Windows\Fonts\msyh.ttc"),
    ]:
        p = Path(path)
        if p.is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p), subfontIndex=0))
                font_name = name
                break
            except Exception:
                continue

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCN", fontName=font_name, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="H1CN", fontName=font_name, fontSize=16, leading=20, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2CN", fontName=font_name, fontSize=13, leading=17, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="CapCN", fontName=font_name, fontSize=9, leading=12, textColor="#444444"))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title,
    )
    story = [Paragraph(html.escape(title), styles["H1CN"]), Spacer(1, 12)]
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:]), styles["H1CN"]))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:]), styles["H2CN"]))
        elif line.startswith("### "):
            story.append(Paragraph(html.escape(line[4:]), styles["H2CN"]))
        elif line.strip() == "---":
            story.append(Spacer(1, 8))
        elif line.strip():
            story.append(Paragraph(html.escape(line), styles["BodyCN"]))
            story.append(Spacer(1, 4))
    story.append(PageBreak())
    story.append(Paragraph("Figures (SciencePlots)", styles["H1CN"]))
    for stem, lab, ftitle in FIGURES:
        png = FIG / f"{stem}.png"
        if png.is_file():
            story.append(Spacer(1, 8))
            story.append(Image(str(png), width=16 * cm, height=7.5 * cm, kind="proportional"))
            story.append(Paragraph(html.escape(f"{lab}. {ftitle}"), styles["CapCN"]))
    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-md", type=str, default=str(REPORTS / "report.md"))
    args = parser.parse_args()
    PAPER.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    p_html = build_paper_html()
    r_html = build_report_html(Path(args.report_md))
    md_to_pdf(PAPER / "paper.md", PAPER / "paper.pdf", "SleepFM-Unify Paper Draft")
    md_to_pdf(Path(args.report_md), REPORTS / "report.pdf", "SleepFM-Unify Research Report")
    print("wrote", p_html)
    print("wrote", r_html)
    print("wrote", PAPER / "paper.pdf")
    print("wrote", REPORTS / "report.pdf")


if __name__ == "__main__":
    main()
