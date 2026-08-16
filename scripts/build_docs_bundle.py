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
<p><strong>来龙去脉：</strong>本图说明 SleepFM-Unify 如何在既有 SleepFM 编码器之上增加共享/私有头，而不是重写骨干网络。
左侧三路输入 BAS（脑电相关通道组）、ECG（心电图）、Respiratory（呼吸）分别进入 1D EffNet 编码器；中间共享头
<code>z<sup>shared</sup></code> 负责跨模态 InfoNCE 对齐，私有头 <code>z<sup>private</sup></code> 保留模态特异信息且不进入对比损失；
右侧汇总 LOO+Pairwise、正交约束与模态丢弃 + <code>L<sub>miss</sub></code>。</p>
<p><strong>如何阅读：</strong>顺着箭头从左到右：信号 → 编码器 → 双头 → 损失。基线 SleepFM 对应 <code>unify.enabled=false</code>，仅 LOO。</p>
<p><strong>结论边界：</strong>这是架构示意图，不含任何 CinC/SHHS 准确率数字。</p>
""",
    "fig02_loss_curves": """
<p><strong>来龙去脉：</strong>为验证 Unify 混合损失在本地可跑通，我们在合成数据集上做了 5 个 epoch 的短训练，并记录总损失与分项
（LOO / pairwise / orth / miss）。</p>
<p><strong>如何阅读：</strong>左图是 train/val 总损失随 epoch 变化；右图是分项。合成数据标签噪声大，曲线波动属预期。</p>
<p><strong>结论：</strong>仅证明工程可运行与损失可分解记录。<span class="todo">待补充：真实 PSG 上的稳定收敛曲线</span>。禁止把合成损失下降解读为临床性能提升。</p>
""",
    "fig03_ablation_schematic": """
<p><strong>来龙去脉：</strong>消融实验设计用于回答“正交项 / 缺失项 / 时序头各自贡献什么”。在真实数据到位前，本图用接近随机的合成 AUROC（≈0.5）占位，并画 chance 线。</p>
<p><strong>如何阅读每一柱：</strong>LOO 基线、完整 Unify、去掉 orth、去掉 miss、加上 temporal。数值标注带 * 表示 demo only。</p>
<p><strong>结论：</strong>当前不能宣称任何方法优于基线。<span class="todo">待补充：CinC/SHHS 消融表</span>。</p>
""",
    "fig04_orthogonality": """
<p><strong>来龙去脉：</strong>正交损失希望共享与私有子空间不要编码同一组因子。我们在合成 batch 上做前向，取共享/私有嵌入的交叉 Gram 子集可视化。</p>
<p><strong>子图解读：</strong>左：热力图（红蓝表示正负相关）；右：Gram 元素直方图。这是诊断图，不是性能排行榜。</p>
<p><strong>结论：</strong>可用于检查训练是否在推动去相关；真实数据上的定量 orth 统计 <span class="todo">待补充</span>。</p>
""",
    "fig05_modality_dropout": """
<p><strong>来龙去脉：</strong>临床 PSG 常缺导联。Unify 用模态丢弃 + <code>L_miss</code> 训练缺失鲁棒性。本图为合成示意折线，接近 chance。</p>
<p><strong>如何阅读：</strong>横轴是保留/丢弃设定，纵轴是合成下游 AUROC。</p>
<p><strong>结论：</strong>仅展示实验版式。<span class="todo">待补充：真实缺失模式表（对齐 CIMSleepNet 设定）</span>。</p>
""",
    "fig06_pipeline": """
<p><strong>来龙去脉：</strong>从原始 PSG 到论文套件 JSON 的流水线。真实 CinC/SHHS 下载仍属用户 DUA 动作。</p>
<p><strong>如何阅读：</strong>从左到右：Raw → Export/Validate → Pretrain → Downstream/Retrieval → Ablation/Night → Paper suite。</p>
<p><strong>结论：</strong>仓库已具备该流水线脚本；纸面数字仍依赖真实数据。</p>
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
