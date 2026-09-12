"""由统一内容源生成完整Markdown和排版PDF；不改算法或实验结果。"""

from __future__ import annotations
import os
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from report_content import pages
from generate_figures import FORMULAS

HERE = Path(__file__).resolve().parent
STEM = "第一问成果汇总_排序队列完整版"


def find_font():
    candidates = [
        os.environ.get("Q1_REPORT_FONT", ""),
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise RuntimeError("需要中文TrueType字体；请设置Q1_REPORT_FONT为字体路径")


def build():
    content = pages()
    md = []
    pdfmetrics.registerFont(TTFont("ChineseReport", find_font(), subfontIndex=0))
    body = ParagraphStyle(
        "Body",
        fontName="ChineseReport",
        fontSize=10.6,
        leading=17.3,
        spaceAfter=9,
        wordWrap="CJK",
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontSize=18,
        leading=25,
        spaceAfter=16,
        textColor=colors.black,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=9,
    )
    table_style = ParagraphStyle(
        "TableText", parent=body, fontSize=9, leading=13, spaceAfter=0
    )
    code_style = ParagraphStyle(
        "Code", parent=body, fontSize=9, leading=14, spaceAfter=8, leftIndent=8
    )
    story = []
    for number, (heading, blocks) in enumerate(content, 1):
        if number > 1:
            story.append(PageBreak())
        story.append(Paragraph(escape(f"{number}  {heading}"), title))
        md.append(("# " if number == 1 else "## ") + heading + "\n")
        for kind, value in blocks:
            if kind == "p":
                story.append(Paragraph(escape(value), body))
                md.append(value + "\n")
            elif kind == "eq":
                image_path = HERE / "formulas" / f"{value}.png"
                w, h = PILImage.open(image_path).size
                ratio = min(15.6 * cm / w, 1.8 * cm / h)
                width = w * ratio
                height = h * ratio
                story.append(Image(str(image_path), width=width, height=height))
                story.append(Spacer(1, 7))
                md.append("$$\n" + FORMULAS[value] + "\n$$\n")
            elif kind == "fig":
                name, label, maxheight = value
                image_path = HERE / "figures" / f"{name}.png"
                w, h = PILImage.open(image_path).size
                ratio = min(16.6 * cm / w, maxheight * cm / h)
                story.append(
                    KeepTogether(
                        [
                            Image(str(image_path), width=w * ratio, height=h * ratio),
                            Spacer(1, 5),
                            Paragraph(escape(label), caption),
                        ]
                    )
                )
                md.append(f"![{label}](figures/{name}.png)\n")
            elif kind == "table":
                headers, rows = value
                data = [
                    [Paragraph(escape(str(x)), table_style) for x in row]
                    for row in [headers] + rows
                ]
                table = Table(
                    data,
                    colWidths=[16.6 * cm / len(headers)] * len(headers),
                    repeatRows=1,
                    hAlign="LEFT",
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE7EF")),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.45,
                                colors.HexColor("#D9D9D9"),
                            ),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.extend([table, Spacer(1, 10)])
                md.extend(
                    [
                        "| " + " | ".join(headers) + " |",
                        "|" + "|".join(["---"] * len(headers)) + "|",
                    ]
                )
                md.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
                md.append("")
            elif kind == "code":
                story.append(
                    Paragraph(
                        escape(value).replace(" ", "&#160;").replace("\n", "<br/>"),
                        code_style,
                    )
                )
                md.append("```text\n" + value + "\n```\n")
    (HERE / (STEM + ".md")).write_text("\n".join(md), encoding="utf-8")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ChineseReport", 9)
        canvas.setFillColor(colors.HexColor("#59636B"))
        canvas.drawString(1.8 * cm, 1.15 * cm, "2026国赛B题 第一问 | 排序队列完整版")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.15 * cm, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(HERE / (STEM + ".pdf")),
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.8 * cm,
        title="B题第一问排序队列算法与完整验证",
        author="建模项目组",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(STEM + ".md / .pdf generated")


if __name__ == "__main__":
    build()
