from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
from xml.sax.saxutils import escape

from docx import Document as DocxDocument
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

TARGETS = {"pdf", "docx", "txt", "png", "jpg"}

def _to_pdf(text: str, out: Path) -> None:
    style = getSampleStyleSheet()["Normal"]
    story = [Paragraph(escape(line) or "&nbsp;", style) for line in text.split("\n")]
    SimpleDocTemplate(str(out), pagesize=A4).build(story)

def _to_docx(text: str, out: Path) -> None:
    doc = DocxDocument()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(out)

def _render_image(text: str) -> Image.Image:
    width, margin, font_size, max_lines = 1240, 60, 24, 200
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=70) or [""])
    lines = lines[:max_lines]  # keeps the image from getting huge

    line_h = font_size + 8
    height = max(400, margin * 2 + line_h * len(lines))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_h
    return img

def image_to_pdf(src: Path, out: Path) -> None:
    from PIL import Image
    with Image.open(src) as img:
        img.convert("RGB").save(out, "PDF", resolution=150.0)

def convert_text(text: str, target: str, out: Path) -> None:
    if target == "pdf":
        _to_pdf(text, out)
    elif target == "docx":
        _to_docx(text, out)
    elif target == "txt":
        out.write_text(text, encoding="utf-8")
    elif target == "png":
        _render_image(text).save(out, "PNG")
    elif target == "jpg":
        _render_image(text).save(out, "JPEG", quality=90)
    else:
        raise ValueError(f"Unsupported target: {target}")