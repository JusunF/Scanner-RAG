import os
from pathlib import Path

import fitz # PyMuPDF
import pytesseract
from dotenv import load_dotenv
from docx import Document as DocxDocument
from PIL import Image

load_dotenv()

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
OCR_LANG = os.getenv("OCR_LANG", "eng")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED = {".pdf", ".docx", ".txt"} | IMAGE_EXT

def _ocr(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang=OCR_LANG)

def _from_pdf(path: Path) -> str:
    pages = []
    with fitz.open(path) as pdf:
        for page in pdf:
            text = page.get_text().strip()
            if len(text) < 20:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text =_ocr(img)
            pages.append(text)
    return "\n\n".join(pages)

def _from_docx(path: Path) -> str:
    return "\n".join(p.text for p in DocxDocument(path).paragraphs)

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _from_pdf(path)
    if ext == ".docx":
        return _from_docx(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in IMAGE_EXT:
        with Image.open(path) as img:
            return _ocr(img.convert("RGB"))
    raise ValueError(f"Unsupported file type: {ext}")