import shutil
import uuid
import tempfile
import yt_dlp
import re
from pydantic import BaseModel
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db, SessionLocal
from app.models import Document
from app.services.converter import TARGETS, convert_text, image_to_pdf
from app.services.extractor import SUPPORTED, extract_text, IMAGE_EXT

from app.services import index_manager
from app.services.rag_pipeline import generate_with_ollama

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
OUTPUT_DIR = BASE_DIR / "storage" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

Base.metadata.create_all(engine) # creates the table on first run

app = FastAPI(title="Document Scanner")

@app.on_event("startup")
def startup():
    with SessionLocal() as db:
        index_manager.rebuild(db)

def to_dict(d: Document, with_text: bool = False) -> dict:
    data = {
        "id": d.id,
        "original_name": d.original_name,
        "content_type": d.content_type,
        "size_bytes": d.size_bytes,
        "created_at": d.created_at.isoformat(),
    }
    if with_text:
        data["extracted_text"] = d.extracted_text
    return data

def get_or_404(db: Session, doc_id: int) -> Document:
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc

def next_free_id(db: Session) -> int:
    used = set(db.scalars(select(Document.id)).all())
    n=1
    while n in used:
        n += 1
    return n

@app.post("/api/documents")
def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename or " ").suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {sorted(SUPPORTED)}")

    stored_name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / stored_name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = extract_text(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not read this file: {e}")

    doc = Document(
        id=next_free_id(db),
        original_name=file.filename,
        stored_name=stored_name,
        content_type=file.content_type or "",
        size_bytes=path.stat().st_size,
        extracted_text=text,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    index_manager.ingest(db, doc)
    return to_dict(doc, with_text=True)

@app.get("/api/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [to_dict(d) for d in docs]

@app.get("/api/documents/{doc_id}")
def get_documents(doc_id: int, db: Session = Depends(get_db)):
    return to_dict(get_or_404(db, doc_id), with_text=True)

@app.get("/api/documents/{doc_id}/download")
def download_original(doc_id: int, db: Session = Depends(get_db)):
    doc = get_or_404(db, doc_id)
    return FileResponse(UPLOAD_DIR / doc.stored_name, filename=doc.original_name)

@app.get("/api/documents/{doc_id}/convert")
def convert(doc_id: int, to: str, background: BackgroundTasks, db: Session = Depends(get_db)):
    to = to.lower()
    if to not in TARGETS:
        raise HTTPException(400, f"Target must be one of {sorted(TARGETS)}")
    doc = get_or_404(db, doc_id)

    out = OUTPUT_DIR / f"{uuid.uuid4().hex}.{to}"
    src = UPLOAD_DIR / doc.stored_name
    if to == "pdf" and src.suffix.lower() in IMAGE_EXT:
        image_to_pdf(src, out)  # keep the screenshot's real look
    else:
        convert_text(doc.extracted_text, to, out)
    background.add_task(out.unlink, missing_ok=True) # clean up after sending

    return FileResponse(out, filename=f"{Path(doc.original_name).stem}.{to}")

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = get_or_404(db, doc_id)
    index_manager.remove(db, doc_id)
    (UPLOAD_DIR / doc.stored_name).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
    index_manager.rebuild(db)
    return {"deleted": doc_id}

class UrlIn(BaseModel):
    url: str


@app.post("/api/url-to-mp3")
def url_to_mp3(body: UrlIn, background: BackgroundTasks):
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(400, "Please enter a full link starting with https://")

    tmp = Path(tempfile.mkdtemp(dir=OUTPUT_DIR))
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp / "%(title).100B.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "ffmpeg_location": r"C:\Users\Justin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
        "no_color": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([body.url])
        mp3 = next(tmp.glob("*.mp3"))
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(422, f"Could not convert: {e}")

    background.add_task(shutil.rmtree, tmp, True)  # clean up after sending
    return FileResponse(mp3, filename=mp3.name, media_type="audio/mpeg")

ID_PATTERN = re.compile(r"\b(?:doc(?:ument)?|id)\s*#?\s*(\d+)\b|#(\d+)\b", re.I)

def parse_doc_id(question: str):
    m = ID_PATTERN.search(question)
    if not m:
        return None, question
    doc_id = int(m.group(1) or m.group(2))
    cleaned = ID_PATTERN.sub("", question).strip(" :,-") or question
    return doc_id, cleaned

class AskIn(BaseModel):
    question: str

@app.post("/api/ask")
def ask(body: AskIn, db: Session = Depends(get_db)):
    results = index_manager.search(body.question, top_k=5)
    if not results:
        return {"answer": "No documents are indexed yet.", "sources": []}
    try:
        answer = generate_with_ollama(body.question, results)
    except Exception as e:
        raise HTTPException(503, f"LLM unavailable (is Ollama running?): {e}")
    doc_ids = {int(c.doc_id) for c, _ in results}
    docs = db.scalars(select(Document).where(Document.id.in_(doc_ids))).all()
    return {"answer": answer, "sources": [d.original_name for d in docs]}

app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
