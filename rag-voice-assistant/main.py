import os
import re
import shutil
import tempfile
import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import edge_tts
from ingest import ingest_document, collection
from retriever import retrieve
from llm import generate_response
from config import DATA_DIR

app = FastAPI(title="RAG Voice Assistant", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(DATA_DIR, exist_ok=True)

# Load whisper model once at startup
# "small" (vs "base") is materially more accurate on Hindi and code-switched Hindi-English speech
whisper_model = whisper.load_model("small")


DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def pick_voice(text: str) -> str:
    """Choose a Hindi or Indian-English neural voice based on the script of the text."""
    return "hi-IN-MadhurNeural" if DEVANAGARI_RE.search(text) else "en-IN-NeerjaNeural"


# --- Request Models ---
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG Voice Assistant is running"}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    district: str = Form("general"),
    category: str = Form("general"),
    topic: str = Form("general")
):
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    metadata = {
        "source": file.filename,
        "district": district,
        "category": category,
        "topic": topic
    }

    try:
        num_chunks = ingest_document(file_path, metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": "Document ingested successfully",
        "file": file.filename,
        "chunks_stored": num_chunks,
        "metadata": metadata
    }


@app.post("/retrieve")
def retrieve_chunks(request: RetrieveRequest):
    chunks = retrieve(request.query, request.top_k)
    return {
        "query": request.query,
        "results": chunks,
        "total": len(chunks)
    }


@app.post("/query")
def query(request: QueryRequest):
    chunks = retrieve(request.query, request.top_k)
    response = generate_response(request.query, chunks)
    return {
        "query": request.query,
        "answer": response["answer"],
        "sources": response["sources"],
        "chunks_used": response["chunks_used"]
    }


@app.post("/voice-query")
async def voice_query(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Auto-detect language: forcing a single language mis-decodes mixed-language speech
    # (e.g. English words get force-transliterated into nonsense Devanagari).
    result = whisper_model.transcribe(tmp_path)
    transcript = result["text"].strip()

    chunks = retrieve(transcript)
    response = generate_response(transcript, chunks)

    audio_path = tmp_path.rsplit(".", 1)[0] + "_response.mp3"
    voice = pick_voice(response["answer"])
    tts = edge_tts.Communicate(text=response["answer"], voice=voice)
    await tts.save(audio_path)

    return {
        "transcript": transcript,
        "answer": response["answer"],
        "sources": response["sources"],
        "audio_url": f"/audio/{os.path.basename(audio_path)}"
    }


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = os.path.join(tempfile.gettempdir(), filename)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/debug")
def debug():
    all_docs = collection.get()
    return {
        "total_chunks": len(all_docs["ids"]),
        "ids": all_docs["ids"],
        "metadatas": all_docs["metadatas"]
    }