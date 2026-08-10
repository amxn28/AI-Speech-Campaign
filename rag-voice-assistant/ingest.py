import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
from config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

# Init ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="campaign_docs", embedding_function=ef)

def parse_file(file_path: str) -> str:
    """Extract text from PDF, DOCX, TXT, or MD."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def ingest_document(file_path: str, metadata: dict) -> int:
    """Parse, chunk, embed, and store a document."""
    text = parse_file(file_path)
    chunks = chunk_text(text)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [metadata for _ in chunks]
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)

    