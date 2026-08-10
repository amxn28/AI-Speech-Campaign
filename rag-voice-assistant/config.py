import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
SIMILARITY_THRESHOLD = 0.1

# LLM
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024