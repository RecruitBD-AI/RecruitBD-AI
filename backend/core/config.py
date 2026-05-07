from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

# SentenceTransformer model name
SBERT_MODEL = "all-MiniLM-L6-v2"
