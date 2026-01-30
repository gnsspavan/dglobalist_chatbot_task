"""Load configuration from environment. Paths are relative to project root (parent of src)."""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

CSV_PATH = PROJECT_ROOT / "data" / "tracxn.csv"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "unicorn_startups"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BM25_INDEX_PATH = CHROMA_PERSIST_DIR / "bm25_index.pkl"

LOGGER_ENABLED = os.getenv("LOGGER_ENABLED", "false").strip().lower() in ("true", "1", "yes")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
if LANGFUSE_HOST and not os.environ.get("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = LANGFUSE_HOST

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
for name in ("transformers", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.ERROR)
