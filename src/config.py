"""
Central configuration. Keeping this as one small module (rather than scattering
env-var reads and magic numbers through the codebase) means every tunable in the
README's "design decisions" section maps to exactly one place in the code.
"""
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # picks up a local .env (see .env.example) before any os.environ.get() below

# --- Dataset -----------------------------------------------------------------
# Fixed "today" for the dataset, per DATA_README.md. All recency math (days since
# last open, etc.) is computed against this constant, NOT wall-clock time. This is
# what makes segment sizes reproducible in the eval harness regardless of when it runs.
DATASET_AS_OF_DATE = datetime.fromisoformat("2026-06-24T00:00:00")

# --- Paths ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_PATH = os.path.join(DATA_DIR, "data.sqlite")
GUIDELINES_DIR = os.path.join(BASE_DIR, "guidelines")
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, "guidelines_index")  # FAISS index on disk
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.yaml")  # system prompt + tool descriptions

# --- LLM -------------------------------------------------------------------------
# Single-key setup: Gemini covers both the planning LLM and embeddings (GOOGLE_API_KEY),
# so a reviewer only needs to set one env var to run the real (non-mock) path.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")  # "gemini" | "mock"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "gemini")  # "gemini" | "openai" | "local"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # no-API-key fallback

# --- Agent loop / cost & latency controls ----------------------------------------
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", 6))
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", 20))
TOOL_TIMEOUT_SECONDS = {
    "query_segment": 5,
    "search_guidelines": 3,
    "create_campaign": 5,
}
TOOL_DEFAULT_TIMEOUT_SECONDS = 5
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0

# --- RAG ---------------------------------------------------------------------------
RETRIEVAL_K = 4
COMPLIANCE_DOC_SLUG = "consent-compliance-and-opt-outs"
EXTERNAL_CHANNELS = {"push", "email"}  # channels that always pull the compliance chunk

# --- API -----------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))
