"""
config.py — Centralized configuration for the Urban Greening Targets Pipeline
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent
BASE_DIR = MODULE_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "pipeline.db"
PROMPTS_DIR = MODULE_DIR / "prompts"

# Ensure dirs exist
for d in [CACHE_DIR, OUTPUT_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load .env ─────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

# ── API Keys ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")

# Load all SERP_API_KEY* entries (SERP_API_KEY, SERP_API_KEY1, SERP_API_KEY2, ...)
def _load_serp_keys() -> list[str]:
    keys: list[str] = []
    # Check plain SERP_API_KEY first
    base = os.getenv("SERP_API_KEY", "")
    if base:
        keys.append(base)
    # Then numbered keys: SERP_API_KEY1, SERP_API_KEY2, ...
    for i in range(1, 100):
        k = os.getenv(f"SERP_API_KEY{i}", "")
        if k:
            keys.append(k)
    return keys

SERP_API_KEYS: list[str] = _load_serp_keys()
# Keep backward-compat single key (first available)
SERP_API_KEY = SERP_API_KEYS[0] if SERP_API_KEYS else ""


class SerpKeyManager:
    """Rotates through SerpAPI keys when one is exhausted (HTTP 429 or quota error)."""

    def __init__(self, keys: list[str]):
        self._keys = list(keys)
        self._idx = 0

    @property
    def current_key(self) -> str:
        if not self._keys:
            return ""
        return self._keys[self._idx % len(self._keys)]

    def rotate(self) -> str | None:
        """Move to next key. Returns new key or None if all exhausted."""
        if not self._keys:
            return None
        self._idx += 1
        if self._idx >= len(self._keys):
            return None  # all keys tried
        return self._keys[self._idx]

    def reset(self):
        """Reset back to the first key (for a new run)."""
        self._idx = 0

    @property
    def remaining(self) -> int:
        if not self._keys:
            return 0
        return len(self._keys) - self._idx

    def __bool__(self):
        return bool(self._keys)


serp_key_mgr = SerpKeyManager(SERP_API_KEYS)

# ── Model Settings ────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── Rate Limits ───────────────────────────────────────────
SEARCH_RATE_LIMIT = int(os.getenv("SEARCH_RATE_LIMIT", "10"))
FETCH_RATE_LIMIT = int(os.getenv("FETCH_RATE_LIMIT", "2"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))

# ── Pipeline Settings ────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
PAGE_CHAR_LIMIT = int(os.getenv("PAGE_CHAR_LIMIT", "30000"))
PDF_CHAR_LIMIT = int(os.getenv("PDF_CHAR_LIMIT", "30000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

# ── HTTP ──────────────────────────────────────────────────
USER_AGENT = "UrbanGreeningBot/1.0 (+https://natcap.stanford.edu)"
REQUEST_TIMEOUT = 30

# ── Logging ───────────────────────────────────────────────
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("greening")
