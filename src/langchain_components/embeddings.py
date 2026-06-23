import logging
import os
from typing import Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

_MEM_WARN_MB = 1500
_MEM_CRIT_MB = 512

_WSL_MEM_HELP = """\
  If using WSL2, configure memory in ~/.wslconfig:
    [wsl2]
    memory=8GB
    swap=8GB
  Then run: wsl --shutdown && wsl
  Alternatively, set embeddings_model_name=text-embedding-3-small in .env \
to use OpenAI's API (no local model load)."""


def _get_available_memory_mb() -> Optional[int]:
    """Check available memory on Linux via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (FileNotFoundError, OSError, IndexError, ValueError):
        pass
    return None


def _check_memory_before_load() -> None:
    """Warn or error before loading a local embedding model that can OOM."""
    avail_mb = _get_available_memory_mb()
    if avail_mb is None:
        return

    if avail_mb < _MEM_CRIT_MB:
        raise MemoryError(
            f"Only ~{avail_mb} MB of memory available. Loading a local embedding model "
            f"(sentence-transformers/all-MiniLM-L6-v2) typically requires more.\n"
            + _WSL_MEM_HELP
        )

    if avail_mb < _MEM_WARN_MB:
        logger.warning(
            "Only ~%d MB memory available. Loading sentence-transformers locally may "
            "cause an OOM crash. If the process dies:\n%s",
            avail_mb,
            _WSL_MEM_HELP,
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_embeddings_openai(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=model_name or settings.embeddings_model_name,
        api_key=settings.openai_api_key,
    )


def get_embeddings_huggingface(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except MemoryError:
        raise
    except Exception:
        logger.exception("Failed to load HuggingFace embeddings model '%s'", model_name)
        raise


def get_embeddings(model_name: Optional[str] = None) -> Any:
    settings = get_settings()
    if model_name is None:
        model_name = settings.embeddings_model_name
    if "text-embedding" in model_name:
        return get_embeddings_openai(model_name)
    _check_memory_before_load()
    return get_embeddings_huggingface(model_name)
