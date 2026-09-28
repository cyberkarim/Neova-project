"""Fabriques de modèles OpenRouter (API compatible OpenAI) et suivi de la consommation de tokens."""
from __future__ import annotations

import logging
import threading

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings

logger = logging.getLogger(__name__)

# Le client OpenAI retente lui-même les 429/5xx avec un backoff exponentiel.
MAX_RETRIES = 6
REQUEST_TIMEOUT_SECONDS = 60


class UsageTracker(BaseCallbackHandler):
    """Cumule les tokens des appels chat pour rapporter la dépense dans le README."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                with self._lock:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    self.calls += 1

    def summary(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def make_chat_model(settings: Settings, *, model: str | None = None, tracker: UsageTracker | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
        callbacks=[tracker] if tracker else None,
    )


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False : sans cela, LangChain tokenise avec tiktoken et envoie des
    # identifiants de tokens OpenAI, que les modèles non-OpenAI (bge-m3, ...) ne comprennent pas.
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        check_embedding_ctx_length=False,
        max_retries=MAX_RETRIES,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
    )
