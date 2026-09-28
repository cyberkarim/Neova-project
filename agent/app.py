"""Assemblage de l'agent de production : vrais modèles OpenRouter, corpus indexé, API FastAPI."""
from __future__ import annotations

import httpx

from .config import Settings
from .graph import build_graph
from .ingest import load_corpus
from .llm import UsageTracker, make_chat_model, make_embeddings
from .ocr import transcribe_image
from .retrieval import build_knowledge_base, make_search_tool
from .tools import build_api_tools

API_TIMEOUT_SECONDS = 10


def build_agent_graph(settings: Settings, tracker: UsageTracker | None = None):
    chat_model = make_chat_model(settings, tracker=tracker)
    judge_model = make_chat_model(settings, model=settings.judge_model, tracker=tracker)
    chunks = load_corpus(transcribe=lambda path: transcribe_image(path, chat_model))
    kb = build_knowledge_base(chunks, make_embeddings(settings), embedding_model=settings.embedding_model)
    http = httpx.Client(base_url=settings.api_url, timeout=API_TIMEOUT_SECONDS)
    return build_graph(chat_model, judge_model, build_api_tools(http), make_search_tool(kb)), http
