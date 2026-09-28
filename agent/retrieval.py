"""Base de connaissances : index vectoriel du corpus, recherche filtrée, tool pour l'agent."""
from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore

from .ocr import CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4
INTERNAL_NOTICE = (
    "DOCUMENT INTERNE : sert à décider quoi faire, mais ne jamais citer au client ses seuils, "
    "plafonds, critères ni procédures. Lui communiquer uniquement la décision et sa raison générale."
)


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


class KnowledgeBase:
    def __init__(self, store: InMemoryVectorStore):
        self._store = store

    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_TOP_K,
        include_deprecated: bool = False,
        min_score: float | None = None,
    ) -> list[Hit]:
        """Cherche les k passages les plus proches. Les documents `deprecated` sont écartés par défaut :
        ils contredisent les documents en vigueur (ex. promo 2024 vs grille 2026)."""

        def keep(doc: Document) -> bool:
            return include_deprecated or doc.metadata.get("status") != "deprecated"

        query_vector = self._store.embedding.embed_query(query)
        scored = self._store.similarity_search_with_score_by_vector(query_vector, k=k, filter=keep)
        return [Hit(doc, score) for doc, score in scored if min_score is None or score >= min_score]


def _index_path(chunks: list[Document], embedding_model: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(embedding_model.encode())
    for chunk in chunks:
        digest.update(chunk.metadata["chunk_id"].encode())
        digest.update(chunk.page_content.encode())
    return cache_dir / "index" / f"{digest.hexdigest()[:16]}.json"


def build_knowledge_base(
    chunks: list[Document],
    embeddings: Embeddings,
    *,
    embedding_model: str,
    cache_dir: Path = CACHE_DIR,
) -> KnowledgeBase:
    """Indexe les chunks. L'index est mis en cache : modèle d'embedding ou corpus changé => réindexation."""
    path = _index_path(chunks, embedding_model, cache_dir)
    if path.exists():
        logger.info("Index chargé depuis le cache: %s", path.name)
        return KnowledgeBase(InMemoryVectorStore.load(str(path), embeddings))

    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(path))
    logger.info("Index construit (%d chunks): %s", len(chunks), path.name)
    return KnowledgeBase(store)


def make_search_tool(kb: KnowledgeBase, *, min_score: float | None = None) -> BaseTool:
    @tool
    def search_knowledge_base(query: str) -> dict:
        """Recherche dans la documentation Néova (FAQ, CGV, grille tarifaire, procédures).
        À utiliser pour toute question sur les tarifs, délais, frais, règles et procédures.
        Si aucun passage pertinent n'est renvoyé, la réponse n'est pas documentée : ne pas inventer."""
        hits = kb.search(query, min_score=min_score)
        results = []
        for hit in hits:
            meta = hit.document.metadata
            entry = {
                "source": meta["doc_id"],
                "section": meta.get("section", ""),
                "updated": meta.get("updated", ""),
                "score": round(hit.score, 3),
                "text": hit.document.page_content,
            }
            if meta.get("audience") == "internal":
                entry["restriction"] = INTERNAL_NOTICE
            results.append(entry)
        return {"ok": True, "results": results}

    return search_knowledge_base


def main(argv: list[str]) -> None:
    """Interroge la base à la main : `python -m agent.retrieval "<question>"` (nécessite le .env)."""
    from .config import load_settings
    from .ingest import load_corpus
    from .llm import make_chat_model, make_embeddings
    from .ocr import transcribe_image

    if not argv:
        raise SystemExit('usage: python -m agent.retrieval "<question>"')
    settings = load_settings()
    chat_model = make_chat_model(settings)
    chunks = load_corpus(transcribe=lambda path: transcribe_image(path, chat_model))
    kb = build_knowledge_base(chunks, make_embeddings(settings), embedding_model=settings.embedding_model)
    sys.stdout.reconfigure(encoding="utf-8")
    for hit in kb.search(" ".join(argv)):
        meta = hit.document.metadata
        print(f"\n--- {hit.score:.3f} | {meta['chunk_id']} | {meta['status']}/{meta['audience']} | {meta['section']}")
        print(hit.document.page_content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
