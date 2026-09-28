import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from agent.ingest import CORPUS_DIR, load_image
from agent.ocr import transcribe_image
from agent.retrieval import build_knowledge_base, make_search_tool

DIMENSIONS = 512


class HashingEmbeddings(Embeddings):
    """Embeddings déterministes hors ligne : sac de mots haché, normalisé (cosinus = recouvrement lexical)."""

    def __init__(self) -> None:
        self.documents_embedded = 0

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


ROAMING_MARKDOWN = """# Utilisation à l'étranger — offres mobiles

Fiche d'information client — Réf. FIC-ROAM-2026-02. Remplace et annule la fiche FIC-ROAM-2025-04.

## Union européenne, DOM inclus
Les appels, SMS et données sont utilisables depuis l'Union européenne sans surcoût.

| Offre | Données utilisables depuis l'UE |
|---|---|
| Mobile Néova 80 Go | 25 Go |

## Hors Union européenne
Aucune consommation n'est incluse. Appel émis : 0,50 € / minute.
"""


class StubVisionModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=ROAMING_MARKDOWN)


@pytest.fixture()
def kb(chunks, tmp_path):
    return build_knowledge_base(chunks, HashingEmbeddings(), embedding_model="hashing", cache_dir=tmp_path)


def test_search_finds_the_relevant_document(kb):
    hits = kb.search("voyant rouge clignotant incident réseau box")
    assert hits[0].document.metadata["doc_id"] == "faq-box-internet"


def test_deprecated_documents_are_excluded_by_default(kb):
    query = "promo rentrée 2024 fibre 19,99 première année"
    default_docs = {h.document.metadata["doc_id"] for h in kb.search(query, k=10)}
    assert "promo-rentree-2024" not in default_docs

    with_deprecated = kb.search(query, k=10, include_deprecated=True)
    assert with_deprecated[0].document.metadata["doc_id"] == "promo-rentree-2024"


def test_min_score_drops_weak_matches(kb):
    assert kb.search("voyant rouge box", min_score=0.99) == []


def test_index_is_cached_and_invalidated_by_model_change(chunks, tmp_path):
    first = HashingEmbeddings()
    build_knowledge_base(chunks, first, embedding_model="model-a", cache_dir=tmp_path)
    assert first.documents_embedded == len(chunks)

    second = HashingEmbeddings()
    cached = build_knowledge_base(chunks, second, embedding_model="model-a", cache_dir=tmp_path)
    assert second.documents_embedded == 0
    assert cached.search("voyant rouge clignotant")[0].document.metadata["doc_id"] == "faq-box-internet"

    third = HashingEmbeddings()
    build_knowledge_base(chunks, third, embedding_model="model-b", cache_dir=tmp_path)
    assert third.documents_embedded == len(chunks)


def test_search_tool_flags_internal_documents(kb):
    tool = make_search_tool(kb)
    results = tool.invoke({"query": "geste commercial conditions plafond mensualité impayé ancienneté"})["results"]
    internal = [r for r in results if r["source"] == "politique-geste-commercial"]
    assert internal and all("INTERNE" in r["restriction"] for r in internal)

    public = tool.invoke({"query": "quel est le prix de la Fibre Néova 500 Mb/s"})["results"]
    assert all("restriction" not in r for r in public if r["source"] == "grille-tarifaire-2026")


def test_search_tool_returns_empty_results_below_threshold(kb):
    tool = make_search_tool(kb, min_score=0.99)
    assert tool.invoke({"query": "voyant rouge box"}) == {"ok": True, "results": []}


def test_image_transcription_is_cached(tmp_path):
    image = tmp_path / "fiche.png"
    image.write_bytes(b"\x89PNG fake image bytes")
    model = StubVisionModel()

    first = transcribe_image(image, model, cache_dir=tmp_path / "cache")
    second = transcribe_image(image, model, cache_dir=tmp_path / "cache")

    assert first == second == ROAMING_MARKDOWN
    assert model.calls == 1


def test_code_fence_around_transcription_is_removed_from_output_and_cache(tmp_path):
    image = tmp_path / "fiche.png"
    image.write_bytes(b"\x89PNG fake image bytes")
    fenced = StubVisionModel()
    fenced.invoke = lambda messages: AIMessage(content=f"```markdown\n{ROAMING_MARKDOWN}\n```")

    result = transcribe_image(image, fenced, cache_dir=tmp_path / "cache")
    assert result == ROAMING_MARKDOWN
    assert transcribe_image(image, fenced, cache_dir=tmp_path / "cache") == ROAMING_MARKDOWN

    stale = tmp_path / "cache" / "ocr"
    (next(stale.glob("*.md"))).write_text(f"```markdown\n{ROAMING_MARKDOWN}\n```", encoding="utf-8")
    assert transcribe_image(image, fenced, cache_dir=tmp_path / "cache") == ROAMING_MARKDOWN


def test_image_documents_are_chunked_with_their_metadata():
    model = StubVisionModel()
    roaming = load_image(CORPUS_DIR / "fiche-roaming-international-scan.png", lambda path: model.invoke([path]).content)

    assert {c.metadata["doc_id"] for c in roaming} == {"fiche-roaming-international"}
    assert {"Union européenne, DOM inclus", "Hors Union européenne"} <= {c.metadata["section"] for c in roaming}
    assert {c.metadata["updated"] for c in roaming} == {"2026-02-27"}
    assert {c.metadata["status"] for c in roaming} == {"current"}
    assert any("25 Go" in c.page_content and "|---|" in c.page_content for c in roaming)
    assert any("annule la fiche FIC-ROAM-2025-04" in c.page_content for c in roaming)
