"""Ingestion du corpus : PDF -> Markdown nettoyé -> chunks LangChain `Document`.

Chaque chunk porte les métadonnées de son document (statut, public, date de mise à
jour) pour que le retrieval puisse écarter les documents `deprecated` et que
l'agent sache quels contenus sont internes (jamais à citer tels quels au client).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP_CHARS = 150
MIN_BODY_CHARS = 40

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# Les images n'ont pas d'en-tête structuré : métadonnées relevées à la main sur la fiche
# (Réf. FIC-ROAM-2026-02, mise à jour le 27 février 2026, "version en vigueur", diffusion clients et conseillers).
IMAGE_METADATA = {
    "fiche-roaming-international-scan.png": {
        "doc_id": "fiche-roaming-international",
        "title": "Utilisation à l'étranger — offres mobiles",
        "status": "current",
        "audience": "public",
        "updated": "2026-02-27",
        "version": "FIC-ROAM-2026-02",
    }
}

HEADERS_TO_SPLIT_ON =[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
FRONT_MATTER_KEYS = ("id", "version", "statut", "public", "maj", "source")

_FRONT_MATTER_LINE = re.compile(r"^\*\*(" + "|".join(FRONT_MATTER_KEYS) + r")\*\*\s")
_FRONT_MATTER_FIELD = re.compile(r"\*\*(\w+)\*\*\s+(.*?)(?=\s+\*\*\w+\*\*|\s*$)")
_NOISE_PATTERNS = [
    re.compile(r"```\s*corpus/[^\n]*\n\s*```\s*"),  # bloc "corpus/x.md · N mots" et pied de page
    re.compile(r"^\*\*CO R P U S\*\*\s*$", re.MULTILINE),
    re.compile(r"^\d+ / \d+\s*$", re.MULTILINE),  # marqueurs de pagination
    re.compile(r"^- *$", re.MULTILINE),  # puces vides produites par l'extraction
    re.compile(r"^Nous utilisons des cookies.*$", re.MULTILINE),  # export du site d'aide
    re.compile(r"^Accueil > .*$", re.MULTILINE),
    re.compile(r"^- (Accueil|Aide|Espace client) *$", re.MULTILINE),
    re.compile(r"^©.*$", re.MULTILINE),
]


def pdf_to_markdown(path: Path) -> str:
    return pymupdf4llm.to_markdown(str(path))


def _strip_bold(text: str) -> str:
    return text.replace("**", "").strip()


def parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    fields: dict[str, str] = {}
    kept: list[str] = []
    for line in markdown.splitlines():
        if _FRONT_MATTER_LINE.match(line):
            for key, value in _FRONT_MATTER_FIELD.findall(line):
                fields[key] = value.strip()
        else:
            kept.append(line)
    return fields, "\n".join(kept)


def extract_title(markdown: str) -> str:
    match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
    return _strip_bold(match.group(1)) if match else ""


def clean_markdown(markdown: str, title: str) -> str:
    for pattern in _NOISE_PATTERNS:
        markdown = pattern.sub("", markdown)
    if title:
        # Le titre courant est répété en pied de page, en texte simple (pas en titre Markdown).
        markdown = re.sub(rf"^{re.escape(title)}\s*$", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"^(#{1,6}) \*\*(.+?)\*\*\s*$", r"\1 \2", markdown, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def chunk_document(markdown: str, metadata: dict) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS)

    sections = header_splitter.split_text(markdown)
    chunks: list[Document] = []
    for section in sections:
        section_path = " > ".join(section.metadata[k] for k in ("h2", "h3", "h4") if k in section.metadata)
        body_without_headings = re.sub(r"^#{1,6} .*$", "", section.page_content, flags=re.MULTILINE).strip()
        if len(body_without_headings) < MIN_BODY_CHARS:
            continue
        pieces = (
            size_splitter.split_text(section.page_content)
            if len(section.page_content) > MAX_CHUNK_CHARS
            else [section.page_content]
        )
        for piece in pieces:
            piece = re.sub(r"^# .*\n?", "", piece, flags=re.MULTILINE).strip()
            chunk_id = f"{metadata['doc_id']}#{len(chunks)}"
            header = f"[{metadata['title']}]" + (f" {section_path}" if section_path else "")
            chunks.append(
                Document(
                    page_content=f"{header}\n\n{piece}",
                    metadata={**metadata, "section": section_path, "chunk_id": chunk_id},
                )
            )
    return chunks


def load_pdf(path: Path) -> list[Document]:
    raw = pdf_to_markdown(path)
    front_matter, body = parse_front_matter(raw)
    title = extract_title(body)
    cleaned = clean_markdown(body, title)
    metadata = {
        "doc_id": front_matter.get("id", path.stem),
        "title": title,
        "status": front_matter.get("statut", "current"),
        "audience": front_matter.get("public", "public"),
        "updated": front_matter.get("maj", ""),
        "version": front_matter.get("version", ""),
        "source_file": path.name,
    }
    return chunk_document(cleaned, metadata)


def load_image(path: Path, transcribe: Callable[[Path], str]) -> list[Document]:
    markdown = transcribe(path).strip()
    known = IMAGE_METADATA.get(path.name, {})
    title = extract_title(markdown) or known.get("title", path.stem)
    metadata = {
        "doc_id": known.get("doc_id", path.stem),
        "title": title,
        "status": known.get("status", "current"),
        "audience": known.get("audience", "public"),
        "updated": known.get("updated", ""),
        "version": known.get("version", ""),
        "source_file": path.name,
    }
    return chunk_document(clean_markdown(markdown, title), metadata)


def load_corpus(corpus_dir: Path = CORPUS_DIR, transcribe: Callable[[Path], str] | None = None) -> list[Document]:
    chunks: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks.extend(load_pdf(path))
        elif suffix in IMAGE_SUFFIXES and transcribe is not None:
            chunks.extend(load_image(path, transcribe))
        else:
            logger.warning("Fichier ignoré par l'ingestion: %s", path.name)
    return chunks
