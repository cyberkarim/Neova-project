"""Transcription des documents image du corpus par un modèle vision, avec cache disque."""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

TRANSCRIPTION_PROMPT = (
    "Transcris fidèlement ce document en Markdown. Utilise `#` pour le titre du document et `##` pour "
    "chaque intertitre. Reproduis les tableaux en tableaux Markdown, ainsi que l'en-tête de référence "
    "et le pied de page. N'ajoute rien, ne corrige rien, ne résume pas. Réponds uniquement avec le Markdown."
)


_OUTER_CODE_FENCE = re.compile(r"\A\s*```[a-zA-Z]*\n(.*?)\n```\s*\Z", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Les modèles enveloppent souvent leur Markdown dans ```markdown ... ``` ; le splitter par titres
    ignore alors tous les titres, qu'il prend pour du code."""
    match = _OUTER_CODE_FENCE.match(text)
    return match.group(1) if match else text


def transcribe_image(path: Path, chat_model: BaseChatModel, cache_dir: Path = CACHE_DIR) -> str:
    image_bytes = path.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    cache_path = cache_dir / "ocr" / f"{path.stem}-{digest}.md"
    if cache_path.exists():
        return strip_code_fence(cache_path.read_text(encoding="utf-8"))

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
    )
    text = chat_model.invoke([message]).content
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Transcription vide pour {path.name}")

    text = strip_code_fence(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text
