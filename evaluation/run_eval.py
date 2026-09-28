"""Lance le jeu d'évaluation contre les vrais modèles OpenRouter : `python -m evaluation.run_eval`.

Chaque cas tourne sur une copie jetable de data/neova_data.json (isolé, comme les tests) ; seuls
les modèles et l'index du corpus sont partagés entre les cas, pour limiter le coût."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from agent.config import load_settings
from agent.ingest import load_corpus
from agent.llm import UsageTracker, make_chat_model, make_embeddings
from agent.ocr import transcribe_image
from agent.retrieval import build_knowledge_base, make_search_tool
from evaluation.cases import CASES
from evaluation.harness import render_report, run_all

REPORT_PATH = Path(__file__).resolve().parent / "results.md"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()

    tracker = UsageTracker()
    chat_model = make_chat_model(settings, tracker=tracker)
    judge_model = make_chat_model(settings, model=settings.judge_model, tracker=tracker)
    chunks = load_corpus(transcribe=lambda path: transcribe_image(path, chat_model))
    kb = build_knowledge_base(chunks, make_embeddings(settings), embedding_model=settings.embedding_model)
    search_tool = make_search_tool(kb)

    results = run_all(CASES, chat_model, judge_model, search_tool)

    report = render_report(results, usage=tracker.summary())
    REPORT_PATH.write_text(report, encoding="utf-8")

    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} cas réussis. Rapport détaillé : {REPORT_PATH}")
    print(f"Consommation LLM : {tracker.summary()}")


if __name__ == "__main__":
    main()
