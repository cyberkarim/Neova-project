"""Discussion en ligne de commande avec l'agent : `python -m agent.cli` (l'API doit tourner)."""
from __future__ import annotations

import logging
import sys

import httpx

from .app import build_agent_graph
from .config import load_settings
from .llm import UsageTracker
from .session import Conversation


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()

    try:
        httpx.get(f"{settings.api_url}/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        raise SystemExit(f"L'API Néova ne répond pas sur {settings.api_url}. Lance : python -m uvicorn api.main:app")

    tracker = UsageTracker()
    graph, http = build_agent_graph(settings, tracker)
    conversation = Conversation(graph)
    print("Conseiller virtuel Néova Télécom (Ctrl+C pour quitter)")
    try:
        while True:
            text = input("\nVous > ").strip()
            if not text:
                continue
            result = conversation.send(text)
            print(f"\nConseiller > {result.reply}")
            if result.tools_called:
                print(f"[outils appelés : {', '.join(result.tools_called)}]")
            if result.escalated:
                print("[dossier transféré à un conseiller humain]")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        http.close()
        print(f"\nConsommation LLM : {tracker.summary()}")


if __name__ == "__main__":
    main()
