from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Motif de transfert en attente d'exécution par le nœud ESCALATE (posé par triage, agent ou verify).
    # Réinitialisé à None par TRIAGE au début de chaque tour non verrouillé.
    escalation: dict | None
    # True si ESCALATE a créé un ticket pendant CE tour. Réinitialisé à False par TRIAGE à chaque tour :
    # ne reflète jamais un transfert passé, seulement celui du tour en cours (utile pour la CLI/l'API).
    escalated: bool
    # True une fois qu'un motif de transfert immédiat et sensible a été détecté (RGPD, procédure
    # judiciaire, décès, fraude, mineur/protégé, détresse). Jamais remis à False : bloque tout
    # traitement automatique pour le reste du fil. Les autres motifs d'escalade (hors périmètre,
    # échec technique) ne le posent pas : la conversation continue normalement ensuite.
    locked: bool
    # Retour du nœud VERIFY à l'agent quand sa réponse a été refusée.
    critique: str | None
    verify_attempts: int
    tool_rounds: int
    confirmed: bool
