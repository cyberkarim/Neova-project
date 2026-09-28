"""Une conversation avec l'agent : gère les tours de parole, la reprise après confirmation et les pannes du LLM."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .graph import _text

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 40
TECHNICAL_ERROR_MESSAGE = (
    "Je rencontre un problème technique et ne peux pas traiter votre demande pour le moment. "
    "Merci de réessayer dans quelques instants."
)


@dataclass
class TurnResult:
    reply: str
    escalated: bool = False
    awaiting_confirmation: bool = False
    error: bool = False
    tools_called: list[str] = field(default_factory=list)
    ticket: dict | None = None


class Conversation:
    def __init__(self, graph, thread_id: str | None = None):
        self._graph = graph
        self._config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": RECURSION_LIMIT}
        self._awaiting_confirmation = False

    def state(self) -> dict:
        """État courant du graphe (utilisé par l'évaluation pour vérifier `locked`, etc.)."""
        return self._graph.get_state(self._config).values

    def send(self, text: str) -> TurnResult:
        payload = Command(resume=text) if self._awaiting_confirmation else {"messages": [HumanMessage(content=text)]}
        try:
            result = self._graph.invoke(payload, self._config)
        except Exception:
            logger.exception("Échec du graphe")
            return TurnResult(reply=TECHNICAL_ERROR_MESSAGE, error=True)

        interrupts = result.get("__interrupt__")
        self._awaiting_confirmation = bool(interrupts)
        if interrupts:
            return TurnResult(reply=interrupts[0].value["recap"], awaiting_confirmation=True)

        messages = result["messages"]
        reply = next((_text(m) for m in reversed(messages) if isinstance(m, AIMessage)), "")
        turn = _current_turn(messages)
        escalation = result.get("escalation") or {}
        return TurnResult(
            reply=reply,
            escalated=bool(result.get("escalated")),
            tools_called=[c["name"] for m in turn if isinstance(m, AIMessage) for c in m.tool_calls],
            ticket=escalation.get("ticket"),
        )


def _current_turn(messages) -> list:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return list(messages)
