"""Graphe LangGraph de l'agent : triage -> agent (outils) -> vérification, avec confirmation
avant toute action qui modifie l'état et escalade vers un conseiller humain."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .prompts import AGENT_PROMPT, CONFIRMATION_PROMPT, CRITIQUE_PROMPT, TRIAGE_PROMPT, VERIFY_PROMPT
from .state import AgentState
from .tools import STATE_CHANGING_TOOLS, EscalationCategory, Urgency

logger = logging.getLogger(__name__)

HANDOVER_TOOL = "request_human_handover"
CREATE_TICKET_TOOL = "create_escalation_ticket"

MAX_TOOL_ROUNDS = 8
MAX_VERIFY_RETRIES = 1
EVIDENCE_BUDGET_CHARS = 12000
EVIDENCE_ITEM_CHARS = 3500

HANDOVER_MESSAGE = (
    "Je transmets votre dossier à un conseiller (référence {ticket_id}). Il vous rappellera sous 45 minutes "
    "en heures ouvrées, ou demain matin si votre demande est faite en dehors de ces horaires."
)
HANDOVER_FAILED_MESSAGE = (
    "Je n'ai pas réussi à enregistrer votre demande de transfert pour le moment. Merci de réessayer dans "
    "quelques instants ou de contacter directement le service client."
)
ALREADY_HANDED_OVER_MESSAGE = (
    "Votre demande a déjà été transmise à un conseiller, qui vous rappellera. "
    "Je ne peux pas traiter d'autre demande dans cette conversation."
)
APPOINTMENT_FEE_REMINDER = (
    "Rappel : l'intervention est gratuite, sauf si le technicien constate une dégradation imputable au client "
    "(câble sectionné, prise arrachée, matériel immergé) ; elle est alors facturée 69 €."
)
REASON_LABELS = {
    "no_internet": "panne internet",
    "slow_internet": "débit lent",
    "installation": "installation",
    "equipment_swap": "échange d'équipement",
}


class Triage(BaseModel):
    mandatory_escalation: bool = Field(description="True si l'une des situations à transfert immédiat est clairement présente")
    situation: (
        Literal["rgpd", "procedure_juridique", "deces", "fraude", "contrat_pro", "mineur_ou_protege", "detresse"] | None
    ) = None
    category: EscalationCategory = "other"
    urgency: Urgency = "normal"
    summary: str = ""


class Verdict(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)


class ConfirmationDecision(BaseModel):
    decision: Literal["confirm", "decline", "other"]


@tool
def request_human_handover(
    category: EscalationCategory, summary: str, urgency: Urgency = "normal", actions_taken: str | None = None
) -> str:
    """Transfère le dossier à un conseiller humain. À utiliser sur demande du client, quand la documentation ne
    permet pas de répondre, ou quand la situation dépasse ton cadre. Le résumé est factuel et permet au conseiller
    de reprendre sans reposer de questions."""
    return "Transfert demandé."


def _text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _tool_result(message: ToolMessage) -> dict | None:
    try:
        parsed = json.loads(_text(message))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text(m)
    return ""


def _since_last_human(messages: list[AnyMessage]) -> list[AnyMessage]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return list(messages)


def _unreported_state_changes(messages: list[AnyMessage]) -> list[str]:
    """Réservations/annulations déjà exécutées dans le tour en cours : si VERIFY rejette la réponse
    qui les annonçait, le client et le conseiller doivent savoir qu'elles ont bien eu lieu, à partir
    du résultat réel de l'outil, jamais du texte rejeté du LLM."""
    notes = []
    for m in _since_last_human(messages):
        if not (isinstance(m, ToolMessage) and m.name in STATE_CHANGING_TOOLS):
            continue
        result = _tool_result(m)
        if not (result and result.get("ok")):
            continue
        data = result["data"]
        if m.name == "book_appointment":
            slot = _describe_slot(_find_slot(messages, data["slot_id"]), data["slot_id"])
            notes.append(f"Rendez-vous déjà réservé : {slot}, référence {data['appointment_id']}.")
        elif m.name == "cancel_appointment":
            notes.append(f"Rendez-vous {data['appointment_id']} déjà annulé, créneau libéré.")
    return notes


def _dialogue_excerpt(messages: list[AnyMessage], limit: int = 6) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Client : {_text(m)}")
        elif isinstance(m, AIMessage) and _text(m).strip() and not m.tool_calls:
            lines.append(f"Conseiller : {_text(m)}")
    return "\n".join(lines[-limit:])


def verified_customer_ids(messages: list[AnyMessage]) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "get_customer":
            result = _tool_result(m)
            if result and result.get("ok"):
                ids.add(result["data"]["customer_id"])
    return ids


def _find_slot(messages: list[AnyMessage], slot_id: str) -> dict | None:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "list_technician_slots":
            result = _tool_result(m)
            for slot in (result or {}).get("data", []):
                if slot.get("slot_id") == slot_id:
                    return slot
    return None


def _describe_slot(slot: dict | None, slot_id: str) -> str:
    if not slot:
        return slot_id
    try:
        start, end = datetime.fromisoformat(slot["start"]), datetime.fromisoformat(slot["end"])
    except (KeyError, ValueError):
        return slot_id
    return f"{start:%d/%m/%Y} de {start:%Hh%M} à {end:%Hh%M} ({slot_id})"


def _evidence(messages: list[AnyMessage]) -> str:
    items: list[str] = []
    budget = EVIDENCE_BUDGET_CHARS
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            item = f"[{m.name}] {_text(m)[:EVIDENCE_ITEM_CHARS]}"
            if len(item) > budget:
                break
            budget -= len(item)
            items.append(item)
    return "\n\n".join(reversed(items)) or "(aucune)"


def format_confirmation(calls: list[dict], messages: list[AnyMessage]) -> str:
    lines = ["Je vais effectuer l'action suivante :"]
    books = False
    for call in calls:
        args = call["args"]
        if call["name"] == "book_appointment":
            books = True
            slot = _describe_slot(_find_slot(messages, args.get("slot_id", "")), args.get("slot_id", "?"))
            reason = REASON_LABELS.get(args.get("reason", ""), args.get("reason", ""))
            lines.append(f"- Réserver le créneau {slot} pour le client {args.get('customer_id')} (motif : {reason}).")
        elif call["name"] == "cancel_appointment":
            lines.append(f"- Annuler le rendez-vous {args.get('appointment_id')}.")
    if books:
        lines.append(APPOINTMENT_FEE_REMINDER)
    lines.append("Confirmez-vous ? (oui / non)")
    return "\n".join(lines)


def _not_executed(calls: list[dict], reason: str, *, only: dict[str, str] | None = None) -> list[ToolMessage]:
    """Une réponse ToolMessage par appel d'outil, sans quoi l'historique devient invalide pour le LLM."""
    only = only or {}
    return [
        ToolMessage(
            content=json.dumps({"ok": False, "error": only.get(c["id"], reason)}, ensure_ascii=False),
            tool_call_id=c["id"],
            name=c["name"],
        )
        for c in calls
    ]


def build_graph(
    chat_model: Any,
    judge_model: Any,
    tools: list[BaseTool],
    search_tool: BaseTool,
    *,
    checkpointer: Any = None,
    today: Callable[[], date] = date.today,
):
    by_name = {t.name: t for t in tools}
    ticket_tool = by_name[CREATE_TICKET_TOOL]
    executable = [t for t in tools if t.name != CREATE_TICKET_TOOL] + [search_tool]

    agent_llm = chat_model.bind_tools([*executable, request_human_handover])
    triage_llm = judge_model.with_structured_output(Triage, method="function_calling")
    verify_llm = judge_model.with_structured_output(Verdict, method="function_calling")
    decision_llm = judge_model.with_structured_output(ConfirmationDecision, method="function_calling")

    def route_start(state: AgentState) -> str:
        return "handed_over" if state.get("locked") else "triage"

    def handed_over_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=ALREADY_HANDED_OVER_MESSAGE)]}

    def triage_node(state: AgentState) -> dict:
        reset = {"critique": None, "verify_attempts": 0, "tool_rounds": 0, "escalation": None, "escalated": False}
        result = triage_llm.invoke(
            [SystemMessage(TRIAGE_PROMPT), HumanMessage(_dialogue_excerpt(state["messages"]))]
        )
        if not result.mandatory_escalation:
            return reset
        summary = result.summary or f"Situation à transfert immédiat : {result.situation}."
        update = {
            **reset,
            "escalation": {
                "category": result.category,
                "urgency": result.urgency,
                "summary": f"{summary} (motif de transfert immédiat : {result.situation})",
            },
        }
        if result.situation != "contrat_pro":
            update["locked"] = True  # situation sensible : verrouille le fil, contrairement à contrat_pro
        return update

    def route_after_triage(state: AgentState) -> str:
        return "escalate" if state.get("escalation") else "agent"

    def agent_node(state: AgentState) -> dict:
        prompt: list[AnyMessage] = [
            SystemMessage(AGENT_PROMPT.format(today=today().strftime("%d/%m/%Y"))),
            *state["messages"],
        ]
        if state.get("critique"):
            prompt.append(SystemMessage(state["critique"]))
        response = agent_llm.invoke(prompt)

        update: dict = {"messages": [response]}
        calls = response.tool_calls or []
        handover = next((c for c in calls if c["name"] == HANDOVER_TOOL), None)
        if handover:
            update["escalation"] = dict(handover["args"])
        elif calls:
            rounds = state.get("tool_rounds", 0) + 1
            update["tool_rounds"] = rounds
            if rounds > MAX_TOOL_ROUNDS:
                update["escalation"] = {
                    "category": "other",
                    "urgency": "normal",
                    "summary": f"Traitement automatique interrompu (trop d'appels d'outils). "
                    f"Demande du client : {_last_user_text(state['messages'])}",
                }
        return update

    def route_after_agent(state: AgentState) -> str:
        if state.get("escalation"):
            return "escalate"
        calls = state["messages"][-1].tool_calls
        if not calls:
            return "verify"
        if any(c["name"] in STATE_CHANGING_TOOLS for c in calls):
            return "confirm"
        return "tools"

    def _identity_errors(calls: list[dict], messages: list[AnyMessage]) -> dict[str, str]:
        verified = verified_customer_ids(messages)
        errors: dict[str, str] = {}
        for call in calls:
            if call["name"] == "book_appointment" and call["args"].get("customer_id") not in verified:
                errors[call["id"]] = (
                    "Identité non vérifiée : demande l'identifiant client et le numéro de téléphone, "
                    "puis appelle get_customer avant de réserver."
                )
            elif call["name"] == "cancel_appointment":
                if not verified:
                    errors[call["id"]] = (
                        "Identité non vérifiée : demande l'identifiant client et le numéro de téléphone, "
                        "puis appelle get_customer avant d'annuler."
                    )
                    continue
                appointment = by_name["get_appointment"].invoke({"appointment_id": call["args"].get("appointment_id")})
                if not appointment.get("ok"):
                    errors[call["id"]] = appointment["error"]
                elif appointment["data"]["customer_id"] not in verified:
                    errors[call["id"]] = "Ce rendez-vous n'appartient pas au client dont l'identité a été vérifiée."
        return errors

    def confirm_node(state: AgentState) -> dict:
        messages = state["messages"]
        calls = messages[-1].tool_calls
        errors = _identity_errors(calls, messages)
        if errors:
            return {
                "messages": _not_executed(calls, "Non exécuté : une autre action du même lot a été refusée.", only=errors),
                "confirmed": False,
            }

        recap = format_confirmation([c for c in calls if c["name"] in STATE_CHANGING_TOOLS], messages)
        reply = interrupt({"type": "confirmation", "recap": recap})

        decision = decision_llm.invoke(
            [SystemMessage(CONFIRMATION_PROMPT.format(recap=recap)), HumanMessage(str(reply))]
        ).decision
        if decision == "confirm":
            return {"confirmed": True}
        return {
            "messages": [
                *_not_executed(calls, "Non exécuté : le client n'a pas confirmé cette action."),
                HumanMessage(content=str(reply)),
            ],
            "confirmed": False,
        }

    def route_after_confirm(state: AgentState) -> str:
        return "tools" if state.get("confirmed") else "agent"

    def verify_node(state: AgentState) -> dict:
        messages = state["messages"]
        draft = messages[-1]
        answer = _text(draft).strip()
        if not answer:
            issues = ["La réponse est vide."]
        else:
            verdict = verify_llm.invoke(
                [
                    SystemMessage(VERIFY_PROMPT),
                    HumanMessage(
                        f"QUESTION DU CLIENT :\n{_last_user_text(messages)}\n\n"
                        f"PREUVES :\n{_evidence(messages)}\n\nRÉPONSE :\n{answer}"
                    ),
                ]
            )
            issues = [] if verdict.ok else (verdict.issues or ["Réponse non conforme."])
        if not issues:
            return {"critique": None}
        logger.info("VERIFY a rejeté une réponse: %s | réponse: %r", "; ".join(issues), answer)

        attempts = state.get("verify_attempts", 0) + 1
        update: dict = {
            "messages": [RemoveMessage(id=draft.id)],
            "verify_attempts": attempts,
            "critique": CRITIQUE_PROMPT.format(issues="\n".join(f"- {i}" for i in issues)),
        }
        if attempts > MAX_VERIFY_RETRIES:
            update["escalation"] = {
                "category": "other",
                "urgency": "normal",
                "summary": f"Réponse automatique non vérifiable. Demande du client : {_last_user_text(messages)}",
            }
        return update

    def route_after_verify(state: AgentState) -> str:
        if state.get("escalation"):
            return "escalate"
        return "agent" if state.get("critique") else "end"

    def escalate_node(state: AgentState) -> dict:
        esc = state.get("escalation") or {}
        messages = state["messages"]
        last = messages[-1]
        pending = (
            _not_executed(last.tool_calls, "Non exécuté : dossier transféré à un conseiller.")
            if isinstance(last, AIMessage) and last.tool_calls
            else []
        )

        verified = verified_customer_ids(messages)
        customer_id = esc.get("customer_id") or (next(iter(verified)) if len(verified) == 1 else None)
        used_tools = sorted({c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls})
        # Une réservation/annulation déjà exécutée dans ce tour reste valable même si l'escalade
        # jette le brouillon du LLM qui l'annonçait : le conseiller et le client doivent le savoir.
        changes = _unreported_state_changes(messages)
        actions_taken = " ".join(
            [*changes, *([f"Outils consultés : {', '.join(used_tools)}"] if used_tools else [])]
        ) or None
        ticket = ticket_tool.invoke(
            {
                "category": esc.get("category", "other"),
                "summary": (esc.get("summary") or f"Demande du client : {_last_user_text(messages)}")
                + ("" if not changes else " " + " ".join(changes)),
                "customer_id": customer_id,
                "urgency": esc.get("urgency", "normal"),
                "actions_taken": esc.get("actions_taken") or actions_taken,
            }
        )
        prefix = (" ".join(changes) + " ") if changes else ""
        if ticket.get("ok"):
            reply = prefix + HANDOVER_MESSAGE.format(ticket_id=ticket["data"]["ticket_id"])
        else:
            logger.error("Création du ticket d'escalade impossible: %s", ticket)
            reply = prefix + HANDOVER_FAILED_MESSAGE
        return {
            "messages": [*pending, AIMessage(content=reply)],
            "escalated": bool(ticket.get("ok")),
            "escalation": {**esc, "customer_id": customer_id, "ticket": ticket},
        }

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(executable))
    graph.add_node("confirm", confirm_node)
    graph.add_node("verify", verify_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("handed_over", handed_over_node)

    graph.add_conditional_edges(START, route_start, {"triage": "triage", "handed_over": "handed_over"})
    graph.add_conditional_edges("triage", route_after_triage, {"escalate": "escalate", "agent": "agent"})
    graph.add_conditional_edges(
        "agent", route_after_agent, {"escalate": "escalate", "verify": "verify", "confirm": "confirm", "tools": "tools"}
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("confirm", route_after_confirm, {"tools": "tools", "agent": "agent"})
    graph.add_conditional_edges("verify", route_after_verify, {"end": END, "agent": "agent", "escalate": "escalate"})
    graph.add_edge("escalate", END)
    graph.add_edge("handed_over", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
