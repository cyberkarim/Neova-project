import itertools

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent.graph import (
    ALREADY_HANDED_OVER_MESSAGE,
    ConfirmationDecision,
    Triage,
    Verdict,
    build_graph,
)
from agent.session import TECHNICAL_ERROR_MESSAGE, Conversation
from agent.tools import build_api_tools

_ids = itertools.count(1)


def ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


class ScriptedAgent:
    """Joue les réponses prévues du LLM principal et garde les prompts reçus."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(list(messages))
        response = self.script.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedJudge:
    """Joue les décisions du LLM à sortie structurée (triage, vérification, confirmation)."""

    DEFAULTS = {Triage: Triage(mandatory_escalation=False), Verdict: Verdict(ok=True)}

    def __init__(self, triage=(), verdicts=(), decisions=()):
        self.queues = {Triage: list(triage), Verdict: list(verdicts), ConfirmationDecision: list(decisions)}
        self.calls = {Triage: 0, Verdict: 0, ConfirmationDecision: 0}

    def with_structured_output(self, schema, **kwargs):
        judge = self

        class Structured:
            def invoke(self, messages):
                judge.calls[schema] += 1
                queue = judge.queues[schema]
                if queue:
                    return queue.pop(0)
                return judge.DEFAULTS.get(schema, ConfirmationDecision(decision="other"))

        return Structured()


@tool
def search_knowledge_base(query: str) -> dict:
    """Recherche documentaire (double de test)."""
    return {"ok": True, "results": [{"source": "grille-tarifaire-2026", "text": "Fibre 500 Mb/s : 29,99 €/mois"}]}


@pytest.fixture()
def make_conversation(client):
    def make(agent_script, judge=None):
        agent = ScriptedAgent(agent_script)
        judge = judge or ScriptedJudge()
        tools = build_api_tools(client, backoff_seconds=0)
        graph = build_graph(agent, judge, tools, search_tool=search_knowledge_base)
        return Conversation(graph, thread_id=f"t{next(_ids)}"), agent, judge, graph

    return make


def slots_available(client, postal_code="75019"):
    return {s["slot_id"] for s in client.get("/appointments/slots", params={"postal_code": postal_code}).json()}


def test_documented_question_goes_through_search_then_verification(make_conversation):
    conv, agent, judge, _ = make_conversation(
        [call("search_knowledge_base", query="prix fibre 500"), ai("La Fibre 500 Mb/s coûte 29,99 € par mois.")]
    )

    result = conv.send("Combien coûte la fibre 500 ?")

    assert result.reply == "La Fibre 500 Mb/s coûte 29,99 € par mois."
    assert result.tools_called == ["search_knowledge_base"]
    assert not result.escalated
    assert judge.calls[Verdict] == 1


def test_mandatory_escalation_skips_the_agent_and_creates_a_ticket(make_conversation, client):
    triage = Triage(
        mandatory_escalation=True,
        situation="rgpd",
        category="other",
        urgency="high",
        summary="Demande de suppression de données personnelles",
    )
    conv, agent, _, _ = make_conversation([], ScriptedJudge(triage=[triage]))

    result = conv.send("Je demande la suppression de toutes mes données, conformément au RGPD.")

    assert not result.error
    assert result.escalated
    assert "45 minutes" in result.reply
    assert agent.prompts == []
    ticket = client.get(f"/tickets/{result.ticket['data']['ticket_id']}").json()
    assert ticket["urgency"] == "high"
    assert "rgpd" in ticket["summary"]

    follow_up = conv.send("Vous avez bien reçu ma demande ?")
    assert follow_up.reply == ALREADY_HANDED_OVER_MESSAGE
    assert agent.prompts == []


def test_non_sensitive_escalation_does_not_lock_the_rest_of_the_conversation(make_conversation):
    triage_pro = Triage(
        mandatory_escalation=True,
        situation="contrat_pro",
        category="other",
        urgency="normal",
        summary="Question sur un contrat professionnel",
    )
    conv, agent, _, _ = make_conversation(
        [call("search_knowledge_base", query="tarif appel Etats-Unis"), ai("0,50 € par minute.")],
        ScriptedJudge(triage=[triage_pro]),
    )

    first = conv.send("Quel est le prix de l'offre professionnelle ?")
    assert first.escalated
    assert not first.error

    second = conv.send("Combien coûte un appel émis depuis les États-Unis ?")

    assert second.reply == "0,50 € par minute."
    assert not second.escalated
    assert agent.prompts, "l'agent doit être sollicité pour la question suivante, sans rapport"


def test_agent_handover_tool_creates_ticket_for_the_verified_customer(make_conversation, client):
    conv, _, _, _ = make_conversation(
        [
            call("get_customer", customer_id="NEO-88213", phone="0612840193"),
            call("request_human_handover", category="billing_dispute", summary="Contestation de 55 €", urgency="high"),
        ]
    )

    result = conv.send("Je conteste ma facture de 55 €. Client NEO-88213, tél 0612840193.")

    assert result.escalated
    ticket = client.get(f"/tickets/{result.ticket['data']['ticket_id']}").json()
    assert ticket["customer_id"] == "NEO-88213"
    assert ticket["category"] == "billing_dispute"


def test_booking_without_verified_identity_is_refused_before_any_confirmation(make_conversation, client):
    conv, agent, _, _ = make_conversation(
        [
            call("book_appointment", customer_id="NEO-88213", slot_id="SLOT-7A31", reason="no_internet"),
            ai("Pouvez-vous me donner votre identifiant client et votre numéro de téléphone ?"),
        ]
    )

    result = conv.send("Réservez-moi le créneau SLOT-7A31, je suis NEO-88213.")

    assert not result.awaiting_confirmation
    assert "SLOT-7A31" in slots_available(client)
    tool_messages = [m for m in agent.prompts[1] if isinstance(m, ToolMessage)]
    assert "Identité non vérifiée" in tool_messages[-1].content


def test_booking_is_executed_only_after_explicit_confirmation(make_conversation, client):
    conv, _, _, _ = make_conversation(
        [
            call("get_customer", customer_id="NEO-88213", phone="0612840193"),
            call("list_technician_slots", postal_code="75019"),
            call("book_appointment", customer_id="NEO-88213", slot_id="SLOT-7A31", reason="no_internet"),
            ai("Votre rendez-vous est confirmé."),
        ],
        ScriptedJudge(decisions=[ConfirmationDecision(decision="confirm")]),
    )

    first = conv.send("Je suis NEO-88213, tél 0612840193, je veux un technicien, le créneau SLOT-7A31.")

    assert first.awaiting_confirmation
    assert "SLOT-7A31" in first.reply and "69 €" in first.reply
    assert "SLOT-7A31" in slots_available(client)

    second = conv.send("oui")

    assert second.reply == "Votre rendez-vous est confirmé."
    assert "book_appointment" in second.tools_called
    assert "SLOT-7A31" not in slots_available(client)


def test_declined_booking_changes_nothing_and_keeps_history_valid(make_conversation, client):
    conv, agent, _, _ = make_conversation(
        [
            call("get_customer", customer_id="NEO-88213", phone="0612840193"),
            call("book_appointment", customer_id="NEO-88213", slot_id="SLOT-7A31", reason="no_internet"),
            ai("D'accord, je ne réserve rien."),
        ],
        ScriptedJudge(decisions=[ConfirmationDecision(decision="decline")]),
    )

    assert conv.send("NEO-88213, 0612840193, réservez SLOT-7A31").awaiting_confirmation
    result = conv.send("non merci")

    assert result.reply == "D'accord, je ne réserve rien."
    assert "SLOT-7A31" in slots_available(client)
    last_prompt = agent.prompts[-1]
    assert isinstance(last_prompt[-3], AIMessage) and last_prompt[-3].tool_calls
    assert isinstance(last_prompt[-2], ToolMessage)
    assert isinstance(last_prompt[-1], HumanMessage) and last_prompt[-1].content == "non merci"


def test_cannot_cancel_another_customers_appointment(make_conversation, client):
    booked = client.post(
        "/appointments", json={"customer_id": "NEO-88213", "slot_id": "SLOT-7A31", "reason": "no_internet"}
    ).json()
    conv, _, _, _ = make_conversation(
        [
            call("get_customer", customer_id="NEO-10467", phone="0778115402"),
            call("cancel_appointment", appointment_id=booked["appointment_id"]),
            ai("Je ne peux pas annuler ce rendez-vous."),
        ]
    )

    result = conv.send("Je suis NEO-10467, tél 0778115402, annulez le rendez-vous " + booked["appointment_id"])

    assert not result.awaiting_confirmation
    assert client.get(f"/appointments/{booked['appointment_id']}").json()["status"] == "confirmed"


def test_rejected_answer_is_regenerated_with_the_critique(make_conversation):
    judge = ScriptedJudge(verdicts=[Verdict(ok=False, issues=["prix non prouvé"]), Verdict(ok=True)])
    conv, agent, _, graph = make_conversation([ai("Ça coûte 10 €."), ai("Je n'ai pas cette information.")], judge)

    result = conv.send("Combien coûte l'option X ?")

    assert result.reply == "Je n'ai pas cette information."
    assert not result.escalated
    critique = [m for m in agent.prompts[1] if isinstance(m, SystemMessage) and "Contrôle qualité" in m.content]
    assert critique and "prix non prouvé" in critique[0].content
    history = graph.get_state(conv._config).values["messages"]
    assert all(m.content != "Ça coûte 10 €." for m in history)


def test_answer_rejected_twice_escalates_instead_of_sending_it(make_conversation, client):
    judge = ScriptedJudge(verdicts=[Verdict(ok=False, issues=["a"]), Verdict(ok=False, issues=["b"])])
    conv, _, _, graph = make_conversation([ai("Ça coûte 10 €."), ai("Ça coûte 12 €.")], judge)

    result = conv.send("Combien coûte l'option X ?")

    assert result.escalated
    assert "45 minutes" in result.reply
    ticket = client.get(f"/tickets/{result.ticket['data']['ticket_id']}").json()
    assert "Combien coûte l'option X ?" in ticket["summary"]
    history = graph.get_state(conv._config).values["messages"]
    assert not any(isinstance(m, AIMessage) and m.content in {"Ça coûte 10 €.", "Ça coûte 12 €."} for m in history)


def test_escalation_after_a_successful_booking_still_reports_the_booking(make_conversation, client):
    judge = ScriptedJudge(
        decisions=[ConfirmationDecision(decision="confirm")],
        verdicts=[Verdict(ok=False, issues=["a"]), Verdict(ok=False, issues=["b"])],
    )
    conv, _, _, _ = make_conversation(
        [
            call("get_customer", customer_id="NEO-88213", phone="0612840193"),
            call("book_appointment", customer_id="NEO-88213", slot_id="SLOT-7A31", reason="no_internet"),
            ai("Votre rendez-vous est confirmé pour le 27 août."),
            ai("C'est noté, votre technicien passera le 27 août."),
        ],
        judge,
    )

    assert conv.send("NEO-88213, 0612840193, réservez SLOT-7A31").awaiting_confirmation
    result = conv.send("oui")

    assert result.escalated
    assert "SLOT-7A31" not in slots_available(client)  # la réservation reste valable
    assert "déjà réservé" in result.reply
    assert "APT-" in result.reply
    ticket = client.get(f"/tickets/{result.ticket['data']['ticket_id']}").json()
    assert "déjà réservé" in ticket["summary"]
    assert "déjà réservé" in ticket["actions_taken"]


def test_runaway_tool_loop_escalates_with_a_valid_history(make_conversation):
    conv, _, _, graph = make_conversation([call("list_network_incidents") for _ in range(9)])

    result = conv.send("Y a-t-il des incidents ?")

    assert result.escalated
    history = graph.get_state(conv._config).values["messages"]
    tool_call_ids = {c["id"] for m in history if isinstance(m, AIMessage) for c in m.tool_calls}
    answered_ids = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    assert tool_call_ids == answered_ids


def test_llm_outage_returns_a_safe_message(make_conversation):
    conv, _, _, _ = make_conversation([RuntimeError("openrouter indisponible")])

    result = conv.send("Bonjour")

    assert result.error
    assert result.reply == TECHNICAL_ERROR_MESSAGE
