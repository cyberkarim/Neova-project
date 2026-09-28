import httpx
import pytest

from agent.tools import build_api_tools


@pytest.fixture()
def tools(client):
    built = build_api_tools(client, backoff_seconds=0)
    return {t.name: t for t in built}


def test_get_customer_success(tools):
    result = tools["get_customer"].invoke({"customer_id": "NEO-88213", "phone": "0612840193"})
    assert result["ok"] is True
    assert result["data"]["full_name"] == "Camille Rousseau"


def test_get_customer_wrong_phone_returns_structured_error(tools):
    result = tools["get_customer"].invoke({"customer_id": "NEO-88213", "phone": "0000000000"})
    assert result["ok"] is False
    assert result["status"] == 403
    assert result["retryable"] is False


def test_book_slot_conflict_is_not_retried_and_readable(tools):
    result = tools["book_appointment"].invoke(
        {"customer_id": "NEO-88213", "slot_id": "SLOT-7A32", "reason": "no_internet"}
    )
    assert result["ok"] is False
    assert result["status"] == 409
    assert "déjà réservé" in result["error"]


def test_book_then_cancel_roundtrip(tools):
    booked = tools["book_appointment"].invoke(
        {"customer_id": "NEO-88213", "slot_id": "SLOT-7A31", "reason": "no_internet"}
    )
    assert booked["ok"] is True
    cancelled = tools["cancel_appointment"].invoke({"appointment_id": booked["data"]["appointment_id"]})
    assert cancelled["data"]["status"] == "cancelled"


def test_create_escalation_ticket(tools):
    result = tools["create_escalation_ticket"].invoke(
        {"category": "technical", "summary": "Panne persistante", "customer_id": "NEO-88213", "urgency": "high"}
    )
    assert result["ok"] is True
    assert result["data"]["ticket_id"].startswith("TCK-")


def test_server_error_is_retried_then_reported_as_retryable():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "boom"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake")
    tool = {t.name: t for t in build_api_tools(http, max_attempts=3, backoff_seconds=0)}["list_network_incidents"]

    result = tool.invoke({})
    assert calls["n"] == 3
    assert result["ok"] is False
    assert result["retryable"] is True


def test_transient_failure_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connexion refusée")
        return httpx.Response(200, json=[])

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake")
    tool = {t.name: t for t in build_api_tools(http, max_attempts=3, backoff_seconds=0)}["list_network_incidents"]

    assert tool.invoke({}) == {"ok": True, "data": []}
    assert calls["n"] == 2
