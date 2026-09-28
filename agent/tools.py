"""Tools LangChain qui encapsulent l'API FastAPI Néova.

Aucun tool ne lève d'exception : toute erreur devient un dict `{"ok": False, ...}`
que le LLM peut lire et sur lequel il peut raisonner (reproposer un créneau,
s'excuser, escalader). Seuls les 5xx et les erreurs réseau sont retentés.
"""
from __future__ import annotations

import time
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool, tool

AppointmentReason = Literal["no_internet", "slow_internet", "installation", "equipment_swap"]
EscalationCategory = Literal["billing_dispute", "technical", "termination", "commercial_gesture", "other"]
Urgency = Literal["low", "normal", "high", "immediate"]

STATE_CHANGING_TOOLS = frozenset({"book_appointment", "cancel_appointment"})


class ApiCaller:
    def __init__(self, http: httpx.Client, *, max_attempts: int = 3, backoff_seconds: float = 0.5):
        self._http = http
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        last_error = "erreur inconnue"
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_error = f"service injoignable: {exc}"
            else:
                if response.status_code < 400:
                    return {"ok": True, "data": response.json()}
                detail = _extract_detail(response)
                if response.status_code < 500:
                    return {"ok": False, "status": response.status_code, "error": detail, "retryable": False}
                last_error = detail
            if attempt < self._max_attempts:
                time.sleep(self._backoff_seconds * 2 ** (attempt - 1))
        return {"ok": False, "status": 503, "error": last_error, "retryable": True}


def _extract_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    return detail if isinstance(detail, str) else str(detail)


def build_api_tools(http: httpx.Client, *, max_attempts: int = 3, backoff_seconds: float = 0.5) -> list[BaseTool]:
    api = ApiCaller(http, max_attempts=max_attempts, backoff_seconds=backoff_seconds)

    @tool
    def get_customer(customer_id: str, phone: str) -> dict:
        """Récupère la fiche d'un client (offre, adresse, solde dû, équipements, incident ouvert).
        Le numéro de téléphone du client est obligatoire pour vérifier son identité."""
        return api.call("GET", f"/customers/{customer_id}", params={"phone": phone})

    @tool
    def get_customer_invoices(customer_id: str, phone: str) -> dict:
        """Récupère les dernières factures d'un client (montant, date, statut payé/impayé/en attente).
        Le numéro de téléphone du client est obligatoire pour vérifier son identité."""
        return api.call("GET", f"/customers/{customer_id}/invoices", params={"phone": phone})

    @tool
    def list_network_incidents(postal_code: str | None = None) -> dict:
        """Liste les incidents réseau connus (panne, dégradation, maintenance), filtrables par code postal."""
        params = {"postal_code": postal_code} if postal_code else None
        return api.call("GET", "/incidents", params=params)

    @tool
    def list_technician_slots(postal_code: str) -> dict:
        """Liste les créneaux techniciens encore disponibles pour un code postal, triés par date."""
        return api.call("GET", "/appointments/slots", params={"postal_code": postal_code})

    @tool
    def book_appointment(customer_id: str, slot_id: str, reason: AppointmentReason, notes: str | None = None) -> dict:
        """Réserve un créneau technicien pour un client. ACTION QUI MODIFIE L'ÉTAT : ne l'appeler
        qu'après confirmation explicite du client sur ce créneau précis."""
        return api.call(
            "POST",
            "/appointments",
            json={"customer_id": customer_id, "slot_id": slot_id, "reason": reason, "notes": notes},
        )

    @tool
    def get_appointment(appointment_id: str) -> dict:
        """Consulte un rendez-vous technicien (client, créneau, motif, statut)."""
        return api.call("GET", f"/appointments/{appointment_id}")

    @tool
    def cancel_appointment(appointment_id: str) -> dict:
        """Annule un rendez-vous technicien et libère le créneau. ACTION QUI MODIFIE L'ÉTAT :
        ne l'appeler qu'après confirmation explicite du client."""
        return api.call("DELETE", f"/appointments/{appointment_id}")

    @tool
    def create_escalation_ticket(
        category: EscalationCategory,
        summary: str,
        customer_id: str | None = None,
        urgency: Urgency = "normal",
        actions_taken: str | None = None,
    ) -> dict:
        """Transmet le dossier à un conseiller humain. Le résumé doit permettre au conseiller de
        reprendre sans reposer de questions au client."""
        return api.call(
            "POST",
            "/tickets",
            json={
                "category": category,
                "summary": summary,
                "customer_id": customer_id,
                "urgency": urgency,
                "actions_taken": actions_taken,
            },
        )

    return [
        get_customer,
        get_customer_invoices,
        list_network_incidents,
        list_technician_slots,
        get_appointment,
        book_appointment,
        cancel_appointment,
        create_escalation_ticket,
    ]
