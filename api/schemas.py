"""Modèles Pydantic pour l'API Néova Télécom.

Reflètent tels quels les objets de data/neova_data.json, plus les modèles de
requête/réponse pour les ressources créées à l'exécution (rendez-vous, tickets).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

InvoiceStatus = Literal["paid", "unpaid", "pending"]
IncidentStatus = Literal["outage", "degraded", "maintenance", "resolved"]
AppointmentReason = Literal["no_internet", "slow_internet", "installation", "equipment_swap"]
AppointmentStatus = Literal["confirmed", "cancelled"]
EscalationCategory = Literal["billing_dispute", "technical", "termination", "commercial_gesture", "other"]
Urgency = Literal["low", "normal", "high", "immediate"]


class Invoice(BaseModel):
    invoice_id: str
    date: date
    amount: float
    status: InvoiceStatus


class Customer(BaseModel):
    customer_id: str
    full_name: str
    phone: str
    plan: str
    monthly_price: float
    contract_start_date: date
    engagement_months: int
    address: str
    postal_code: str
    balance_due: float
    open_incident_id: str | None = None
    equipment: list[str] = Field(default_factory=list)
    last_invoices: list[Invoice] = Field(default_factory=list)


class NetworkIncident(BaseModel):
    incident_id: str
    postal_codes: list[str]
    status: IncidentStatus
    cause: str
    affected_customers: int
    started_at: datetime
    estimated_resolution: datetime | None = None


class TechnicianSlot(BaseModel):
    slot_id: str
    postal_codes: list[str]
    start: datetime
    end: datetime
    available: bool


class AppointmentCreate(BaseModel):
    customer_id: str
    slot_id: str
    reason: AppointmentReason
    notes: str | None = None


class Appointment(BaseModel):
    appointment_id: str
    customer_id: str
    slot_id: str
    reason: AppointmentReason
    notes: str | None = None
    status: AppointmentStatus
    created_at: datetime


class TicketCreate(BaseModel):
    category: EscalationCategory
    summary: str
    customer_id: str | None = None
    urgency: Urgency = "normal"
    actions_taken: str | None = None


class Ticket(BaseModel):
    ticket_id: str
    category: EscalationCategory
    summary: str
    customer_id: str | None = None
    urgency: Urgency
    actions_taken: str | None = None
    created_at: datetime
