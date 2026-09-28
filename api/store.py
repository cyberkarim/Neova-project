"""Store en mémoire pour les données Néova Télécom.

data/neova_data.json est traité comme un fixture en lecture seule : on ne le
réécrit jamais. Les mutations (rendez-vous pris/annulés, disponibilité des
créneaux, tickets créés) sont conservées en mémoire et journalisées dans
data/runtime_state.json (overlay, écrit de façon atomique) afin de survivre à
un redémarrage sans jamais toucher au jeu de données fourni.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .errors import (
    AppointmentAlreadyCancelled,
    AppointmentNotFound,
    CustomerNotFound,
    IncidentNotFound,
    PhoneMismatch,
    PostalCodeMismatch,
    SlotNotFound,
    SlotUnavailable,
    TicketNotFound,
)
from .schemas import (
    Appointment,
    AppointmentCreate,
    Customer,
    NetworkIncident,
    TechnicianSlot,
    Ticket,
    TicketCreate,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DATA_PATH = DATA_DIR / "neova_data.json"
RUNTIME_STATE_PATH = DATA_DIR / "runtime_state.json"


class DataStore:
    def __init__(self, base_path: Path = BASE_DATA_PATH, runtime_path: Path = RUNTIME_STATE_PATH):
        self._base_path = base_path
        self._runtime_path = runtime_path
        self._lock = threading.Lock()
        self._customers: dict[str, Customer] = {}
        self._incidents: dict[str, NetworkIncident] = {}
        self._slots: dict[str, TechnicianSlot] = {}
        self._appointments: dict[str, Appointment] = {}
        self._tickets: dict[str, Ticket] = {}
        self._load()

    # ---- chargement -------------------------------------------------

    def _load(self) -> None:
        raw = json.loads(self._base_path.read_text(encoding="utf-8"))
        self._customers = {c["customer_id"]: Customer(**c) for c in raw["customers"]}
        self._incidents = {i["incident_id"]: NetworkIncident(**i) for i in raw["network_incidents"]}
        self._slots = {s["slot_id"]: TechnicianSlot(**s) for s in raw["technician_slots"]}
        self._appointments = {}
        self._tickets = {}

        if self._runtime_path.exists():
            runtime = json.loads(self._runtime_path.read_text(encoding="utf-8"))
            for slot_id, available in runtime.get("slot_availability", {}).items():
                if slot_id in self._slots:
                    self._slots[slot_id] = self._slots[slot_id].model_copy(update={"available": available})
            for a in runtime.get("appointments", []):
                appt = Appointment(**a)
                self._appointments[appt.appointment_id] = appt
            for t in runtime.get("tickets", []):
                tk = Ticket(**t)
                self._tickets[tk.ticket_id] = tk

    def _persist(self) -> None:
        payload = {
            "slot_availability": {sid: s.available for sid, s in self._slots.items()},
            "appointments": [a.model_dump(mode="json") for a in self._appointments.values()],
            "tickets": [t.model_dump(mode="json") for t in self._tickets.values()],
        }
        tmp_path = self._runtime_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._runtime_path)

    # ---- clients ------------------------------------------------------

    def get_customer(self, customer_id: str, *, phone: str) -> Customer:
        customer = self._customers.get(customer_id)
        if customer is None:
            raise CustomerNotFound(customer_id)
        if customer.phone != phone:
            raise PhoneMismatch(customer_id)
        return customer

    # ---- incidents ------------------------------------------------------

    def list_incidents(self, *, postal_code: str | None = None) -> list[NetworkIncident]:
        incidents = list(self._incidents.values())
        if postal_code:
            incidents = [i for i in incidents if postal_code in i.postal_codes]
        return incidents

    def get_incident(self, incident_id: str) -> NetworkIncident:
        incident = self._incidents.get(incident_id)
        if incident is None:
            raise IncidentNotFound(incident_id)
        return incident

    # ---- créneaux techniciens --------------------------------------------

    def list_slots(self, *, postal_code: str | None = None, available_only: bool = True) -> list[TechnicianSlot]:
        slots = list(self._slots.values())
        if postal_code:
            slots = [s for s in slots if postal_code in s.postal_codes]
        if available_only:
            slots = [s for s in slots if s.available]
        return sorted(slots, key=lambda s: s.start)

    # ---- rendez-vous (state-changing) ------------------------------------

    def book_appointment(self, payload: AppointmentCreate) -> Appointment:
        with self._lock:
            customer = self._customers.get(payload.customer_id)
            if customer is None:
                raise CustomerNotFound(payload.customer_id)

            slot = self._slots.get(payload.slot_id)
            if slot is None:
                raise SlotNotFound(payload.slot_id)
            if not slot.available:
                raise SlotUnavailable(payload.slot_id)
            if customer.postal_code not in slot.postal_codes:
                raise PostalCodeMismatch(customer.postal_code, payload.slot_id)

            self._slots[slot.slot_id] = slot.model_copy(update={"available": False})
            appointment = Appointment(
                appointment_id=f"APT-{uuid4().hex[:8].upper()}",
                customer_id=payload.customer_id,
                slot_id=payload.slot_id,
                reason=payload.reason,
                notes=payload.notes,
                status="confirmed",
                created_at=datetime.now(timezone.utc),
            )
            self._appointments[appointment.appointment_id] = appointment
            self._persist()
            return appointment

    def get_appointment(self, appointment_id: str) -> Appointment:
        appointment = self._appointments.get(appointment_id)
        if appointment is None:
            raise AppointmentNotFound(appointment_id)
        return appointment

    def cancel_appointment(self, appointment_id: str) -> Appointment:
        with self._lock:
            appointment = self._appointments.get(appointment_id)
            if appointment is None:
                raise AppointmentNotFound(appointment_id)
            if appointment.status == "cancelled":
                raise AppointmentAlreadyCancelled(appointment_id)

            cancelled = appointment.model_copy(update={"status": "cancelled"})
            self._appointments[appointment_id] = cancelled

            slot = self._slots.get(appointment.slot_id)
            if slot is not None:
                self._slots[slot.slot_id] = slot.model_copy(update={"available": True})

            self._persist()
            return cancelled

    # ---- tickets d'escalade ----------------------------------------------

    def create_ticket(self, payload: TicketCreate) -> Ticket:
        with self._lock:
            if payload.customer_id is not None and payload.customer_id not in self._customers:
                raise CustomerNotFound(payload.customer_id)

            ticket = Ticket(
                ticket_id=f"TCK-{uuid4().hex[:8].upper()}",
                category=payload.category,
                summary=payload.summary,
                customer_id=payload.customer_id,
                urgency=payload.urgency,
                actions_taken=payload.actions_taken,
                created_at=datetime.now(timezone.utc),
            )
            self._tickets[ticket.ticket_id] = ticket
            self._persist()
            return ticket

    def get_ticket(self, ticket_id: str) -> Ticket:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketNotFound(ticket_id)
        return ticket
