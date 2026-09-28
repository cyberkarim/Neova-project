"""Exceptions métier levées par le store, traduites en codes HTTP par les routers."""


class CustomerNotFound(Exception):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__(f"Client introuvable: {customer_id}")


class PhoneMismatch(Exception):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__("Le numéro de téléphone ne correspond pas à ce client")


class IncidentNotFound(Exception):
    def __init__(self, incident_id: str):
        self.incident_id = incident_id
        super().__init__(f"Incident introuvable: {incident_id}")


class SlotNotFound(Exception):
    def __init__(self, slot_id: str):
        self.slot_id = slot_id
        super().__init__(f"Créneau introuvable: {slot_id}")


class SlotUnavailable(Exception):
    def __init__(self, slot_id: str):
        self.slot_id = slot_id
        super().__init__(f"Créneau déjà réservé: {slot_id}")


class PostalCodeMismatch(Exception):
    def __init__(self, postal_code: str, slot_id: str):
        self.postal_code = postal_code
        self.slot_id = slot_id
        super().__init__(
            f"Le créneau {slot_id} ne dessert pas le secteur {postal_code}"
        )


class AppointmentNotFound(Exception):
    def __init__(self, appointment_id: str):
        self.appointment_id = appointment_id
        super().__init__(f"Rendez-vous introuvable: {appointment_id}")


class AppointmentAlreadyCancelled(Exception):
    def __init__(self, appointment_id: str):
        self.appointment_id = appointment_id
        super().__init__(f"Rendez-vous déjà annulé: {appointment_id}")


class TicketNotFound(Exception):
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        super().__init__(f"Ticket introuvable: {ticket_id}")
