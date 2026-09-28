from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_store
from ..errors import (
    AppointmentAlreadyCancelled,
    AppointmentNotFound,
    CustomerNotFound,
    PostalCodeMismatch,
    SlotNotFound,
    SlotUnavailable,
)
from ..schemas import Appointment, AppointmentCreate, TechnicianSlot
from ..store import DataStore

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.get("/slots", response_model=list[TechnicianSlot])
def list_slots(
    postal_code: str | None = Query(None, description="Filtrer par code postal"),
    available_only: bool = Query(True, description="Ne renvoyer que les créneaux libres"),
    store: DataStore = Depends(get_store),
) -> list[TechnicianSlot]:
    return store.list_slots(postal_code=postal_code, available_only=available_only)


@router.post("", response_model=Appointment, status_code=201)
def book_appointment(payload: AppointmentCreate, store: DataStore = Depends(get_store)) -> Appointment:
    """Réserve un créneau technicien pour un client.

    C'est l'endpoint state-changing du service : il doit être appelé par
    l'agent seulement après confirmation explicite du client sur le créneau
    proposé, jamais de façon spéculative.
    """
    try:
        return store.book_appointment(payload)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SlotNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SlotUnavailable as exc:
        # 409 : le client (ou l'agent) doit re-proposer un autre créneau, pas retenter le même appel.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PostalCodeMismatch as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{appointment_id}", response_model=Appointment)
def get_appointment(appointment_id: str, store: DataStore = Depends(get_store)) -> Appointment:
    try:
        return store.get_appointment(appointment_id)
    except AppointmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{appointment_id}", response_model=Appointment)
def cancel_appointment(appointment_id: str, store: DataStore = Depends(get_store)) -> Appointment:
    try:
        return store.cancel_appointment(appointment_id)
    except AppointmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AppointmentAlreadyCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
