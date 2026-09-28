from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_store
from ..errors import CustomerNotFound, TicketNotFound
from ..schemas import Ticket, TicketCreate
from ..store import DataStore

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=Ticket, status_code=201)
def create_ticket(payload: TicketCreate, store: DataStore = Depends(get_store)) -> Ticket:
    """Trace une escalade vers un conseiller humain (voir corpus/procedure-escalade-n2)."""
    try:
        return store.create_ticket(payload)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: str, store: DataStore = Depends(get_store)) -> Ticket:
    try:
        return store.get_ticket(ticket_id)
    except TicketNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
