from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_store
from ..errors import CustomerNotFound, PhoneMismatch
from ..schemas import Customer, Invoice
from ..store import DataStore

router = APIRouter(prefix="/customers", tags=["customers"])

PHONE_QUERY = Query(..., description="Téléphone du client, pour vérification d'identité")


@router.get("/{customer_id}", response_model=Customer)
def get_customer(customer_id: str, phone: str = PHONE_QUERY, store: DataStore = Depends(get_store)) -> Customer:
    try:
        return store.get_customer(customer_id, phone=phone)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PhoneMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/{customer_id}/invoices", response_model=list[Invoice])
def get_customer_invoices(
    customer_id: str, phone: str = PHONE_QUERY, store: DataStore = Depends(get_store)
) -> list[Invoice]:
    try:
        customer = store.get_customer(customer_id, phone=phone)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PhoneMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return customer.last_invoices
