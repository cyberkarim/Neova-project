from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_store
from ..errors import IncidentNotFound
from ..schemas import NetworkIncident
from ..store import DataStore

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.get("", response_model=list[NetworkIncident])
def list_incidents(
    postal_code: str | None = Query(None, description="Filtrer par code postal"),
    store: DataStore = Depends(get_store),
) -> list[NetworkIncident]:
    return store.list_incidents(postal_code=postal_code)


@router.get("/{incident_id}", response_model=NetworkIncident)
def get_incident(incident_id: str, store: DataStore = Depends(get_store)) -> NetworkIncident:
    try:
        return store.get_incident(incident_id)
    except IncidentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
