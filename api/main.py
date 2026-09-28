from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import appointments, customers, incidents, tickets
from .store import DataStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = DataStore()
    yield


app = FastAPI(
    title="Néova Télécom — API interne",
    description="Expose les données clients, incidents réseau, créneaux techniciens et tickets d'escalade, à l'usage de l'agent de relation client.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(customers.router)
app.include_router(incidents.router)
app.include_router(appointments.router)
app.include_router(tickets.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
