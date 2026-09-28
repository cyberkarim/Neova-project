from fastapi import Request

from .store import DataStore


def get_store(request: Request) -> DataStore:
    return request.app.state.store
