import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.store import DataStore

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_PATH = REPO_ROOT / "data" / "neova_data.json"


@pytest.fixture(scope="session")
def chunks():
    """Chunks des PDF du corpus (le parsing prend quelques secondes : partagé par toute la session)."""
    from agent.ingest import load_corpus

    return load_corpus()


@pytest.fixture()
def client(tmp_path):
    """TestClient dont le store pointe vers une copie jetable du fixture JSON.

    Isole chaque test (aucune écriture dans data/runtime_state.json) tout en
    exerçant le vrai code de chargement/persistance de DataStore.
    """
    base_copy = tmp_path / "neova_data.json"
    shutil.copy(BASE_DATA_PATH, base_copy)
    runtime_path = tmp_path / "runtime_state.json"

    with TestClient(app) as test_client:
        test_client.app.state.store = DataStore(base_path=base_copy, runtime_path=runtime_path)
        yield test_client
