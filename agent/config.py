import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    chat_model: str
    embedding_model: str
    judge_model: str
    api_url: str


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variable d'environnement manquante: {name} (voir .env.example)")
    return value


def load_settings() -> Settings:
    chat_model = _require("CHAT_MODEL")
    return Settings(
        openrouter_api_key=_require("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        chat_model=chat_model,
        embedding_model=_require("EMBEDDING_MODEL"),
        judge_model=os.getenv("JUDGE_MODEL") or chat_model,
        api_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    )
