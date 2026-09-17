"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with local-development defaults."""

    openai_api_key: str
    postgres_url: str = "postgresql+psycopg://citecraft:citecraft@localhost:6024/citecraft"
    redis_url: str = "redis://localhost:6379/0"
    collection_name: str = "citecraft_documents"
    chat_model: str = "gpt-4o-mini"
    openai_store_responses: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1_536
    top_k: int = 5
    partition_strategy: str = "hi_res"
    max_context_characters: int = 24_000
    summary_concurrency: int = 4

    @classmethod
    def from_env(cls, *, api_key: str | None = None) -> Settings:
        load_dotenv()
        return cls(
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            postgres_url=os.getenv(
                "POSTGRES_URL",
                "postgresql+psycopg://citecraft:citecraft@localhost:6024/citecraft",
            ),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            collection_name=os.getenv("PGVECTOR_COLLECTION", "citecraft_documents"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            openai_store_responses=_boolean("OPENAI_STORE_RESPONSES", True),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=_positive_int("EMBEDDING_DIMENSIONS", 1_536),
            top_k=_positive_int("TOP_K", 5),
            partition_strategy=os.getenv("PARTITION_STRATEGY", "hi_res"),
            max_context_characters=_positive_int("MAX_CONTEXT_CHARACTERS", 24_000),
            summary_concurrency=_positive_int("SUMMARY_CONCURRENCY", 4),
        )

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if self.partition_strategy not in {"auto", "fast", "hi_res", "ocr_only"}:
            raise ValueError(
                "PARTITION_STRATEGY must be one of: auto, fast, hi_res, ocr_only."
            )


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw_value!r}.") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, str(default)).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, off; "
        f"received {raw_value!r}."
    )
