"""PostgreSQL/pgvector child index with Redis parent-document storage."""

from __future__ import annotations

import json
from collections.abc import Sequence

from citecraft.config import Settings
from citecraft.models import ExtractedElement, RetrievedElement


class DocumentRepository:
    """Persist summary vectors separately from their full parent elements."""

    INDEX_PREFIX = "citecraft:indexed:"
    PARENT_PREFIX = "citecraft:parent:"
    DOCUMENT_SET = "citecraft:documents"

    def __init__(self, settings: Settings) -> None:
        import redis
        from langchain_openai import OpenAIEmbeddings

        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self._postgres_url = _normalize_postgres_url(settings.postgres_url)
        self._table_name = settings.collection_name
        self._embedding_dimensions = settings.embedding_dimensions
        self._ensure_schema()

    def ping(self) -> None:
        self._redis.ping()
        self._postgres_ping()

    def indexed_metadata(self, file_hash: str) -> dict[str, object] | None:
        value = self._redis.get(f"{self.INDEX_PREFIX}{file_hash}")
        return json.loads(value) if value else None

    def add(self, elements: Sequence[ExtractedElement], summaries: Sequence[str]) -> None:
        if len(elements) != len(summaries):
            raise ValueError("Each extracted element must have one retrieval summary.")

        parent_mapping = {
            f"{self.PARENT_PREFIX}{element.parent_id}": json.dumps(_element_to_dict(element))
            for element in elements
        }
        self._redis.mset(parent_mapping)

        embeddings = self._embeddings.embed_documents(list(summaries))
        rows = [
            (
                element.parent_id,
                summary,
                _vector_literal(embedding),
                json.dumps(
                    {
                        "parent_id": element.parent_id,
                        "file_hash": element.file_hash,
                        "source": element.source,
                        "page": element.page or 0,
                        "kind": element.kind,
                    }
                ),
            )
            for element, summary, embedding in zip(
                elements, summaries, embeddings, strict=True
            )
        ]

        import psycopg
        from psycopg import sql

        statement = sql.SQL(
            """
            INSERT INTO {} (id, summary, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                summary = EXCLUDED.summary,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
            """
        ).format(sql.Identifier(self._table_name))
        with (
            psycopg.connect(self._postgres_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.executemany(statement, rows)

    def mark_indexed(self, *, file_hash: str, filename: str, element_count: int) -> None:
        metadata = json.dumps({"filename": filename, "element_count": element_count})
        pipeline = self._redis.pipeline()
        pipeline.set(f"{self.INDEX_PREFIX}{file_hash}", metadata)
        pipeline.sadd(self.DOCUMENT_SET, file_hash)
        pipeline.execute()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        allowed_hashes: Sequence[str],
    ) -> list[RetrievedElement]:
        if not allowed_hashes:
            return []

        import psycopg
        from psycopg import sql

        query_embedding = self._embeddings.embed_query(query)
        statement = sql.SQL(
            """
            SELECT metadata->>'parent_id'
            FROM {}
            WHERE metadata->>'file_hash' = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(sql.Identifier(self._table_name))
        with (
            psycopg.connect(self._postgres_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                statement,
                (list(allowed_hashes), _vector_literal(query_embedding), top_k),
            )
            parent_ids = [str(row[0]) for row in cursor.fetchall()]

        values = self._redis.mget(
            [f"{self.PARENT_PREFIX}{parent_id}" for parent_id in parent_ids]
        )
        retrieved: list[RetrievedElement] = []
        for rank, value in enumerate(values, start=1):
            if value:
                retrieved.append(
                    RetrievedElement(element=_element_from_dict(json.loads(value)), rank=rank)
                )
        return retrieved

    def _ensure_schema(self) -> None:
        import psycopg
        from psycopg import sql

        dimensions = sql.SQL(str(self._embedding_dimensions))
        table = sql.Identifier(self._table_name)
        vector_index = sql.Identifier(f"{self._table_name}_embedding_hnsw")
        hash_index = sql.Identifier(f"{self._table_name}_file_hash")
        create_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                embedding vector({}) NOT NULL,
                metadata JSONB NOT NULL
            )
            """
        ).format(table, dimensions)
        create_vector_index = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
        ).format(vector_index, table)
        create_hash_index = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} ((metadata->>'file_hash'))"
        ).format(hash_index, table)

        with (
            psycopg.connect(self._postgres_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(create_table)
            cursor.execute(create_vector_index)
            cursor.execute(create_hash_index)

    def _postgres_ping(self) -> None:
        import psycopg

        with (
            psycopg.connect(self._postgres_url, connect_timeout=5) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("SELECT 1")


def _element_to_dict(element: ExtractedElement) -> dict[str, object]:
    return {
        "parent_id": element.parent_id,
        "text": element.text,
        "source": element.source,
        "file_hash": element.file_hash,
        "page": element.page,
        "kind": element.kind,
    }


def _element_from_dict(data: dict[str, object]) -> ExtractedElement:
    page = data.get("page")
    return ExtractedElement(
        parent_id=str(data["parent_id"]),
        text=str(data["text"]),
        source=str(data["source"]),
        file_hash=str(data["file_hash"]),
        page=int(page) if page is not None else None,
        kind=str(data["kind"]),
    )


def _normalize_postgres_url(value: str) -> str:
    """Convert a SQLAlchemy psycopg URL into a native psycopg connection URL."""
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _vector_literal(values: Sequence[float]) -> str:
    """Serialize numeric embeddings using pgvector's text input format."""
    return "[" + ",".join(format(float(value), ".12g") for value in values) + "]"
