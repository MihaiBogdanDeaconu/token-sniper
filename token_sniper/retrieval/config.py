from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

EmbeddingProvider = Literal["voyageai", "nomic-local", "sentence-transformers"]


class EmbeddingConfig(BaseModel):
    name: str = "jinaai/jina-code-embeddings-0.5b"
    provider: EmbeddingProvider = "sentence-transformers"
    dimension: int = Field(default=256, ge=0, description="Matryoshka truncation dim (256 retains 99% quality)")
    api_key_env: str = "VOYAGE_API_KEY"
    batch_size: int = Field(default=64, gt=0, description="Texts per GPU batch (0.5B model fits easily on H100)")
    max_tokens_per_batch: int = Field(
        default=18_000,
        gt=0,
        description="Soft token-budget per batch — keeps requests small to avoid timeouts",
    )
    max_seq_length: int = Field(default=8192, gt=0, description="Max input tokens (matches paper eval config)")
    device: str = Field(default="cuda", description="Device for local models (cuda, cpu, auto)")
    trust_remote_code: bool = Field(default=True, description="For HF models requiring custom code")

    model_config = {"frozen": True}

    def resolve_api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class VectorStoreConfig(BaseModel):
    backend: Literal["chromadb"] = "chromadb"
    persist_directory: str = "data/chromadb/"
    collection_prefix: str = "token_sniper"

    model_config = {"frozen": True}


class RetrievalSubsystemConfig(BaseModel):
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    model_config = {"frozen": True}
