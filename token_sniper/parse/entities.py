from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["function", "method", "class", "module_preamble", "file_blob"]


class CodeEntity(BaseModel):
    entity_type: EntityType
    fully_qualified_name: str
    file_path: str = Field(description="Path as provided (prefer POSIX for portability)")
    module_qualname: str = Field(description="Dotted module path used to build FQN")
    start_line: int = Field(ge=1, description="1-based inclusive start line")
    end_line: int = Field(ge=1, description="1-based inclusive end line")
    source_text: str
    referenced_identifiers: list[str] = Field(
        default_factory=list,
        description="Names referenced in the entity body (see parser rules per entity type)",
    )
    signature_summary: str | None = Field(
        default=None,
        description="Present when estimated_tokens > threshold: compact signature + docstring head",
    )
    estimated_tokens: int = Field(
        ge=0,
        description="Rough token estimate (~len(UTF-8 bytes)/4) for gating signature_summary",
    )

    model_config = {"frozen": True}
