from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SSG_SCHEMA_VERSION = "1"

SSG_META_VERSION_KEY = "ssg_schema_version"
SSG_META_UNRESOLVED_KEY = "unresolved_edges"

FILE_NODE_PREFIX = "file:"

NodeType = Literal["file", "class", "function", "method", "module_preamble", "file_blob"]

EdgeType = Literal["contains", "inherits", "calls", "imports"]

CallConfidence = Literal["high", "low"]


def file_node_id(posix_path: str) -> str:
    return f"{FILE_NODE_PREFIX}{posix_path}"


def is_file_node_id(node_id: str) -> bool:
    return node_id.startswith(FILE_NODE_PREFIX)


class NodeAttrs(BaseModel):
    node_type: NodeType
    file_path: str = Field(description="POSIX path of the defining file")
    start_line: int = Field(ge=1, description="1-based inclusive start line")
    end_line: int = Field(ge=1, description="1-based inclusive end line")
    module_qualname: str = Field(
        default="",
        description="Dotted module path; empty for file-only nodes if unused",
    )
    fully_qualified_name: str = Field(
        default="",
        description="Same as NetworkX node id for entities; empty for file nodes",
    )

    model_config = {"frozen": True}


class EdgeAttrs(BaseModel):
    edge_type: EdgeType
    confidence: CallConfidence | None = Field(
        default=None,
        description="Set for calls edges when callee resolution is scored",
    )
    target_name: str | None = Field(
        default=None,
        description="Unresolved import/inherits target symbol, if any",
    )
    resolved: bool = Field(
        default=True,
        description="False when target_name is set and no graph target node exists",
    )

    model_config = {"frozen": True}
