from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import networkx as nx

    from token_sniper.retrieval.embeddings import Embedder
    from token_sniper.retrieval.vectorstore import EntityVectorStore

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class EntryPointResult(BaseModel):
    method: str = Field(description="'llm_prompt' or 'embedding'")
    issue_text_preview: str = Field(default="", description="First 200 chars of issue text")
    candidate_count: int = 0
    selected_fqns: list[str] = Field(default_factory=list)
    matched_node_ids: list[str] = Field(default_factory=list)
    llm_raw_response: str | None = None
    error: str | None = None

    model_config = {"frozen": True}


def identify_entry_points_by_embedding(
    *,
    issue_text: str,
    repo: str,
    embedder: Embedder,
    vector_store: EntityVectorStore,
    graph: nx.MultiDiGraph[Any],
    n_candidates: int = 3,
) -> EntryPointResult:
    from token_sniper.graph.schema import is_file_node_id

    try:
        query_embedding = embedder.embed_query(issue_text)
        results = vector_store.query(
            repo=repo,
            query_embedding=query_embedding,
            k=n_candidates * 3,
            include_documents=False,
        )
    except Exception as exc:
        return EntryPointResult(
            method="embedding",
            issue_text_preview=issue_text[:200],
            error=f"Embedding query failed: {exc}",
        )

    graph_nodes = set(graph.nodes())
    selected: list[str] = []
    for r in results.results:
        fqn = r.metadata.fully_qualified_name
        if fqn in graph_nodes and not is_file_node_id(fqn):
            selected.append(fqn)
            if len(selected) >= n_candidates:
                break

    return EntryPointResult(
        method="embedding",
        issue_text_preview=issue_text[:200],
        candidate_count=len(results.results),
        selected_fqns=selected,
        matched_node_ids=selected,
    )


_ENTRY_POINT_SYSTEM_PROMPT = """\
You are a code navigation expert. Given an issue description and a list of code \
entities in a repository, identify 1-3 specific entities (functions, classes, or \
methods) that are most likely to be the primary entry points for understanding and \
fixing this issue.

Rules:
- Pick entities that are DIRECTLY related to the bug or feature described.
- Prefer functions/methods over classes when the issue is about specific behavior.
- Return ONLY fully qualified names from the provided entity list, copied EXACTLY.
- Do NOT invent names — only choose from the entities shown.
- Use dotted Python notation (e.g. ``django.db.models.query.QuerySet.filter``), \
NOT file paths with slashes.

Respond with a JSON object: {"entry_points": ["fqn1", "fqn2"]}\
"""


def identify_entry_points_by_llm(
    *,
    issue_text: str,
    graph: nx.MultiDiGraph[Any],
    llm_client: Any,
    n_candidates: int = 3,
    max_listing_tokens: int = 8000,
    embedder: Any | None = None,
    vector_store: Any | None = None,
    repo: str = "",
) -> EntryPointResult:
    from token_sniper.graph.schema import is_file_node_id

    entity_listing = _build_entity_listing(
        graph,
        max_tokens=max_listing_tokens,
        issue_text=issue_text,
        embedder=embedder,
        vector_store=vector_store,
        repo=repo,
    )
    user_prompt = (
        f"Issue description:\n{issue_text}\n\n"
        f"Repository entities (grouped by file, most relevant first):\n{entity_listing}\n\n"
        f"Identify 1-{n_candidates} entry-point entities from the list above. "
        f'Respond with JSON: {{"entry_points": ["fqn1", ...]}}'
    )

    try:
        response = llm_client.complete(
            messages=[
                {"role": "system", "content": _ENTRY_POINT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[],
            response_format={"type": "json_object"},
        )
        raw_text = response.content
    except Exception as exc:
        return EntryPointResult(
            method="llm_prompt",
            issue_text_preview=issue_text[:200],
            error=f"LLM call failed: {exc}",
        )

    parsed_fqns = _parse_fqn_response(raw_text)
    graph_nodes = set(graph.nodes())
    node_set = {n for n in graph_nodes if not is_file_node_id(n)}
    matched = _match_fqns_to_graph(parsed_fqns, node_set)

    logger.info(
        "LLM entry points: parsed=%d, matched=%d, raw=%s",
        len(parsed_fqns),
        len(matched),
        raw_text[:300],
    )

    return EntryPointResult(
        method="llm_prompt",
        issue_text_preview=issue_text[:200],
        candidate_count=len(parsed_fqns),
        selected_fqns=parsed_fqns[:n_candidates],
        matched_node_ids=matched[:n_candidates],
        llm_raw_response=raw_text,
    )


def _build_entity_listing(
    graph: nx.MultiDiGraph[Any],
    *,
    max_tokens: int,
    issue_text: str = "",
    embedder: Any | None = None,
    vector_store: Any | None = None,
    repo: str = "",
) -> str:
    from token_sniper.graph.schema import is_file_node_id

    files: dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        if is_file_node_id(node_id):
            continue
        file_path = data.get("file_path", "")
        node_type = data.get("node_type", "")
        if node_type == "module_preamble":
            continue
        files.setdefault(file_path, []).append(node_id)

    file_order = _rank_files_by_relevance(
        files=files,
        issue_text=issue_text,
        embedder=embedder,
        vector_store=vector_store,
        repo=repo,
    )

    lines: list[str] = []
    total_chars = 0
    char_budget = max_tokens * 4
    files_shown = 0

    for file_path in file_order:
        entities = sorted(files[file_path])
        file_line = f"\n{file_path}:"
        entity_lines = [f"  - {fqn}" for fqn in entities]
        block = file_line + "\n" + "\n".join(entity_lines)
        if total_chars + len(block) > char_budget:
            remaining = len(files) - files_shown
            if remaining > 0:
                lines.append(f"\n... ({remaining} more files not shown)")
            break
        lines.append(block)
        total_chars += len(block)
        files_shown += 1

    return "\n".join(lines)


def _rank_files_by_relevance(
    *,
    files: dict[str, list[str]],
    issue_text: str,
    embedder: Any | None,
    vector_store: Any | None,
    repo: str,
) -> list[str]:
    if not issue_text or embedder is None or vector_store is None or not repo:
        return sorted(files.keys())

    try:
        query_emb = embedder.embed_query(issue_text)
        results = vector_store.query(
            repo=repo,
            query_embedding=query_emb,
            k=min(200, max(50, len(files))),
            include_documents=False,
        )
    except Exception:
        logger.debug("Embedding ranking failed, falling back to alphabetical", exc_info=True)
        return sorted(files.keys())

    file_scores: dict[str, float] = {}
    for r in results.results:
        fp = r.metadata.file_path
        if fp in files and fp not in file_scores:
            file_scores[fp] = r.score

    ranked = sorted(file_scores.keys(), key=lambda fp: file_scores[fp], reverse=True)
    remaining = sorted(set(files.keys()) - set(ranked))
    return ranked + remaining


def _parse_fqn_response(text: str) -> list[str]:
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            arr = obj.get("entry_points", obj.get("fqns", []))
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, str)]
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                for key in ("entry_points", "fqns", "results"):
                    arr = obj.get(key)
                    if isinstance(arr, list):
                        return [str(x) for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, str)]
        except json.JSONDecodeError:
            pass

    return re.findall(r'"([^"]+\.[^"]+)"', cleaned)


def _normalize_fqn(raw: str) -> str:
    normalized = raw.replace("/", ".").replace("\\", ".")
    normalized = re.sub(r"\.py\b", "", normalized)
    normalized = normalized.lstrip(".")
    return normalized


def _match_fqns_to_graph(
    parsed_fqns: list[str],
    node_set: set[str],
) -> list[str]:
    by_suffix: dict[str, list[str]] = {}
    for nid in node_set:
        parts = nid.split(".")
        for start in range(1, len(parts)):
            suffix = ".".join(parts[start:])
            by_suffix.setdefault(suffix, []).append(nid)

    matched: list[str] = []
    seen: set[str] = set()

    for raw_fqn in parsed_fqns:
        if raw_fqn in node_set and raw_fqn not in seen:
            matched.append(raw_fqn)
            seen.add(raw_fqn)
            continue

        normalized = _normalize_fqn(raw_fqn)
        if normalized in node_set and normalized not in seen:
            matched.append(normalized)
            seen.add(normalized)
            continue

        parts = normalized.split(".")
        found = False
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            candidates = by_suffix.get(suffix, [])
            if len(candidates) == 1 and candidates[0] not in seen:
                matched.append(candidates[0])
                seen.add(candidates[0])
                found = True
                break
        if found:
            continue

        logger.debug("LLM FQN not matched: %r (normalized: %r)", raw_fqn, normalized)

    return matched


def identify_entry_points(
    *,
    method: str,
    issue_text: str,
    repo: str,
    graph: nx.MultiDiGraph[Any],
    n_candidates: int = 3,
    embedder: Embedder | None = None,
    vector_store: EntityVectorStore | None = None,
    llm_client: Any | None = None,
) -> EntryPointResult:
    if method == "embedding":
        if embedder is None or vector_store is None:
            raise ValueError("embedder and vector_store required for embedding entry points")
        return identify_entry_points_by_embedding(
            issue_text=issue_text,
            repo=repo,
            embedder=embedder,
            vector_store=vector_store,
            graph=graph,
            n_candidates=n_candidates,
        )

    if method == "llm_prompt":
        if llm_client is None:
            raise ValueError("llm_client required for LLM-based entry points")
        result = identify_entry_points_by_llm(
            issue_text=issue_text,
            graph=graph,
            llm_client=llm_client,
            n_candidates=n_candidates,
            embedder=embedder,
            vector_store=vector_store,
            repo=repo,
        )
        if not result.matched_node_ids and embedder is not None and vector_store is not None:
            logger.warning("LLM entry points returned no matches; falling back to embedding")
            fallback = identify_entry_points_by_embedding(
                issue_text=issue_text,
                repo=repo,
                embedder=embedder,
                vector_store=vector_store,
                graph=graph,
                n_candidates=n_candidates,
            )
            return fallback.model_copy(
                update={
                    "method": "llm_prompt_with_embedding_fallback",
                    "llm_raw_response": result.llm_raw_response,
                }
            )
        return result

    raise ValueError(f"Unknown entry point method: {method!r}")
