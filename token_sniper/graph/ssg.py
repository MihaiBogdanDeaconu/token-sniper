from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from token_sniper.graph.python_static_edges import (
    CallSiteTarget,
    ImportRecord,
    extract_call_targets,
    extract_class_bases,
    extract_imports,
)
from token_sniper.graph.schema import (
    SSG_META_UNRESOLVED_KEY,
    SSG_META_VERSION_KEY,
    SSG_SCHEMA_VERSION,
    CallConfidence,
    EdgeAttrs,
    NodeAttrs,
    NodeType,
    file_node_id,
)
from token_sniper.parse import CodeEntity, extract_python_entities


_MAX_FILES_DEFAULT = 30_000

_NON_PY_EXTENSIONS = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".md", ".rst", ".txt",
    ".rs", ".go", ".js", ".ts", ".jsx", ".tsx",
    ".sh", ".bash",
    ".textfsm",
    ".xml", ".html",
})

_MAX_BLOB_BYTES = 32_000


def _read_file_blob(abs_path: Path, rel_posix: str) -> CodeEntity | None:
    try:
        raw = abs_path.read_bytes()
        if len(raw) > _MAX_BLOB_BYTES:
            raw = raw[:_MAX_BLOB_BYTES]
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return None

    if not text.strip():
        return None

    n_lines = text.count("\n") + 1
    est_tokens = max(1, len(raw) // 4)

    return CodeEntity(
        entity_type="file_blob",
        fully_qualified_name=f"blob:{rel_posix}",
        file_path=rel_posix,
        module_qualname="",
        start_line=1,
        end_line=n_lines,
        source_text=text,
        referenced_identifiers=[],
        signature_summary=None,
        estimated_tokens=est_tokens,
    )


def build_ssg(
    repo_root: Path | str,
    *,
    include_glob: str = "**/*.py",
    extra_globs: list[str] | None = None,
    max_files: int = _MAX_FILES_DEFAULT,
) -> nx.MultiDiGraph:
    root = Path(repo_root).resolve()
    py_files = sorted(p for p in root.glob(include_glob) if p.is_file())
    if len(py_files) > max_files:
        raise ValueError(
            f"Repo has {len(py_files)} Python files (limit {max_files}). "
            f"Skipping to avoid OOM."
        )

    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    g.graph[SSG_META_VERSION_KEY] = SSG_SCHEMA_VERSION
    g.graph[SSG_META_UNRESOLVED_KEY] = []

    file_entities: list[tuple[str, str, list[CodeEntity]]] = []
    for abs_path in py_files:
        rel = abs_path.relative_to(root)
        rel_posix = rel.as_posix()
        try:
            raw_ents = extract_python_entities(abs_path, project_root=root)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        ents = [e.model_copy(update={"file_path": rel_posix}) for e in raw_ents]
        mod_q = ents[0].module_qualname if ents else ""
        file_entities.append((rel_posix, mod_q, ents))

    blob_entities: list[tuple[str, CodeEntity]] = []
    if extra_globs is None:
        extra_globs = [f"**/*{ext}" for ext in sorted(_NON_PY_EXTENSIONS)]
    seen_paths: set[str] = {rp for rp, _, _ in file_entities}
    for glob_pat in extra_globs:
        for abs_path in sorted(root.glob(glob_pat)):
            if not abs_path.is_file():
                continue
            rel_posix = abs_path.relative_to(root).as_posix()
            if rel_posix in seen_paths:
                continue
            seen_paths.add(rel_posix)
            blob = _read_file_blob(abs_path, rel_posix)
            if blob is not None:
                blob_entities.append((rel_posix, blob))

    entity_by_fqn: dict[str, CodeEntity] = {}
    for _rel_posix, _mq, ents in file_entities:
        for e in ents:
            entity_by_fqn[e.fully_qualified_name] = e
    for _rp, blob in blob_entities:
        entity_by_fqn[blob.fully_qualified_name] = blob

    node_ids: set[str] = set(entity_by_fqn)
    for rel_posix, _, _ in file_entities:
        node_ids.add(file_node_id(rel_posix))
    for rel_posix, _ in blob_entities:
        node_ids.add(file_node_id(rel_posix))

    last_seg: defaultdict[str, list[str]] = defaultdict(list)
    for fqn, ent in entity_by_fqn.items():
        if ent.entity_type == "module_preamble":
            continue
        last_seg[fqn.split(".")[-1]].append(fqn)

    module_to_file: dict[str, str] = {}
    for rel_posix, mq, _ents in file_entities:
        if mq and mq not in module_to_file:
            module_to_file[mq] = rel_posix

    for rel_posix, _mq, ents in file_entities:
        fid = file_node_id(rel_posix)
        g.add_node(
            fid,
            **NodeAttrs(
                node_type="file",
                file_path=rel_posix,
                start_line=1,
                end_line=1,
                module_qualname="",
                fully_qualified_name="",
            ).model_dump(),
        )
        for e in ents:
            g.add_node(
                e.fully_qualified_name,
                **NodeAttrs(
                    node_type=_entity_node_type(e),
                    file_path=e.file_path,
                    start_line=e.start_line,
                    end_line=e.end_line,
                    module_qualname=e.module_qualname,
                    fully_qualified_name=e.fully_qualified_name,
                ).model_dump(),
            )

    for rel_posix, blob in blob_entities:
        fid = file_node_id(rel_posix)
        g.add_node(
            fid,
            **NodeAttrs(
                node_type="file",
                file_path=rel_posix,
                start_line=1,
                end_line=1,
                module_qualname="",
                fully_qualified_name="",
            ).model_dump(),
        )
        g.add_node(
            blob.fully_qualified_name,
            **NodeAttrs(
                node_type="file_blob",
                file_path=rel_posix,
                start_line=1,
                end_line=blob.end_line,
                module_qualname="",
                fully_qualified_name=blob.fully_qualified_name,
            ).model_dump(),
        )
        _add_edge(
            g,
            fid,
            blob.fully_qualified_name,
            EdgeAttrs(edge_type="contains", resolved=True),
        )

    for _rp, _mq, ents in file_entities:
        fid = file_node_id(_rp)
        for e in ents:
            _add_edge(
                g,
                fid,
                e.fully_qualified_name,
                EdgeAttrs(edge_type="contains", resolved=True),
            )
            parent = _longest_container_fqn(e.fully_qualified_name, node_ids)
            if parent is not None:
                _add_edge(
                    g,
                    parent,
                    e.fully_qualified_name,
                    EdgeAttrs(edge_type="contains", resolved=True),
                )

    for rel_posix, _mq, ents in file_entities:
        if not ents:
            continue
        abs_path = root / rel_posix
        try:
            source_bytes = abs_path.read_bytes()
        except OSError:
            continue
        importer = file_node_id(rel_posix)
        for rec in extract_imports(source_bytes):
            _add_import_edges(
                g,
                importer=importer,
                rec=rec,
                module_to_file=module_to_file,
                entity_by_fqn=entity_by_fqn,
                last_seg=last_seg,
            )

    for rel_posix, mq, _ents in file_entities:
        abs_path = root / rel_posix
        try:
            source_bytes = abs_path.read_bytes()
        except OSError:
            continue
        for cb in extract_class_bases(source_bytes, mq):
            if cb.class_fqn not in entity_by_fqn:
                continue
            sub_ent = entity_by_fqn[cb.class_fqn]
            if sub_ent.entity_type != "class":
                continue
            for base in cb.base_names:
                tgt = _resolve_class_base(
                    base_name=base,
                    subclass_module=sub_ent.module_qualname,
                    entity_by_fqn=entity_by_fqn,
                    last_seg=last_seg,
                )
                if tgt is None:
                    g.graph[SSG_META_UNRESOLVED_KEY].append(
                        {
                            "edge_type": "inherits",
                            "source": cb.class_fqn,
                            "target": None,
                            "target_name": base,
                            "resolved": False,
                        }
                    )
                    continue
                _add_edge(
                    g,
                    cb.class_fqn,
                    tgt,
                    EdgeAttrs(edge_type="inherits", resolved=True),
                )

    entity_fqns = [ef for ef, e in entity_by_fqn.items() if e.entity_type != "module_preamble"]
    for fqn, ent in entity_by_fqn.items():
        if ent.entity_type not in ("function", "method"):
            continue
        for site in extract_call_targets(ent.source_text.encode("utf-8")):
            cands = _call_resolution_candidates(
                site,
                caller_fqn=fqn,
                entity_fqns=entity_fqns,
                last_seg=last_seg,
            )
            if not cands:
                continue
            conf: CallConfidence = "high" if len(cands) == 1 else "low"
            for tgt in cands:
                _add_edge(
                    g,
                    fqn,
                    tgt,
                    EdgeAttrs(edge_type="calls", confidence=conf, resolved=True),
                )

    return g


def _call_resolution_candidates(
    site: CallSiteTarget,
    *,
    caller_fqn: str,
    entity_fqns: list[str],
    last_seg: defaultdict[str, list[str]],
) -> list[str]:
    qh = site.qualified_hint
    if qh:
        suffix = "." + qh
        cands = [f for f in entity_fqns if f == qh or f.endswith(suffix)]
        cands = [c for c in cands if c != caller_fqn]
        if cands:
            return cands
    return [c for c in last_seg.get(site.leaf, []) if c != caller_fqn]


def _entity_node_type(e: CodeEntity) -> NodeType:
    if e.entity_type == "module_preamble":
        return "module_preamble"
    if e.entity_type == "class":
        return "class"
    if e.entity_type == "method":
        return "method"
    return "function"


def _longest_container_fqn(fqn: str, node_ids: set[str]) -> str | None:
    parts = fqn.split(".")
    for i in range(len(parts) - 1, 0, -1):
        cand = ".".join(parts[:i])
        if cand in node_ids and cand != fqn:
            return cand
    return None


def _add_edge(g: nx.MultiDiGraph[Any], u: str, v: str, attrs: EdgeAttrs) -> None:
    g.add_edge(u, v, **attrs.model_dump())


def _add_import_edges(
    g: nx.MultiDiGraph[Any],
    *,
    importer: str,
    rec: ImportRecord,
    module_to_file: dict[str, str],
    entity_by_fqn: dict[str, CodeEntity],
    last_seg: defaultdict[str, list[str]],
) -> None:
    if rec.names is None:
        tgt_file = module_to_file.get(rec.module)
        if tgt_file is None:
            g.graph[SSG_META_UNRESOLVED_KEY].append(
                {
                    "edge_type": "imports",
                    "source": importer,
                    "target": None,
                    "target_name": rec.module,
                    "resolved": False,
                    "module": rec.module,
                }
            )
            return
        _add_edge(
            g,
            importer,
            file_node_id(tgt_file),
            EdgeAttrs(edge_type="imports", resolved=True),
        )
        return

    for sym in rec.names:
        fqn = f"{rec.module}.{sym}"
        if fqn in entity_by_fqn:
            _add_edge(
                g,
                importer,
                fqn,
                EdgeAttrs(edge_type="imports", resolved=True),
            )
            continue
        sub_mod = f"{rec.module}.{sym}"
        tgt_file = module_to_file.get(sub_mod)
        if tgt_file is not None:
            _add_edge(
                g,
                importer,
                file_node_id(tgt_file),
                EdgeAttrs(edge_type="imports", resolved=True),
            )
            continue
        cands = [c for c in last_seg.get(sym, []) if c.startswith(rec.module + ".")]
        if len(cands) == 1:
            _add_edge(
                g,
                importer,
                cands[0],
                EdgeAttrs(edge_type="imports", resolved=True),
            )
        elif not cands:
            g.graph[SSG_META_UNRESOLVED_KEY].append(
                {
                    "edge_type": "imports",
                    "source": importer,
                    "target": None,
                    "target_name": f"{rec.module}.{sym}",
                    "resolved": False,
                    "module": rec.module,
                    "symbol": sym,
                }
            )
        else:
            for c in cands:
                _add_edge(
                    g,
                    importer,
                    c,
                    EdgeAttrs(edge_type="imports", confidence="low", resolved=True),
                )


def _resolve_class_base(
    *,
    base_name: str,
    subclass_module: str,
    entity_by_fqn: dict[str, CodeEntity],
    last_seg: defaultdict[str, list[str]],
) -> str | None:
    cands = [c for c in last_seg.get(base_name, []) if entity_by_fqn[c].entity_type == "class"]
    if not cands:
        return None
    same = [c for c in cands if entity_by_fqn[c].module_qualname == subclass_module]
    if len(same) == 1:
        return same[0]
    if len(cands) == 1:
        return cands[0]
    return None
