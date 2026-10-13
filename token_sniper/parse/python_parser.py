from __future__ import annotations

from pathlib import Path
from typing import Literal

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

from token_sniper.parse.entities import CodeEntity, EntityType

_ROLE = Literal["function", "method"]


def infer_module_qualname(file_path: Path, project_root: Path | None = None) -> str:
    path = file_path.resolve()
    if project_root is not None:
        try:
            rel = path.relative_to(project_root.resolve())
        except ValueError:
            rel = Path(path.name)
    else:
        rel = Path(path.name)

    parts = list(rel.parts)
    if not parts:
        return path.stem
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else path.stem


def extract_python_entities(
    path: Path | str,
    source: bytes | str | None = None,
    *,
    project_root: Path | None = None,
    token_threshold: int = 500,
) -> list[CodeEntity]:
    file_path = Path(path)
    if source is None:
        source_bytes = file_path.read_bytes()
    elif isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source

    module_qualname = infer_module_qualname(file_path, project_root)
    file_posix = file_path.as_posix()

    language = Language(tsp.language())
    parser = Parser(language)
    tree = parser.parse(source_bytes)
    root = tree.root_node
    if root.type != "module":
        msg = f"Expected root 'module', got {root.type!r}"
        raise ValueError(msg)

    entities: list[CodeEntity] = []
    preamble_nodes, roots = _partition_module_children(root)
    if preamble_nodes:
        preamble_entity = _build_module_preamble(
            preamble_nodes, source_bytes, file_posix, module_qualname
        )
        if preamble_entity.estimated_tokens > 0 or preamble_entity.source_text.strip():
            entities.append(preamble_entity)

    for child in roots:
        _process_module_entity_root(
            child,
            module_qualname=module_qualname,
            file_path=file_posix,
            source_bytes=source_bytes,
            entities=entities,
            token_threshold=token_threshold,
        )

    return entities


def _partition_module_children(module: Node) -> tuple[list[Node], list[Node]]:
    preamble: list[Node] = []
    roots: list[Node] = []
    for ch in module.children:
        if ch.type in ("function_definition", "class_definition", "decorated_definition"):
            roots.append(ch)
        else:
            preamble.append(ch)
    return preamble, roots


def _build_module_preamble(
    nodes: list[Node],
    source_bytes: bytes,
    file_path: str,
    module_qualname: str,
) -> CodeEntity:
    texts = [_node_text(n, source_bytes) for n in nodes]
    src = "\n\n".join(t for t in texts if t.strip() or t == "")
    start_line = nodes[0].start_point[0] + 1 if nodes else 1
    end_line = nodes[-1].end_point[0] + 1 if nodes else 1
    est = _estimate_tokens(src)
    fqn = f"{module_qualname}.__module_preamble__" if module_qualname else "__module_preamble__"
    return CodeEntity(
        entity_type="module_preamble",
        fully_qualified_name=fqn,
        file_path=file_path,
        module_qualname=module_qualname,
        start_line=start_line,
        end_line=end_line,
        source_text=src,
        referenced_identifiers=_collect_identifiers_from_nodes(nodes),
        signature_summary=None,
        estimated_tokens=est,
    )


def _collect_identifiers_from_nodes(nodes: list[Node]) -> list[str]:
    found: set[str] = set()
    for n in nodes:
        found.update(_collect_identifiers(n))
    return sorted(found)


def _process_module_entity_root(
    node: Node,
    module_qualname: str,
    file_path: str,
    source_bytes: bytes,
    entities: list[CodeEntity],
    token_threshold: int,
) -> None:
    if node.type == "decorated_definition":
        inner = _decorated_inner(node)
        if inner.type == "function_definition":
            _emit_function(
                inner,
                node,
                [],
                "function",
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )
        else:
            _emit_class(
                inner, node, [], module_qualname, file_path, source_bytes, entities, token_threshold
            )
    elif node.type == "function_definition":
        _emit_function(
            node,
            None,
            [],
            "function",
            module_qualname,
            file_path,
            source_bytes,
            entities,
            token_threshold,
        )
    elif node.type == "class_definition":
        _emit_class(
            node, None, [], module_qualname, file_path, source_bytes, entities, token_threshold
        )


def _decorated_inner(node: Node) -> Node:
    for c in reversed(node.children):
        if c.type in ("function_definition", "class_definition"):
            return c
    msg = "decorated_definition without function or class"
    raise ValueError(msg)


def _emit_class(
    class_node: Node,
    decorated_outer: Node | None,
    scope: list[str],
    module_qualname: str,
    file_path: str,
    source_bytes: bytes,
    entities: list[CodeEntity],
    token_threshold: int,
) -> None:
    full_node = decorated_outer or class_node
    name_n = class_node.child_by_field_name("name")
    class_name = _node_text(name_n, source_bytes) if name_n else "<class>"
    new_scope = scope + [class_name]

    supers = class_node.child_by_field_name("superclasses")
    super_ids: set[str] = _collect_identifiers(supers) if supers else set()
    body = class_node.child_by_field_name("body")
    class_level = _class_level_identifiers(body) if body else []
    referenced = sorted(super_ids | set(class_level))

    src = _node_text(full_node, source_bytes)
    start_line, end_line = _lines(full_node)
    est = _estimate_tokens(src)
    sig = _class_signature(class_node, source_bytes)
    summary = sig if est > token_threshold else None

    entities.append(
        CodeEntity(
            entity_type="class",
            fully_qualified_name=_fqn(module_qualname, scope, class_name),
            file_path=file_path,
            module_qualname=module_qualname,
            start_line=start_line,
            end_line=end_line,
            source_text=src,
            referenced_identifiers=referenced,
            signature_summary=summary,
            estimated_tokens=est,
        )
    )

    if body:
        for stmt in body.children:
            _dispatch_class_body(
                stmt,
                new_scope,
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )


def _dispatch_class_body(
    stmt: Node,
    scope: list[str],
    module_qualname: str,
    file_path: str,
    source_bytes: bytes,
    entities: list[CodeEntity],
    token_threshold: int,
) -> None:
    if stmt.type == "decorated_definition":
        inner = _decorated_inner(stmt)
        if inner.type == "function_definition":
            _emit_function(
                inner,
                stmt,
                scope,
                "method",
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )
        else:
            _emit_class(
                inner,
                stmt,
                scope,
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )
    elif stmt.type == "function_definition":
        _emit_function(
            stmt,
            None,
            scope,
            "method",
            module_qualname,
            file_path,
            source_bytes,
            entities,
            token_threshold,
        )
    elif stmt.type == "class_definition":
        _emit_class(
            stmt, None, scope, module_qualname, file_path, source_bytes, entities, token_threshold
        )


def _emit_function(
    fn_node: Node,
    decorated_outer: Node | None,
    scope: list[str],
    role: _ROLE,
    module_qualname: str,
    file_path: str,
    source_bytes: bytes,
    entities: list[CodeEntity],
    token_threshold: int,
) -> None:
    full_node = decorated_outer or fn_node
    name_n = fn_node.child_by_field_name("name")
    fn_name = _node_text(name_n, source_bytes) if name_n else "<lambda>"
    body = fn_node.child_by_field_name("body")

    ref_ids = sorted(_collect_identifiers(body)) if body else []
    src = _node_text(full_node, source_bytes)
    start_line, end_line = _lines(full_node)
    est = _estimate_tokens(src)
    sig = _function_signature(fn_node, source_bytes)
    summary = sig if est > token_threshold else None
    etype: EntityType = role

    entities.append(
        CodeEntity(
            entity_type=etype,
            fully_qualified_name=_fqn(module_qualname, scope, fn_name),
            file_path=file_path,
            module_qualname=module_qualname,
            start_line=start_line,
            end_line=end_line,
            source_text=src,
            referenced_identifiers=ref_ids,
            signature_summary=summary,
            estimated_tokens=est,
        )
    )

    if body:
        inner_scope = scope + [fn_name]
        for stmt in body.children:
            _dispatch_function_body(
                stmt,
                inner_scope,
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )


def _dispatch_function_body(
    stmt: Node,
    scope: list[str],
    module_qualname: str,
    file_path: str,
    source_bytes: bytes,
    entities: list[CodeEntity],
    token_threshold: int,
) -> None:
    if stmt.type == "decorated_definition":
        inner = _decorated_inner(stmt)
        if inner.type == "function_definition":
            _emit_function(
                inner,
                stmt,
                scope,
                "function",
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )
        else:
            _emit_class(
                inner,
                stmt,
                scope,
                module_qualname,
                file_path,
                source_bytes,
                entities,
                token_threshold,
            )
    elif stmt.type == "function_definition":
        _emit_function(
            stmt,
            None,
            scope,
            "function",
            module_qualname,
            file_path,
            source_bytes,
            entities,
            token_threshold,
        )
    elif stmt.type == "class_definition":
        _emit_class(
            stmt, None, scope, module_qualname, file_path, source_bytes, entities, token_threshold
        )


def _fqn(module_qualname: str, scope: list[str], name: str) -> str:
    parts = [p for p in [module_qualname, *scope, name] if p]
    return ".".join(parts)


def _lines(node: Node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // 4)


def _collect_identifiers(node: Node | None) -> set[str]:
    if node is None:
        return set()
    found: set[str] = set()
    stack: list[Node] = [node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            raw = n.text or b""
            found.add(raw.decode("utf-8", errors="replace"))
        stack.extend(n.children)
    return found


def _class_level_identifiers(body: Node) -> list[str]:
    found: set[str] = set()
    for stmt in body.children:
        if stmt.type in ("function_definition", "class_definition"):
            continue
        if stmt.type == "decorated_definition":
            continue
        found.update(_collect_identifiers(stmt))
    return sorted(found)


def _class_signature(class_node: Node, source_bytes: bytes) -> str:
    name_n = class_node.child_by_field_name("name")
    name = _node_text(name_n, source_bytes) if name_n else "?"
    supers = class_node.child_by_field_name("superclasses")
    sup = _node_text(supers, source_bytes) if supers else ""
    head = f"class {name}{sup}:"
    body = class_node.child_by_field_name("body")
    if body:
        for ch in body.children:
            if ch.type == "expression_statement":
                txt = _node_text(ch, source_bytes).strip()
                if txt.startswith(('"""', "'''")):
                    return f"{head}\n{txt[:200]}"
    return head


def _function_signature(fn_node: Node, source_bytes: bytes) -> str:
    is_async = any(c.type == "async" for c in fn_node.children)
    prefix = "async def" if is_async else "def"
    name_n = fn_node.child_by_field_name("name")
    name = _node_text(name_n, source_bytes) if name_n else "?"
    params = fn_node.child_by_field_name("parameters")
    ps = _node_text(params, source_bytes) if params else "()"
    ret = fn_node.child_by_field_name("return_type")
    ann = ""
    if ret is not None:
        ann = f" -> {_node_text(ret, source_bytes)}"
    head = f"{prefix} {name}{ps}{ann}:"
    body = fn_node.child_by_field_name("body")
    if body:
        for ch in body.children:
            if ch.type == "expression_statement":
                txt = _node_text(ch, source_bytes).strip()
                if txt.startswith(('"""', "'''")):
                    return f"{head}\n{txt[:200]}"
    return head
