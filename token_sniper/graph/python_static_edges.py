from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser


@dataclass(frozen=True)
class ImportRecord:
    module: str
    """Target dotted module (e.g. ``os``, ``pkg.util``)."""

    names: tuple[str, ...] | None
    """For ``from m import a, b``; ``None`` means ``import module`` (whole module)."""


@dataclass(frozen=True)
class ClassBasesRecord:
    class_fqn: str
    """Fully qualified class name (``module_qualname`` + nested class path)."""

    base_names: tuple[str, ...]
    """Unqualified base symbols (last segment of attributes)."""


@dataclass(frozen=True)
class CallSiteTarget:
    leaf: str
    """Rightmost callee identifier (``join`` in ``os.path.join()``)."""

    qualified_hint: str | None
    """When the callee is ``identifier.attribute`` (not ``self.attr`` / nested attr), ``obj.attr``."""


def _parser() -> Parser:
    return Parser(Language(tsp.language()))


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _dotted_name_text(node: Node | None, source_bytes: bytes) -> str:
    if node is None:
        return ""
    return _node_text(node, source_bytes).strip()


def extract_imports(source_bytes: bytes) -> list[ImportRecord]:
    tree = _parser().parse(source_bytes)
    root = tree.root_node
    if root.type != "module":
        return []

    out: list[ImportRecord] = []
    for ch in root.children:
        if ch.type == "import_statement":
            out.extend(_import_statement_records(ch, source_bytes))
        elif ch.type == "import_from_statement":
            rec = _import_from_statement_record(ch, source_bytes)
            if rec is not None:
                out.append(rec)
    return out


def _import_statement_records(node: Node, source_bytes: bytes) -> list[ImportRecord]:
    records: list[ImportRecord] = []
    i = 0
    children = list(node.children)
    while i < len(children):
        c = children[i]
        if c.type == "dotted_name":
            records.append(ImportRecord(module=_dotted_name_text(c, source_bytes), names=None))
            i += 1
        elif c.type == "aliased_import":
            dn = c.child_by_field_name("name")
            mod = _dotted_name_text(dn, source_bytes)
            records.append(ImportRecord(module=mod, names=None))
            i += 1
        else:
            i += 1
    return records


def _import_from_statement_record(node: Node, source_bytes: bytes) -> ImportRecord | None:
    module: str | None = None
    past_import_kw = False
    names: list[str] = []
    for c in node.children:
        if c.type == "import":
            past_import_kw = True
            continue
        if c.type == "wildcard_import":
            return ImportRecord(module=module or "", names=None)
        if c.type == "dotted_name" and not past_import_kw:
            module = _dotted_name_text(c, source_bytes)
        elif c.type == "dotted_name" and past_import_kw:
            names.append(_dotted_name_text(c, source_bytes).split(".")[-1])
        elif c.type == "aliased_import" and past_import_kw:
            dn = c.child_by_field_name("name")
            if dn is not None:
                names.append(_dotted_name_text(dn, source_bytes).split(".")[-1])
        elif c.type == "dotted_as_names" and past_import_kw:
            for g in c.children:
                if g.type == "dotted_as_name":
                    inner = g.child_by_field_name("name")
                    if inner is None:
                        inner = next((x for x in g.children if x.type == "dotted_name"), None)
                    if inner is not None:
                        names.append(_dotted_name_text(inner, source_bytes).split(".")[-1])
                elif g.type == "dotted_name":
                    names.append(_dotted_name_text(g, source_bytes).split(".")[-1])

    if module is None:
        return None
    if not names:
        return ImportRecord(module=module, names=None)
    return ImportRecord(module=module, names=tuple(dict.fromkeys(names)))


def extract_class_bases(source_bytes: bytes, module_qualname: str) -> list[ClassBasesRecord]:
    tree = _parser().parse(source_bytes)
    root = tree.root_node
    if root.type != "module":
        return []

    results: list[ClassBasesRecord] = []

    def class_fqn(scope: list[str], name: str) -> str:
        parts = [p for p in [module_qualname, *scope, name] if p]
        return ".".join(parts)

    def bases_from_class_node(class_node: Node) -> tuple[str, ...]:
        supers = class_node.child_by_field_name("superclasses")
        if supers is None:
            return ()
        found: list[str] = []
        stack: list[Node] = [supers]
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                found.append(_node_text(n, source_bytes))
            elif n.type == "attribute":
                attr = n.child_by_field_name("attribute")
                if attr is not None:
                    stack.append(attr)
                else:
                    for c in reversed(n.children):
                        stack.append(c)
            else:
                stack.extend(reversed(n.children))
        return tuple(dict.fromkeys(found))

    def walk(node: Node, scope: list[str]) -> None:
        if node.type == "class_definition":
            name_n = node.child_by_field_name("name")
            cname = _node_text(name_n, source_bytes) if name_n else "<class>"
            fq = class_fqn(scope, cname)
            results.append(ClassBasesRecord(class_fqn=fq, base_names=bases_from_class_node(node)))
            body = node.child_by_field_name("body")
            if body is not None:
                for stmt in body.children:
                    walk(stmt, scope + [cname])
        else:
            for ch in node.children:
                walk(ch, scope)

    for ch in root.children:
        walk(ch, [])
    return results


def extract_call_targets(source_bytes: bytes) -> tuple[CallSiteTarget, ...]:
    tree = _parser().parse(source_bytes)
    root = tree.root_node
    found: list[CallSiteTarget] = []
    stack: list[Node] = [root]
    while stack:
        n = stack.pop()
        if n.type == "call":
            tgt = _call_site_target(n, source_bytes)
            if tgt is not None:
                found.append(tgt)
        stack.extend(reversed(n.children))
    seen: set[tuple[str | None, str]] = set()
    out: list[CallSiteTarget] = []
    for t in found:
        key = (t.qualified_hint, t.leaf)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return tuple(out)


def extract_call_sites(source_bytes: bytes) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t.leaf for t in extract_call_targets(source_bytes)))


def _call_site_target(call_node: Node, source_bytes: bytes) -> CallSiteTarget | None:
    func = call_node.child_by_field_name("function")
    if func is None:
        return None
    leaf = _expression_leaf_name(func, source_bytes)
    if not leaf:
        return None
    hint = _simple_attribute_prefix_hint(func, source_bytes)
    return CallSiteTarget(leaf=leaf, qualified_hint=hint)


def _simple_attribute_prefix_hint(func_node: Node, source_bytes: bytes) -> str | None:
    if func_node.type != "attribute":
        return None
    obj = func_node.child_by_field_name("object")
    attr = func_node.child_by_field_name("attribute")
    if obj is None or attr is None or obj.type != "identifier" or attr.type != "identifier":
        return None
    left = _node_text(obj, source_bytes)
    right = _node_text(attr, source_bytes)
    return f"{left}.{right}"


def _expression_leaf_name(node: Node, source_bytes: bytes) -> str | None:
    if node.type == "identifier":
        return _node_text(node, source_bytes)
    if node.type == "attribute":
        attr = node.child_by_field_name("attribute")
        if attr is not None and attr.type == "identifier":
            return _node_text(attr, source_bytes)
        return _expression_leaf_name(attr, source_bytes) if attr is not None else None
    return None
