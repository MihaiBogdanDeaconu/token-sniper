from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from token_sniper.parse import extract_python_entities, infer_module_qualname


def test_infer_module_qualname_init(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    f = pkg / "__init__.py"
    f.write_text("# init\n", encoding="utf-8")
    assert infer_module_qualname(f, root) == "pkg"


def test_infer_module_qualname_module_file(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    mod = root / "pkg" / "util.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 1\n", encoding="utf-8")
    assert infer_module_qualname(mod, root) == "pkg.util"


def test_module_preamble_and_top_level_function(tmp_path: Path) -> None:
    src = """import os\n\nX = 1\n\ndef foo():\n    return os.path.join(X, "a")\n"""
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    ents = extract_python_entities(f, project_root=tmp_path, token_threshold=10_000)
    assert len(ents) == 2
    pre, fn = ents
    assert pre.entity_type == "module_preamble"
    assert "import os" in pre.source_text
    assert "os" in pre.referenced_identifiers
    assert fn.entity_type == "function"
    assert fn.fully_qualified_name.endswith(".foo")
    assert "os" in fn.referenced_identifiers


def test_decorated_class_and_method() -> None:
    src = '''from abc import ABC\n\n@dataclass\nclass C(Base):\n    """cls doc"""\n    x = 1\n\n    @staticmethod\n    def m(self):\n        return self\n'''
    ents = extract_python_entities(
        "dummy.py", source=src, project_root=None, token_threshold=10_000
    )
    types = [e.entity_type for e in ents]
    assert "module_preamble" in types
    classes = [e for e in ents if e.entity_type == "class"]
    methods = [e for e in ents if e.entity_type == "method"]
    assert len(classes) == 1
    assert classes[0].fully_qualified_name.endswith(".C")
    assert "Base" in classes[0].referenced_identifiers
    assert len(methods) == 1
    assert methods[0].fully_qualified_name.endswith(".C.m")
    assert "self" in methods[0].referenced_identifiers


def test_nested_function_fqn() -> None:
    src = "def outer():\n    def inner():\n        pass\n    return inner\n"
    ents = extract_python_entities("m.py", source=src, project_root=None, token_threshold=10_000)
    fns = [e for e in ents if e.entity_type == "function"]
    fqns = {e.fully_qualified_name for e in fns}
    assert any(f.endswith(".outer") for f in fqns)
    assert any(f.endswith(".outer.inner") for f in fqns)


def test_signature_summary_when_over_token_threshold() -> None:
    body = "\n".join([f"    x_{i} = {i}" for i in range(400)])
    src = f"def big():\n{body}\n    return x_0\n"
    ents = extract_python_entities("b.py", source=src, project_root=None, token_threshold=50)
    fn = next(e for e in ents if e.entity_type == "function")
    assert fn.signature_summary is not None
    assert "def big" in fn.signature_summary


def test_broken_syntax_does_not_crash() -> None:
    src = "def incomplete(\n"
    ents = extract_python_entities("bad.py", source=src, project_root=None)
    assert isinstance(ents, list)


def test_metaclass_parse() -> None:
    src = "class Meta(type):\n    pass\n\nclass C(metaclass=Meta):\n    FIELD = 1\n"
    ents = extract_python_entities("meta.py", source=src, project_root=None, token_threshold=10_000)
    names = [e.fully_qualified_name for e in ents if e.entity_type == "class"]
    assert any(n.endswith(".Meta") for n in names)
    assert any(n.endswith(".C") for n in names)


@pytest.mark.parametrize(
    "src",
    [
        "from x import *\n",
        "class C:\n    __slots__ = ('a',)\n",
    ],
)
def test_edge_import_patterns_parse(src: str) -> None:
    ents = extract_python_entities("edge.py", source=src, project_root=None)
    assert isinstance(ents, list)
