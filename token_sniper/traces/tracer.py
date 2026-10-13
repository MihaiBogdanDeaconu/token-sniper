from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class TraceResult:
    executed_nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str, int]] = field(default_factory=set)
    dropped_out_of_repo: int = 0
    dropped_unmappable: int = 0
    call_events: int = 0


class CallEdgeTracer:
    def __init__(self, project_root: Path):
        self._root = project_root.resolve()
        self._root_str = str(self._root)
        self._file_to_module: dict[str, str] = {}
        self._fqn_cache: dict[tuple[str, str], str | None] = {}
        self.result = TraceResult()
        self._previous: object = None

    def _module_qualname(self, abs_file: str) -> str | None:
        cached = self._file_to_module.get(abs_file)
        if cached is not None:
            return cached or None

        if not abs_file.startswith(self._root_str):
            self._file_to_module[abs_file] = ""
            return None

        rel = abs_file[len(self._root_str):].lstrip("/\\")
        if not rel.endswith(".py"):
            self._file_to_module[abs_file] = ""
            return None

        parts = rel[:-3].replace("\\", "/").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_q = ".".join(p for p in parts if p)
        self._file_to_module[abs_file] = module_q
        return module_q

    _WRAPPER_NAMES = frozenset({
        "wrapper", "_wrapper", "inner", "_inner", "_wrapped",
        "wrapped", "decorator", "_decorator", "new_func",
        "bound_method", "_bound_method",
    })

    def _resolve_frame(self, frame: FrameType | None) -> str | None:
        if frame is None:
            return None
        code = frame.f_code
        qualname = getattr(code, "co_qualname", code.co_name)

        qualname, filename = self._unwrap_decorated(frame, qualname, code.co_filename)

        key = (filename, qualname)
        cached = self._fqn_cache.get(key)
        if cached is not None:
            return cached or None

        module_q = self._module_qualname(filename)
        if module_q is None:
            self._fqn_cache[key] = ""
            return None

        lower = qualname.lower()
        if "<locals>" in qualname and any(
            marker in lower for marker in ("<lambda>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")
        ):
            self._fqn_cache[key] = ""
            self.result.dropped_unmappable += 1
            return None
        if qualname in {"<lambda>", "<module>"}:
            self._fqn_cache[key] = ""
            self.result.dropped_unmappable += 1
            return None

        qualname = self._strip_locals_wrapper(qualname)

        fqn = f"{module_q}.{qualname}" if module_q else qualname
        self._fqn_cache[key] = fqn
        return fqn

    def _unwrap_decorated(
        self, frame: FrameType, qualname: str, filename: str
    ) -> tuple[str, str]:
        if "<locals>" not in qualname:
            return qualname, filename
        last = qualname.rsplit(".", 1)[-1]
        if last not in self._WRAPPER_NAMES:
            return qualname, filename

        for ns in (frame.f_locals, frame.f_globals):
            wrapped = ns.get("__wrapped__")
            if wrapped is not None and callable(wrapped):
                try:
                    inner_code = wrapped.__code__
                    return getattr(inner_code, "co_qualname", inner_code.co_name), inner_code.co_filename
                except AttributeError:
                    pass

        for name in ("func", "fn", "wrapped", "original", "method"):
            obj = frame.f_locals.get(name)
            if obj is not None and callable(obj):
                try:
                    inner_code = obj.__code__
                    return getattr(inner_code, "co_qualname", inner_code.co_name), inner_code.co_filename
                except AttributeError:
                    pass

        if code_closure := frame.f_code.co_freevars:
            for var in code_closure:
                try:
                    obj = frame.f_locals.get(var)
                    if obj is not None and callable(obj) and hasattr(obj, "__code__"):
                        inner_code = obj.__code__
                        return getattr(inner_code, "co_qualname", inner_code.co_name), inner_code.co_filename
                except (TypeError, AttributeError):
                    pass

        return qualname, filename

    @staticmethod
    def _strip_locals_wrapper(qualname: str) -> str:
        if "<locals>" not in qualname:
            return qualname
        parts = qualname.split(".")
        stripped: list[str] = []
        i = 0
        while i < len(parts):
            if parts[i] == "<locals>" and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part.lstrip("_") in {
                    "wrapper", "inner", "wrapped", "decorator",
                    "newfunc", "boundfunc", "boundmethod",
                } or next_part.startswith("_wrap"):
                    i += 2
                    continue
            stripped.append(parts[i])
            i += 1
        return ".".join(stripped) if stripped else qualname

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if event != "call":
            return
        self.result.call_events += 1

        callee_fqn = self._resolve_frame(frame)
        if callee_fqn is None:
            self.result.dropped_out_of_repo += 1
            return
        self.result.executed_nodes.add(callee_fqn)

        back = frame.f_back
        caller_fqn = self._resolve_frame(back)
        if caller_fqn is None or back is None:
            return
        self.result.edges.add((caller_fqn, callee_fqn, back.f_lineno))

    @contextmanager
    def trace(self) -> Iterator["CallEdgeTracer"]:
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        try:
            yield self
        finally:
            sys.setprofile(self._previous)  # type: ignore[arg-type]
            logger.info(
                "Tracer stopped: %d call events, %d unique edges, %d unique callees, "
                "%d dropped (out-of-repo), %d dropped (unmappable)",
                self.result.call_events,
                len(self.result.edges),
                len(self.result.executed_nodes),
                self.result.dropped_out_of_repo,
                self.result.dropped_unmappable,
            )
