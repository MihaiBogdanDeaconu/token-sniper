from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from token_sniper.evaluation.arms.context import Span

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
LLMLINGUA_INSTALL = "pip install llmlingua==0.2.2"
SWE_PRUNER_INSTALL = (
    "pip install swe-pruner==0.1.1 plus a flash-attn wheel matching torch/CUDA "
    "(needs Python >= 3.12, torch 2.8, a CUDA GPU); weights: "
    "huggingface-cli download ayanami-kitasan/code-pruner --local-dir <dir>"
)
_EXTRA_LINE_BREAKS = re.compile("[\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def ppr_ranking(graph: Any, seeds: Sequence[str], candidate_files: Sequence[str]) -> list[tuple[str, float]]:
    import networkx as nx

    from token_sniper.graph.schema import is_file_node_id

    if not seeds:
        raise ValueError("personalized PageRank needs at least one seed node")
    undirected = nx.Graph(graph.to_undirected(as_view=True))
    scores = nx.pagerank(undirected, alpha=0.85, personalization={s: 1.0 / len(seeds) for s in seeds},
                         max_iter=200, tol=1e-6)
    files = set(candidate_files)
    nodes = [(nid, scores[nid]) for nid, data in graph.nodes(data=True)
             if not is_file_node_id(nid) and data["file_path"] in files]
    return sorted(nodes, key=lambda pair: (-pair[1], pair[0]))


class LLMLingua2:
    def __init__(self, device: str) -> None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise RuntimeError(f"the llmlingua2 arm needs LLMLingua: {LLMLINGUA_INSTALL}") from exc
        self._compressor = PromptCompressor(model_name=LLMLINGUA2_MODEL, use_llmlingua2=True, device_map=device)

    def compress(self, text: str, rate: float) -> str:
        out = self._compressor.compress_prompt(text, rate=rate, force_tokens=["\n"], drop_consecutive=True)
        return out["compressed_prompt"]


class SwePruner:
    def __init__(self, model_path: Path) -> None:
        try:
            from swe_pruner.prune_wrapper import PruneRequest, SwePrunerForCodePruning
        except ImportError as exc:
            raise RuntimeError(f"the swe_pruner arm needs SWE-Pruner: {SWE_PRUNER_INSTALL}") from exc
        if not (model_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"no SWE-Pruner weights at {model_path}: {SWE_PRUNER_INSTALL}")
        self._request = PruneRequest
        self._model = SwePrunerForCodePruning.from_pretrained(str(model_path))

    def kept_rows(self, query: str, rows: list[str], threshold: float) -> list[int]:
        code = _EXTRA_LINE_BREAKS.sub(" ", "\n".join(rows))
        response = self._model.prune(self._request(query=query, code=code, threshold=threshold))
        if response.error_msg:
            raise RuntimeError(f"SWE-Pruner failed: {response.error_msg}")
        return sorted(response.kept_frags)


def row_runs(path: str, kept: Sequence[int]) -> list[Span]:
    spans: list[Span] = []
    for row in kept:
        if spans and row == spans[-1].end_line + 1:
            spans[-1] = Span(path, spans[-1].start_line, row, "lines")
        else:
            spans.append(Span(path, row, row, "lines"))
    return spans
