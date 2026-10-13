from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from token_sniper.evaluation.arms.compressors import (
    LLMLingua2,
    SwePruner,
    ppr_ranking,
    row_runs,
)
from token_sniper.evaluation.arms.context import (
    CONTEXT_HEADER,
    Span,
    count_tokens,
    gold_units,
    max_under_budget,
    normalize,
    pack_to_budget,
    patch_files,
    recall,
    render,
    source_rows,
    truncate_to_budget,
    whole_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from token_sniper.evaluation.phase0 import Phase0Task

logger = logging.getLogger(__name__)

ARMS = (
    "shell_only",
    "bm25",
    "dense_knn",
    "hybrid",
    "dense_reranker",
    "dense_reranker_vgib",
    "dense_reranker_vgib_rl",
    "oracle_gold_files",
    "truncate_to_budget",
    "ppr_topk",
    "llmlingua2",
    "swe_pruner",
)
BUDGET_ARMS = ("truncate_to_budget", "ppr_topk", "llmlingua2", "swe_pruner")
CHECKPOINT_ARMS = {"dense_reranker_vgib": "vgib_checkpoint", "dense_reranker_vgib_rl": "grpo_checkpoint"}
RERANKER_ARMS = ("dense_reranker", "dense_reranker_vgib", "dense_reranker_vgib_rl", *BUDGET_ARMS)
STAGE_A_ARMS = ("hybrid", *RERANKER_ARMS)
LIVE_RETRIEVAL_ARMS = ("bm25", "dense_knn", "ppr_topk")
SELECTION_VERSION = 1


@dataclass(frozen=True)
class SelectionConfig:
    ssg_dir: Path
    repos_dir: Path
    cache_root: Path | None = None
    cache_name: str | None = None
    chromadb_dir: Path | None = None
    reranker_model: Path | None = None
    vgib_checkpoint: Path | None = None
    grpo_checkpoint: Path | None = None
    swe_pruner_model: Path | None = None
    top_k_files: int = 10
    subgraph_top_k: int = 20
    fallback_top_k: int = 5
    retrieval_k: int = 200
    ppr_seeds: int = 5
    max_context_tokens: int = 32_000
    budget_tokens: int | None = None
    budget_arm: str = "dense_reranker_vgib"
    device: str = "cpu"
    compressor_device: str = "cuda"


@dataclass(frozen=True)
class TaskInput:
    task: Phase0Task
    issue: str
    gold_patch: str
    graph: Any
    gold_files: list[str] = field(default_factory=list)
    units: dict[str, frozenset[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    arm: str
    mode: str
    """"blind" (no preloaded context) or "oracle" (context preloaded), as agent.runner names them."""
    context: str | None
    context_tokens: int
    retrieved_files: list[str]
    retrieved_spans: list[dict[str, Any]]
    retained_span_count: int
    fallback_triggered: bool
    selector_latency_ms: float | None
    selector_checkpoint_sha256: str | None
    beta: float | None
    gold_files: list[str]
    file_recall: float
    function_recall: float
    recall_on: str
    """"shown": recall of the rows in the context; "compressor_input" for llmlingua2 (see report)."""


def checkpoint_identity(path: Path, *, require_beta: bool) -> tuple[str, float | None]:
    import torch

    from token_sniper.evaluation.eval_stages import sha256_file

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    beta = ckpt.get("beta", ckpt.get("train_cfg", {}).get("beta"))
    if beta is None and require_beta:
        raise ValueError(f"{path} stores no beta (neither a 'beta' key nor train_cfg['beta'])")
    if beta is None:
        logger.warning("%s (mode %s) stores no beta; its records carry beta=None", path, ckpt.get("mode"))
    return sha256_file(path), None if beta is None else float(beta)


class Selector:
    def __init__(self, cfg: SelectionConfig, *, cache_dir: Path | None, tokenizer: Any = None) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._resources: dict[str, Any] = {}

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from token_sniper.vgib_selector.token_counts import load_tokenizer

            self._tokenizer = load_tokenizer()
        return self._tokenizer

    def _resource(self, name: str, build: Any) -> Any:
        if name not in self._resources:
            self._resources[name] = build()
        return self._resources[name]

    def _store(self) -> Any:
        def build() -> Any:
            from token_sniper.retrieval.vectorstore import EntityVectorStore

            if self.cfg.chromadb_dir is None:
                raise ValueError("this arm retrieves live and needs --chromadb-dir")
            return EntityVectorStore(persist_directory=str(self.cfg.chromadb_dir), collection_prefix="token_sniper")
        return self._resource("store", build)

    def _embedder(self) -> Any:
        def build() -> Any:
            from scripts.eval_candidates import EMBEDDING_DIM, EMBEDDING_MODEL
            from token_sniper.retrieval.config import EmbeddingConfig
            from token_sniper.retrieval.embeddings import create_embedder

            return create_embedder(EmbeddingConfig(
                provider="sentence-transformers", name=EMBEDDING_MODEL, dimension=EMBEDDING_DIM,
                device=self.cfg.device,
            ))
        return self._resource("embedder", build)

    def _booster(self) -> tuple[Any, str]:
        def build() -> tuple[Any, str]:
            import xgboost as xgb

            from token_sniper.evaluation.eval_stages import sha256_file

            if self.cfg.reranker_model is None:
                raise ValueError("this arm needs --reranker-model")
            booster = xgb.Booster()
            booster.load_model(str(self.cfg.reranker_model))
            return booster, sha256_file(self.cfg.reranker_model)
        return self._resource("booster", build)

    def _checkpoint(self, arm: str) -> tuple[str, float | None]:
        option = CHECKPOINT_ARMS[arm]
        path = getattr(self.cfg, option)
        if path is None:
            raise ValueError(f"arm {arm} needs --{option.replace('_', '-')}")
        return self._resource(option, lambda: checkpoint_identity(path, require_beta=arm == "dense_reranker_vgib"))

    def _stage_cache(self) -> Path:
        if self.cfg.cache_root is None or self.cfg.cache_name is None:
            raise ValueError("this arm reads the cached eval stages and needs --cache-root and --cache-name")
        return self.cfg.cache_root / self.cfg.cache_name

    def sources(self, task: Phase0Task, paths: list[str]) -> dict[str, list[str]]:
        from token_sniper.vgib_selector.token_counts import read_blobs

        git_dir = self.cfg.repos_dir / task.repo.replace("/", "__")
        if not git_dir.is_dir():
            raise FileNotFoundError(f"no cached bare repo for {task.repo} at {git_dir}")
        blobs = read_blobs(git_dir, task.base_commit, sorted(set(paths)))
        return {path: source_rows(blob) for path, blob in blobs.items()}

    def candidates(self, task: Phase0Task) -> tuple[dict[str, Any], str]:
        from token_sniper.evaluation.eval_stages import candidates_path, read_record, sha256_file

        path = candidates_path(self._stage_cache(), task.instance_id)
        if not path.exists():
            raise FileNotFoundError(f"no stage-A record {path}: run scripts/eval_candidates.py first")
        _, body = read_record(path)
        if "excluded" in body:
            raise ValueError(f"stage A excluded {task.instance_id}: {body['excluded']}")
        return body, sha256_file(path)

    def header(self, arm: str, ti: TaskInput) -> dict[str, Any]:
        from token_sniper.vgib_selector.token_counts import TOKENIZER_NAME, TOKENIZER_REVISION

        cfg = self.cfg
        config: dict[str, Any] = {
            "version": SELECTION_VERSION,
            "context_header": CONTEXT_HEADER,
            "tokenizer": [TOKENIZER_NAME, TOKENIZER_REVISION],
            "max_context_tokens": cfg.max_context_tokens,
            "base_commit": ti.task.base_commit,
        }
        if arm not in ("shell_only", "oracle_gold_files"):
            config["top_k_files"] = cfg.top_k_files
        if arm in LIVE_RETRIEVAL_ARMS:
            config["retrieval_k"] = cfg.retrieval_k
        if arm in STAGE_A_ARMS:
            config["cache_name"] = cfg.cache_name
            config["candidates_sha256"] = self.candidates(ti.task)[1]
        if arm in RERANKER_ARMS:
            config["reranker_sha256"] = self._booster()[1]
        if arm in CHECKPOINT_ARMS:
            config["checkpoint_sha256"] = self._checkpoint(arm)[0]
            config["subgraph_top_k"] = cfg.subgraph_top_k
            config["fallback_top_k"] = cfg.fallback_top_k
        if arm in BUDGET_ARMS:
            config["budget"] = cfg.budget_tokens if cfg.budget_tokens is not None else {
                "arm": cfg.budget_arm, **self.header(cfg.budget_arm, ti)["config"]}
        if arm == "ppr_topk":
            config["subgraph_top_k"] = cfg.subgraph_top_k
            config["ppr_seeds"] = cfg.ppr_seeds
        return {"arm": arm, "instance_id": ti.task.instance_id, "config": config}

    def get(self, arm: str, ti: TaskInput) -> Selection:
        from token_sniper.evaluation.eval_stages import read_record, write_record

        header = self.header(arm, ti)
        path = None if self.cache_dir is None else self.cache_dir / arm / f"{ti.task.instance_id}.json"
        if path is not None and path.exists():
            _, body = read_record(path, header)
            return Selection(**body)
        selection = self.compute(arm, ti)
        if path is not None:
            write_record(path, header, asdict(selection))
        return selection

    def compute(self, arm: str, ti: TaskInput) -> Selection:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        return getattr(self, f"_select_{arm}")(ti)

    def _finish(self, arm: str, ti: TaskInput, spans: list[Span], *, retained: int = 0,
                fallback: bool = False, latency_ms: float | None = None,
                checkpoint: tuple[str, float | None] = (None, None)) -> Selection:
        sources = self.sources(ti.task, [s.file_path for s in spans])
        groups = truncate_to_budget(normalize(spans, sources), sources, self.tokenizer, self.cfg.max_context_tokens)
        context = render(groups, sources)
        return self._selection(arm, ti, groups, context, retained=retained, fallback=fallback,
                               latency_ms=latency_ms, checkpoint=checkpoint)

    def _selection(self, arm: str, ti: TaskInput, groups: list[Span], context: str, *, retained: int,
                   fallback: bool, latency_ms: float | None, checkpoint: tuple[str, float | None],
                   recall_on: str = "shown") -> Selection:
        files: list[str] = []
        for group in groups:
            if group.file_path not in files:
                files.append(group.file_path)
        partial = any(g.node_type != "file" for g in groups)
        file_recall, function_recall = recall(groups, ti.gold_files, ti.units, ti.graph)
        return Selection(
            arm=arm, mode="oracle", context=context, context_tokens=count_tokens(self.tokenizer, context),
            retrieved_files=files, retrieved_spans=[g.to_dict() for g in groups] if partial else [],
            retained_span_count=retained, fallback_triggered=fallback, selector_latency_ms=latency_ms,
            selector_checkpoint_sha256=checkpoint[0], beta=checkpoint[1], gold_files=ti.gold_files,
            file_recall=file_recall, function_recall=function_recall, recall_on=recall_on,
        )

    def _whole_files(self, ti: TaskInput, files: list[str]) -> list[Span]:
        sources = self.sources(ti.task, files)
        return [whole_file(path, sources) for path in files]

    def _reranked(self, ti: TaskInput) -> tuple[list[str], float]:
        import numpy as np
        import xgboost as xgb

        from token_sniper.retrieval.reranker import FEATURE_NAMES

        body, _ = self.candidates(ti.task)
        booster, _ = self._booster()
        t0 = time.perf_counter()
        scores = booster.predict(xgb.DMatrix(np.asarray(body["feature_matrix"]), feature_names=FEATURE_NAMES))
        latency = (time.perf_counter() - t0) * 1000
        ranked = sorted(zip(body["candidate_files"], scores.tolist(), strict=True), key=lambda pair: -pair[1])
        return [fp for fp, _ in ranked], latency

    def _collection_repo(self, ti: TaskInput) -> str:
        return f"{ti.task.repo}@{ti.task.base_commit[:12]}"

    def _live_files(self, ti: TaskInput, retriever: Any) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        result = retriever.retrieve(issue_text=ti.issue, repo=self._collection_repo(ti), k=self.cfg.retrieval_k)
        latency = (time.perf_counter() - t0) * 1000
        files = result.file_paths[:self.cfg.top_k_files]
        if not files:
            raise ValueError(f"{result.strategy} retrieval returned no entities for {self._collection_repo(ti)}")
        return files, latency

    def _budget(self, ti: TaskInput) -> int:
        if self.cfg.budget_tokens is not None:
            return self.cfg.budget_tokens
        return self.get(self.cfg.budget_arm, ti).context_tokens

    def _select_shell_only(self, ti: TaskInput) -> Selection:
        file_recall, function_recall = recall([], ti.gold_files, ti.units, ti.graph)
        return Selection(
            arm="shell_only", mode="blind", context=None, context_tokens=0, retrieved_files=[],
            retrieved_spans=[], retained_span_count=0, fallback_triggered=False, selector_latency_ms=None,
            selector_checkpoint_sha256=None, beta=None, gold_files=ti.gold_files, file_recall=file_recall,
            function_recall=function_recall, recall_on="shown",
        )

    def _select_bm25(self, ti: TaskInput) -> Selection:
        from token_sniper.retrieval.retriever import BM25Retriever

        retriever = self._resource("bm25", lambda: BM25Retriever(vector_store=self._store()))
        files, latency = self._live_files(ti, retriever)
        return self._finish("bm25", ti, self._whole_files(ti, files), latency_ms=latency)

    def _select_dense_knn(self, ti: TaskInput) -> Selection:
        from token_sniper.retrieval.retriever import DenseRetriever

        retriever = self._resource("dense", lambda: DenseRetriever(embedder=self._embedder(),
                                                                   vector_store=self._store()))
        files, latency = self._live_files(ti, retriever)
        return self._finish("dense_knn", ti, self._whole_files(ti, files), latency_ms=latency)

    def _select_hybrid(self, ti: TaskInput) -> Selection:
        body, _ = self.candidates(ti.task)
        if not body["entities"]:
            raise ValueError(f"the stage-A record of {ti.task.instance_id} has no entities (imported record); "
                             "rebuild it with scripts/eval_candidates.py")
        files: list[str] = []
        for entity in sorted(body["entities"], key=lambda e: e["rank"]):
            if entity["file_path"] not in files:
                files.append(entity["file_path"])
        return self._finish("hybrid", ti, self._whole_files(ti, files[:self.cfg.top_k_files]))

    def _select_dense_reranker(self, ti: TaskInput) -> Selection:
        ranked, latency = self._reranked(ti)
        return self._finish("dense_reranker", ti, self._whole_files(ti, ranked[:self.cfg.top_k_files]),
                            latency_ms=latency)

    def _vgib(self, arm: str, ti: TaskInput) -> Selection:
        from scripts.eval_retention import retention_header
        from token_sniper.evaluation.eval_stages import KEEP_THRESHOLD, read_record, retention_path
        from token_sniper.graph.schema import is_file_node_id

        checkpoint = self._checkpoint(arm)
        _, candidates_sha = self.candidates(ti.task)
        path = retention_path(self._stage_cache(), checkpoint[0], ti.task.instance_id)
        if not path.exists():
            raise FileNotFoundError(f"no stage-B record {path}: run scripts/eval_retention.py for this checkpoint")
        expected = retention_header(checkpoint[0], self._booster()[1], candidates_sha, self.cfg.subgraph_top_k)
        _, body = read_record(path, expected)
        if "selector_latency_ms" not in body:
            raise ValueError(f"{path} predates selector_latency_ms: delete it and rerun scripts/eval_retention.py")

        spans = []
        for node in body["nodes"]:
            if node["p_v"] < KEEP_THRESHOLD or is_file_node_id(node["id"]):
                continue
            data = ti.graph.nodes[node["id"]]
            spans.append(Span(data["file_path"], data["start_line"], data["end_line"], data["node_type"]))
        if spans:
            return self._finish(arm, ti, spans, retained=len(spans), latency_ms=body["selector_latency_ms"],
                                checkpoint=checkpoint)
        logger.warning("%s kept no node for %s: showing the reranker's top %d files (fallback_triggered)",
                       arm, ti.task.instance_id, self.cfg.fallback_top_k)
        files = [fp for fp, _ in body["ranked_files"][:self.cfg.fallback_top_k]]
        return self._finish(arm, ti, self._whole_files(ti, files), fallback=True,
                            latency_ms=body["selector_latency_ms"], checkpoint=checkpoint)

    def _select_dense_reranker_vgib(self, ti: TaskInput) -> Selection:
        return self._vgib("dense_reranker_vgib", ti)

    def _select_dense_reranker_vgib_rl(self, ti: TaskInput) -> Selection:
        return self._vgib("dense_reranker_vgib_rl", ti)

    def _select_oracle_gold_files(self, ti: TaskInput) -> Selection:
        existing = [f for f in ti.gold_files if f not in _created_files(ti.gold_patch)]
        return self._finish("oracle_gold_files", ti, self._whole_files(ti, existing))

    def _reranker_groups(self, ti: TaskInput) -> tuple[list[Span], dict[str, list[str]]]:
        ranked, _ = self._reranked(ti)
        files = ranked[:self.cfg.top_k_files]
        sources = self.sources(ti.task, files)
        groups = truncate_to_budget(normalize([whole_file(f, sources) for f in files], sources), sources,
                                    self.tokenizer, self.cfg.max_context_tokens)
        return groups, sources

    def _select_truncate_to_budget(self, ti: TaskInput) -> Selection:
        budget = self._budget(ti)
        groups, sources = self._reranker_groups(ti)
        groups = truncate_to_budget(groups, sources, self.tokenizer, budget)
        return self._selection("truncate_to_budget", ti, groups, render(groups, sources), retained=0,
                               fallback=False, latency_ms=None, checkpoint=(None, None))

    def _select_ppr_topk(self, ti: TaskInput) -> Selection:
        from token_sniper.retrieval.entry_points import identify_entry_points_by_embedding

        budget = self._budget(ti)
        ranked, _ = self._reranked(ti)
        files = ranked[:self.cfg.subgraph_top_k]
        t0 = time.perf_counter()
        entry = identify_entry_points_by_embedding(
            issue_text=ti.issue, repo=self._collection_repo(ti), embedder=self._embedder(),
            vector_store=self._store(), graph=ti.graph, n_candidates=self.cfg.ppr_seeds,
        )
        if entry.error is not None:
            raise RuntimeError(f"PPR seeds for {ti.task.instance_id}: {entry.error}")
        nodes = ppr_ranking(ti.graph, entry.matched_node_ids, files)
        spans = [Span(ti.graph.nodes[nid]["file_path"], ti.graph.nodes[nid]["start_line"],
                      ti.graph.nodes[nid]["end_line"], ti.graph.nodes[nid]["node_type"]) for nid, _ in nodes]
        sources = self.sources(ti.task, files)
        kept, groups = pack_to_budget(spans, sources, self.tokenizer, budget)
        latency = (time.perf_counter() - t0) * 1000
        return self._selection("ppr_topk", ti, groups, render(groups, sources), retained=len(kept),
                               fallback=False, latency_ms=latency, checkpoint=(None, None))

    def _select_llmlingua2(self, ti: TaskInput) -> Selection:
        budget = self._budget(ti)
        groups, sources = self._reranker_groups(ti)
        body = render(groups, sources).removeprefix(CONTEXT_HEADER + "\n")
        compressor = self._resource("llmlingua2", lambda: LLMLingua2(self.cfg.compressor_device))
        t0 = time.perf_counter()
        _, context = max_under_budget(lambda rate: f"{CONTEXT_HEADER}\n{compressor.compress(body, rate)}",
                                      lambda text: count_tokens(self.tokenizer, text), budget, lo=0.02, hi=1.0)
        latency = (time.perf_counter() - t0) * 1000
        return self._selection("llmlingua2", ti, groups, context, retained=0, fallback=False,
                               latency_ms=latency, checkpoint=(None, None), recall_on="compressor_input")

    def _select_swe_pruner(self, ti: TaskInput) -> Selection:
        if self.cfg.swe_pruner_model is None:
            raise ValueError("arm swe_pruner needs --swe-pruner-model")
        budget = self._budget(ti)
        ranked, _ = self._reranked(ti)
        files = ranked[:self.cfg.top_k_files]
        sources = self.sources(ti.task, files)
        pruner = self._resource("swe_pruner", lambda: SwePruner(self.cfg.swe_pruner_model))
        kept_by_threshold: dict[float, list[Span]] = {}

        def pruned(keep: float) -> str:
            threshold = 1.0 - keep
            spans = [s for f in files for s in row_runs(f, pruner.kept_rows(ti.issue, sources[f], threshold))]
            kept_by_threshold[keep] = spans
            return render(normalize(spans, sources), sources)

        t0 = time.perf_counter()
        keep, _ = max_under_budget(pruned, lambda text: count_tokens(self.tokenizer, text), budget, lo=0.0, hi=1.0)
        latency = (time.perf_counter() - t0) * 1000
        spans = kept_by_threshold[keep]
        groups = normalize(spans, sources)
        return self._selection("swe_pruner", ti, groups, render(groups, sources), retained=len(spans),
                               fallback=False, latency_ms=latency, checkpoint=(None, None))


def _created_files(patch: str) -> set[str]:
    created: set[str] = set()
    old: str | None = None
    for line in patch.splitlines():
        if line.startswith("--- "):
            old = line[4:].split("\t")[0].strip()
        elif line.startswith("+++ ") and old == "/dev/null":
            created.add(line[4:].split("\t")[0].strip().removeprefix("b/"))
    return created


def task_input(task: Phase0Task, *, issue: str, gold_patch: str, graph: Any) -> TaskInput:
    return TaskInput(task=task, issue=issue, gold_patch=gold_patch, graph=graph,
                     gold_files=patch_files(gold_patch), units=gold_units(graph, gold_patch))
