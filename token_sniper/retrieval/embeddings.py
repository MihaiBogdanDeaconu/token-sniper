from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from token_sniper.retrieval.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class VoyageCodeEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "voyage-code-3",
        batch_size: int = 32,
        max_tokens_per_batch: int = 18_000,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        max_concurrent_batches: int = 6,
    ) -> None:
        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'voyageai' package is required for Voyage embeddings. "
                "Install the project with the 'retrieval' extra."
            ) from exc

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._max_tokens_per_batch = max_tokens_per_batch
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._max_concurrent_batches = max_concurrent_batches
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        return 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = _token_aware_batches(texts, self._batch_size, self._max_tokens_per_batch)
        total_batches = len(batches)
        results_by_index: dict[int, list[list[float]]] = {}

        logger.info(
            "Voyage embedding: %d texts → %d batches (concurrency=%d)",
            len(texts), total_batches, self._max_concurrent_batches,
        )

        with ThreadPoolExecutor(max_workers=self._max_concurrent_batches) as pool:
            future_to_idx = {
                pool.submit(self._embed_with_retry, batch, input_type="document"): idx
                for idx, batch in enumerate(batches)
            }
            for completed, future in enumerate(as_completed(future_to_idx), start=1):
                idx = future_to_idx[future]
                results_by_index[idx] = future.result()
                if completed % 50 == 0 or completed == total_batches:
                    done_texts = sum(len(batches[i]) for i in results_by_index)
                    logger.info(
                        "Voyage progress: %d/%d batches done (~%d/%d texts)",
                        completed, total_batches, done_texts, len(texts),
                    )

        all_embeddings: list[list[float]] = []
        for idx in range(total_batches):
            all_embeddings.extend(results_by_index[idx])

        if self._dimension is None and all_embeddings:
            self._dimension = len(all_embeddings[0])

        logger.info(
            "Voyage embed_documents: %d texts → %d vectors (dim=%s)",
            len(texts), len(all_embeddings), self._dimension,
        )
        return all_embeddings

    def _embed_with_retry(self, batch: list[str], *, input_type: str) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                result = self._client.embed(batch, model=self._model, input_type=input_type)
                return result.embeddings  # type: ignore[no-any-return]
            except Exception as exc:
                last_exc = exc
                error_str = str(exc).lower()
                is_retryable = any(
                    kw in error_str
                    for kw in ("timeout", "timed out", "rate limit", "429", "503", "502", "connection")
                )
                if not is_retryable:
                    raise
                delay = self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Voyage API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self._max_retries, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Voyage API failed after {self._max_retries} retries: {last_exc}"
        ) from last_exc

    def embed_query(self, text: str) -> list[float]:
        result = self._embed_with_retry([text], input_type="query")
        vector = result[0]
        if self._dimension is None:
            self._dimension = len(vector)
        return vector


class SentenceTransformerEmbedder:
    _JINA_CODE_MODELS = {
        "jinaai/jina-code-embeddings-0.5b",
        "jinaai/jina-code-embeddings-1.5b",
    }

    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-code-embeddings-0.5b",
        device: str | None = None,
        batch_size: int = 64,
        trust_remote_code: bool = True,
        target_dimension: int = 256,
        max_seq_length: int = 8192,
    ) -> None:
        self._model_name = model_name
        if device is None:
            import torch
            try:
                if torch.cuda.is_available():
                    torch.cuda.init()
                    device = "cuda"
                else:
                    device = "cpu"
            except RuntimeError:
                device = "cpu"
        self._device = device
        self._batch_size = batch_size
        self._trust_remote_code = trust_remote_code
        self._target_dimension = target_dimension
        self._max_seq_length = max_seq_length
        self._model: object | None = None
        self._dim: int = 0
        self._is_jina_code = model_name in self._JINA_CODE_MODELS

    @property
    def dimension(self) -> int:
        if self._dim == 0:
            self._ensure_model()
        return self._dim

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The 'sentence-transformers' package is required. "
                "pip install sentence-transformers"
            ) from exc

        import torch

        logger.info("Loading embedding model %s on %s …", self._model_name, self._device)
        t0 = time.monotonic()

        model_kwargs: dict[str, object] = {"torch_dtype": torch.bfloat16}
        tokenizer_kwargs: dict[str, object] = {}

        if self._is_jina_code:
            tokenizer_kwargs["padding_side"] = "left"
            try:
                from flash_attn import flash_attn_func  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("FlashAttention2 available — using it")
            except ImportError:
                logger.info("FlashAttention2 not available — using default attention")

        model = SentenceTransformer(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            device=self._device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        model.max_seq_length = self._max_seq_length
        elapsed = time.monotonic() - t0
        logger.info("Model loaded in %.1fs (max_seq_length=%d)", elapsed, self._max_seq_length)

        if self._target_dimension:
            self._dim = self._target_dimension
        else:
            dim = model.get_sentence_embedding_dimension()
            self._dim = int(dim) if dim else 896

        logger.info("Embedding dimension: %d (Matryoshka truncation)", self._dim)
        self._model = model
        return model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._ensure_model()

        logger.info(
            "Embedding %d texts with %s (batch_size=%d)",
            len(texts), self._model_name, self._batch_size,
        )
        t0 = time.monotonic()

        encode_kwargs: dict[str, object] = {
            "batch_size": self._batch_size,
            "show_progress_bar": len(texts) > 100,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if self._target_dimension:
            encode_kwargs["truncate_dim"] = self._target_dimension
        if self._is_jina_code:
            encode_kwargs["prompt_name"] = "nl2code_document"

        embeddings = model.encode(texts, **encode_kwargs)  # type: ignore[union-attr]

        elapsed = time.monotonic() - t0
        rate = len(texts) / max(elapsed, 0.001)
        logger.info(
            "Embedded %d texts in %.1fs (%.0f texts/s, dim=%d)",
            len(texts), elapsed, rate, self._dim,
        )
        return embeddings.tolist()  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()

        encode_kwargs: dict[str, object] = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if self._target_dimension:
            encode_kwargs["truncate_dim"] = self._target_dimension
        if self._is_jina_code:
            encode_kwargs["prompt_name"] = "nl2code_query"

        vec = model.encode([text], **encode_kwargs)  # type: ignore[union-attr]
        return vec[0].tolist()  # type: ignore[union-attr]


class NomicEmbedCodeEmbedder:
    def __init__(
        self,
        *,
        model: str = "nomic-embed-code-v1",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self._model: object | None = None
        self._dim: int = 768

    @property
    def dimension(self) -> int:
        return self._dim

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from nomic import embed as nomic_embed  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'nomic' package is required for local Nomic embeddings. "
                "Install the project with the 'local-models' extra."
            ) from exc
        self._model = nomic_embed
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        nomic_embed = self._ensure_model()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = nomic_embed.text(  # type: ignore[union-attr]
                texts=batch,
                model=self._model_name,
                task_type="search_document",
            )
            all_embeddings.extend(result["embeddings"])

        if all_embeddings:
            self._dim = len(all_embeddings[0])

        logger.info(
            "Nomic embed_documents: %d texts → %d vectors (dim=%d)",
            len(texts),
            len(all_embeddings),
            self._dim,
        )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        nomic_embed = self._ensure_model()
        result = nomic_embed.text(  # type: ignore[union-attr]
            texts=[text],
            model=self._model_name,
            task_type="search_query",
        )
        vector = result["embeddings"][0]
        self._dim = len(vector)
        return vector


def create_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=config.name,
            device=config.device,
            batch_size=config.batch_size,
            trust_remote_code=config.trust_remote_code,
            target_dimension=config.dimension,
            max_seq_length=config.max_seq_length,
        )

    if config.provider == "voyageai":
        api_key = config.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Voyage API key not found. Set the {config.api_key_env} environment variable."
            )
        return VoyageCodeEmbedder(
            api_key=api_key,
            model=config.name,
            batch_size=config.batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
        )

    if config.provider == "nomic-local":
        return NomicEmbedCodeEmbedder(model=config.name)

    raise ValueError(f"Unknown embedding provider: {config.provider!r}")


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _token_aware_batches(
    texts: list[str],
    max_batch_size: int,
    max_tokens: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = _estimate_tokens(text)
        if current_batch and (
            len(current_batch) >= max_batch_size or current_tokens + text_tokens > max_tokens
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def entity_text_for_embedding(
    source_text: str,
    signature_summary: str | None,
    estimated_tokens: int,
    *,
    token_threshold: int = 512,
    file_path: str | None = None,
    chunk_large: bool = True,
) -> str | list[str]:
    prefix = f"# file: {file_path}\n" if file_path else ""

    if estimated_tokens <= token_threshold:
        return prefix + source_text

    if chunk_large and estimated_tokens > token_threshold:
        chunks = _chunk_source(source_text, token_threshold, signature_summary)
        return [prefix + chunk for chunk in chunks]

    if signature_summary:
        return prefix + signature_summary
    return prefix + source_text


def _chunk_source(
    source_text: str,
    token_threshold: int,
    signature_summary: str | None,
) -> list[str]:
    sig_prefix = (signature_summary or "").split("\n")[0] + "\n# ...\n"
    sig_tokens = max(1, len(sig_prefix.encode("utf-8")) // 4)
    chunk_budget = max(100, token_threshold - sig_tokens)

    lines = source_text.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = max(1, len(line.encode("utf-8")) // 4)
        if current_tokens + line_tokens > chunk_budget and current_lines:
            chunk_text = "\n".join(current_lines)
            if chunks:
                chunks.append(sig_prefix + chunk_text)
            else:
                chunks.append(chunk_text)
            overlap = current_lines[-3:] if len(current_lines) > 3 else []
            current_lines = list(overlap)
            current_tokens = sum(max(1, len(l.encode("utf-8")) // 4) for l in current_lines)
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk_text = "\n".join(current_lines)
        if chunks:
            chunks.append(sig_prefix + chunk_text)
        else:
            chunks.append(chunk_text)

    return chunks if chunks else [source_text]
