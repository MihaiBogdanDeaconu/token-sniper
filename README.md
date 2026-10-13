# Token Sniper

Graph-aware context retrieval for repository-level code agents.

Code agents spend most of their token budget navigating a repository with `grep`, `find` and `cat` before they write a single line of a patch. Token Sniper replaces that search with learned retrieval over a typed code graph: it finds the entities relevant to an issue, ranks them with structural signals, and compresses them to the smallest subgraph that still contains what the agent needs.

## Results

SWE-bench Verified, 50-task subset. Every arm uses the same agent and evaluation harness; only the retrieval stack changes.

| Retrieval stack | Pass@1 | Resolved | Median session tokens | Median wall time |
|---|---|---|---|---|
| Dense kNN | 42% | 21/50 | 3.54M | 202 s |
| + Graph-feature reranker | 54% | 27/50 | 3.69M | 229 s |
| + VGIB compression | 54% | 27/50 | 3.07M | 186 s |
| + GRPO fine-tuning | **60%** | **30/50** | **2.67M** | 197 s |

The full stack resolves 18 points more tasks than dense retrieval while consuming 25% fewer tokens per session.

Component-level results:

| Component | Result |
|---|---|
| Upstream retrieval | ~97% gold-file recall over 497 SWE-bench tasks |
| Edge predictor | 0.94 mean AUC-ROC on 12 held-out tasks |
| Graph reranker | +11.4 pp MRR over dense kNN retrieval |
| VGIB selector | 100% validation gold-file recall at 8.8% token retention |
| GRPO fine-tuning | Token retention 10.0% to 9.0% at 99.6% recall, on 77 tasks from unseen repositories |

## Method

The pipeline has five stages. Each stage was gated by a kill criterion fixed before its experiment ran.

**1. Structural code graph.** Every Python file is parsed with Tree-sitter into a static structural graph whose nodes are files, classes, functions, methods and module preambles, connected by `calls`, `imports`, `inherits` and `contains` edges. Node identifiers are fully qualified names, so a graph serializes deterministically for a given repository and commit.

**2. Learned edge completion.** Static parsing misses calls resolved at runtime through dynamic dispatch, decorators and `getattr`. A 3-layer GraphSAGE encoder with JumpingKnowledge, trained with focal loss on call edges recorded by executing each repository's test suite, predicts the missing edges and augments the graph.

**3. Hybrid retrieval and graph reranking.** Candidates come from dense kNN and BM25, merged by reciprocal rank fusion and expanded along graph edges. An XGBoost pairwise ranker then scores each candidate on structural features, including personalized PageRank seeded on the issue, call distance, degree, file PageRank and issue mentions.

**4. Variational Graph Information Bottleneck.** A GraphSAGE selector predicts a retention probability for every candidate entity. Retention masks are relaxed with a concrete (Gumbel-Softmax) distribution, and the subgraph is pooled by cross-attention from the issue embedding. Training minimizes negative soft file recall plus a token-ratio penalty and a variational upper bound on the mutual information between the kept subgraph and the input graph, so the model learns the minimum sufficient subgraph, the smallest context that still covers the files the fix touches.

**5. GRPO fine-tuning.** The pretrained selector is treated as a policy over Bernoulli retention masks and refined with Group Relative Policy Optimization: eight sampled subgraphs per task, group-normalized advantages, a clipped surrogate objective, an entropy bonus and an adaptive KL penalty against the frozen supervised policy. Two rewards are implemented. The proxy reward combines gold-node recall, a token-ratio penalty and trace-edge connectivity, and costs nothing to compute. The verified reward runs the agent inside the task's SWE-bench container on exactly the context a rollout selects, applies the patch, and scores 1 only if every failing test passes. Train and validation tasks are split by repository to prevent leakage.

A separate guardrail module trains a transformer classifier on the agent's top-k log-probability sequence to flag likely hallucinations without access to model internals.

Evaluation runs against pinned task sets, SWE-bench Lite (274), SWE-bench Verified (500) and Loc-Bench (560), checked to share no instance and no repository with the reranker and selector training splits. Entity-level localization is scored with the LocAgent evaluation code, vendored under `third_party/` at a pinned commit.

## Repository layout

```
token_sniper/
  parse/            Tree-sitter entity extraction
  graph/            structural graph construction, static call edges, metrics
  traces/           runtime call tracing and graph alignment
  edge_pred/        GraphSAGE edge prediction
  retrieval/        embeddings, vector store, graph retrievers, reranker
  vgib_selector/    VGIB backbone, selector, losses and training
  rl/               GRPO policy, rewards, rollouts and trainer
  guardrails/       log-probability hallucination classifier
  agent/            shell and Docker tool agent runner
  evaluation/       SWE-bench harness, evaluation arms, metrics, statistics, Kubernetes jobs
scripts/            entry points for every stage, benchmark and cluster job
configs/            base configuration, per-stage configs, task splits and evaluation pins
tests/              unit and integration tests
third_party/        LocAgent scoring code, Apache 2.0, pinned commit
```

## Installation

Python 3.11 or newer.

```bash
pip install -e ".[dev]"
pip install -e ".[dev,research]"
```

The first command installs the graph and parsing core with test tooling. The second adds PyTorch, PyTorch Geometric, the retrieval stack and experiment tracking. Dense retrieval with Voyage Code 3 reads `VOYAGE_API_KEY` from the environment; the agent connects to any OpenAI-compatible endpoint configured in `configs/base.yaml`.

## Usage

Build a structural graph for a local repository:

```bash
python scripts/build_ssg_repo.py --repo path/to/project --output graph.json --metrics
```

Score static call edges against a gold list:

```bash
python scripts/evaluate_ssg_calls.py --repo tests/fixtures/minimal_pkg \
  --gold tests/fixtures/minimal_pkg/calls_gold.json --metrics-only
```

Run the agent on an issue, with or without oracle context:

```bash
python scripts/run_agent_task.py --repo path/to/project --issue-file issue.txt --mode blind
python scripts/run_agent_task.py --repo path/to/project --issue-file issue.txt --mode oracle \
  --oracle-file package/module.py
```

Train each stage:

```bash
python scripts/phase2_train_edge_predictor.py --help
python scripts/train_reranker.py --help
python scripts/train_vgib.py --help
python scripts/train_rl.py --help
```

Evaluate a retrieval stack end to end on SWE-bench:

```bash
python scripts/eval_ablation_swebench.py --help
```

Every script under `scripts/submit_*_k8s.py` submits the corresponding stage as a Kubernetes job.

## Development

```bash
make lint
make typecheck
make test
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
