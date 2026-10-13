#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KUBECONFIG_PATH="${KUBECONFIG:-kubelet.yaml}"
RERANKER_PVC="${RERANKER_PVC:-reranker-20260518221350-workspace}"
VGIB_PVC="${VGIB_PVC:-vgib-20260519-workspace}"
CACHE_PVC="${CACHE_PVC:-token-sniper-shared-cache}"
STAGGER="${STAGGER:-120}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
GIT_REF="${GIT_REF:-feature/ablation-study}"
NODE_NAME="${NODE_NAME:-k8s-worker-cpu-node-r770}"
MAX_HARNESS_WORKERS="${MAX_HARNESS_WORKERS:-4}"
ORACLE_TOP_K="${ORACLE_TOP_K:-10}"
DRY_RUN="${DRY_RUN:-}"
RESUME="${RESUME:-}"
WORKSPACE_PVC="${WORKSPACE_PVC:-}"
RUN_ID="${RUN_ID:-}"
SEQUENTIAL="${SEQUENTIAL:-}"
SUBMIT_TIMEOUT="${SUBMIT_TIMEOUT:-24h}"

NAMESPACE="${NAMESPACE:-$(kubectl --kubeconfig "$KUBECONFIG_PATH" config view --minify -o jsonpath='{.contexts[0].context.namespace}' 2>/dev/null || true)}"
NAMESPACE="${NAMESPACE:-runai-romania-dev}"

ALL_CONDITIONS=(
    dense_knn
    dense_reranker
    dense_reranker_vgib
    dense_reranker_vgib_rl
)

if [ $# -eq 0 ]; then
    CONDITIONS=("${ALL_CONDITIONS[@]}")
else
    CONDITIONS=("$@")
fi

echo "==> Plan"
echo "  kubeconfig:    $KUBECONFIG_PATH"
echo "  reranker_pvc:  $RERANKER_PVC"
echo "  vgib_pvc:      $VGIB_PVC"
echo "  cache_pvc:     $CACHE_PVC"
echo "  git_ref:       $GIT_REF"
echo "  node_name:     $NODE_NAME"
echo "  agent_workers: $AGENT_WORKERS"
echo "  harness_workers: $MAX_HARNESS_WORKERS"
echo "  stagger:       ${STAGGER}s between launches"
echo "  resume:        ${RESUME:-0}"
echo "  sequential:    ${SEQUENTIAL:-0}"
echo "  conditions:    ${CONDITIONS[*]}"
echo

if [ -z "${GITHUB_TOKEN:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
    if GITHUB_TOKEN=$(kubectl --kubeconfig "$KUBECONFIG_PATH" get secret token-sniper-github \
        -n "$NAMESPACE" -o jsonpath='{.data.token}' 2>/dev/null | base64 -d); then
        export GITHUB_TOKEN
        echo "==> Loaded GITHUB_TOKEN from secret token-sniper-github"
    else
        echo "WARNING: GITHUB_TOKEN not set and could not load from cluster secret."
    fi
    echo
fi

echo "==> Checking PVCs exist..."
for pvc in "$RERANKER_PVC" "$VGIB_PVC" "$CACHE_PVC"; do
    if ! kubectl --kubeconfig "$KUBECONFIG_PATH" get pvc "$pvc" >/dev/null 2>&1; then
        echo "ERROR: PVC '$pvc' not found"
        exit 1
    fi
    echo "    $pvc: OK"
done

echo
echo "==> Checking qwen-vllm is Running..."
vllm_phase=$(kubectl --kubeconfig "$KUBECONFIG_PATH" get pod -l app=qwen-vllm -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo missing)
if [ "$vllm_phase" != "Running" ]; then
    echo "WARNING: qwen-vllm phase is '$vllm_phase' (expected Running). Submitting anyway."
else
    echo "    qwen-vllm: Running"
fi

submit_one() {
    local cond=$1
    echo
    echo "==> Submitting $cond"
    local -a cmd=(
        python scripts/submit_ablation_swebench_k8s.py
        --kubeconfig "$KUBECONFIG_PATH"
        --condition "$cond"
        --git-ref "$GIT_REF"
        --node-name "$NODE_NAME"
        --reranker-pvc "$RERANKER_PVC"
        --cache-pvc "$CACHE_PVC"
        --oracle-top-k "$ORACLE_TOP_K"
        --agent-workers "$AGENT_WORKERS"
        --max-harness-workers "$MAX_HARNESS_WORKERS"
    )
    if [ "$cond" = "dense_reranker_vgib" ] || [ "$cond" = "dense_reranker_vgib_rl" ]; then
        cmd+=(--vgib-pvc "$VGIB_PVC")
    fi
    local ws_pvc
    ws_pvc="$(workspace_pvc_for "$cond")"
    if [ -n "$ws_pvc" ]; then
        cmd+=(--workspace-pvc "$ws_pvc")
    fi
    if [ -n "$RESUME" ]; then
        cmd+=(--resume)
    fi
    if [ -n "$RUN_ID" ]; then
        cmd+=(--run-id "$RUN_ID")
    fi
    echo "    ${cmd[*]}"
    if [ -z "$DRY_RUN" ]; then
        "${cmd[@]}"
    fi
}

workspace_pvc_for() {
    local cond=$1
    if [ -n "$WORKSPACE_PVC" ]; then
        echo "$WORKSPACE_PVC"
        return
    fi
    local suffix
    suffix="$(echo "$cond" | tr '[:lower:]' '[:upper:]')"
    local var="WORKSPACE_PVC_${suffix}"
    echo "${!var:-}"
}

wait_for_latest_job() {
    local cond=$1
    local component="ablation-${cond//_/-}"
    local job
    job="$(kubectl --kubeconfig "$KUBECONFIG_PATH" get jobs -n "$NAMESPACE" \
        -l "app.kubernetes.io/component=$component" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)"
    if [ -z "$job" ]; then
        echo "WARNING: could not find latest job for $cond to wait on"
        return 0
    fi
    echo "==> Waiting for job/$job to complete (timeout=$SUBMIT_TIMEOUT)"
    kubectl --kubeconfig "$KUBECONFIG_PATH" wait \
        --for=condition=complete \
        --timeout="$SUBMIT_TIMEOUT" \
        -n "$NAMESPACE" "job/$job"
}

last_idx=$(( ${#CONDITIONS[@]} - 1 ))
for i in "${!CONDITIONS[@]}"; do
    submit_one "${CONDITIONS[$i]}"
    if [ -z "$DRY_RUN" ] && [ -n "$SEQUENTIAL" ]; then
        wait_for_latest_job "${CONDITIONS[$i]}"
    elif [ "$i" -lt "$last_idx" ] && [ -z "$DRY_RUN" ]; then
        echo "    sleeping ${STAGGER}s to stagger vllm load..."
        sleep "$STAGGER"
    fi
done

if [ -z "$DRY_RUN" ]; then
    echo
    echo "==> All arms submitted. Monitoring commands:"
    echo
    echo "  # List ablation pods"
    echo "  kubectl --kubeconfig=$KUBECONFIG_PATH get pods | grep ablation-"
    echo
    echo "  # Tail one arm's runner log"
    echo "  kubectl --kubeconfig=$KUBECONFIG_PATH logs -l app.kubernetes.io/component=ablation-dense_reranker_vgib -c runner -f"
    echo
    echo "  # Check qwen-vllm health"
    echo "  kubectl --kubeconfig=$KUBECONFIG_PATH get pod qwen-vllm-... -o wide"
    echo
    echo "Each arm produces 50 patches; expected wallclock ~2-3h per arm in parallel."
    echo "Results land in: /workspace/repo/experiments/ablation/<condition>/<run-id>/"
fi
