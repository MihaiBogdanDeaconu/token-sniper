#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KCFG="${KUBECONFIG:-kubelet.yaml}"
NS="${NAMESPACE:-runai-romania-dev}"
NODE="${NODE_NAME:-k8s-worker-cpu-node-r770}"

CONDITIONS=(
  "dense_knn:ablation-dense-knn-20260520160014-workspace"
  "dense_reranker:ablation-dense-reranker-20260520160218-workspace"
  "dense_reranker_vgib:ablation-dense-reranker-vgib-20260520160422-workspace"
)

echo "==> Refresh analyze script ConfigMap"
kubectl --kubeconfig "$KCFG" create configmap ablation-analyze-script -n "$NS" \
  --from-file=analyze_ablation_runs.py=scripts/analyze_ablation_runs.py \
  --dry-run=client -o yaml | kubectl --kubeconfig "$KCFG" apply -f -

analyze_one() {
  local cond=$1
  local pvc=$2
  local job="ablation-analyze-${cond//_/-}"

  kubectl --kubeconfig "$KCFG" delete job -n "$NS" "$job" --ignore-not-found=true --wait=true 2>/dev/null || true

  kubectl --kubeconfig "$KCFG" apply -n "$NS" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job}
  namespace: ${NS}
  labels:
    runai/node-pool: default
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        runai/node-pool: default
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: ${NODE}
      containers:
        - name: analyze
          image: python:3.12-slim
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: "1"
              memory: 1Gi
          command:
            - /bin/bash
            - -lc
            - |
              set -euo pipefail
              COND=${cond}
              SRC=/pvc/repo/experiments/ablation/\$COND
              test -f "\$SRC/summary.json"
              mkdir -p /tmp/ablation
              ln -sfn "\$SRC" /tmp/ablation/\$COND
              python3 /scripts/analyze_ablation_runs.py \\
                --base-dir /tmp/ablation \\
                --conditions "\$COND" \\
                --json-out /tmp/report.json
              echo "===== \${COND} ====="
              cat /tmp/report.json
          volumeMounts:
            - name: script
              mountPath: /scripts
            - name: data
              mountPath: /pvc
              readOnly: true
      volumes:
        - name: script
          configMap:
            name: ablation-analyze-script
            defaultMode: 0755
        - name: data
          persistentVolumeClaim:
            claimName: ${pvc}
            readOnly: true
EOF

  echo "==> Waiting for job/${job}"
  kubectl --kubeconfig "$KCFG" wait -n "$NS" --for=condition=complete "job/${job}" --timeout=30m
  kubectl --kubeconfig "$KCFG" logs -n "$NS" "job/${job}"
}

for entry in "${CONDITIONS[@]}"; do
  cond="${entry%%:*}"
  pvc="${entry##*:}"
  analyze_one "$cond" "$pvc"
done

VGIB_RL_POD=$(kubectl --kubeconfig "$KCFG" get pod -n "$NS" \
  -l app.kubernetes.io/component=ablation-dense_reranker_vgib_rl \
  -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null | head -1)
if [ -n "${VGIB_RL_POD:-}" ]; then
  echo "==> dense_reranker_vgib_rl via exec on ${VGIB_RL_POD}"
  kubectl --kubeconfig "$KCFG" cp "$ROOT/scripts/analyze_ablation_runs.py" \
    "${NS}/${VGIB_RL_POD}:/tmp/analyze_ablation_runs.py" -c runner
  kubectl --kubeconfig "$KCFG" exec -n "$NS" "${VGIB_RL_POD}" -c runner -- \
    python3 /tmp/analyze_ablation_runs.py \
      --base-dir /workspace/repo/experiments/ablation \
      --conditions dense_reranker_vgib_rl \
      --json-out /tmp/ablation_report_vgib_rl.json
else
  analyze_one dense_reranker_vgib_rl ablation-dense-reranker-vgib-rl-20260520160627-workspace
fi

echo "==> Done"
