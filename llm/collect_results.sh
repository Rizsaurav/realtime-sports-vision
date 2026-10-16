#!/usr/bin/env bash
# Copy the small raw logs of the experiments from the GPU box into llm/results/.
# Server logs, agent trajectories and container logs stay on the box under /data/saurav.
set -euo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-scc}"
mkdir -p results/sweep results/swe results/concurrency
ssh "$HOST" "sudo -iu saurav bash -c 'cd /data/saurav/logs/sweep && tar cf - --exclude=server.log --exclude=server.pid .'" | tar xf - -C results/sweep 2>/dev/null || true
for run in ${RUNS:-}; do
  mkdir -p "results/swe/$run"
  ssh "$HOST" "sudo -iu saurav bash -c 'cd /data/saurav/results/swe/$run && tar cf - run.jsonl summary.json *.yaml mtp3/*.patch nospec/*.patch 2>/dev/null'" | tar xf - -C "results/swe/$run" 2>/dev/null || true
done
ssh "$HOST" "sudo -iu saurav bash -c 'cd /data/saurav/logs/concurrency && tar cf - .'" | tar xf - -C results/concurrency 2>/dev/null || true
find results -name '._*' -delete
echo "collected into $(pwd)/results"
