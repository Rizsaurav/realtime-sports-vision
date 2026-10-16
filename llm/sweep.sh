#!/usr/bin/env bash
# Inference-technique sweep: start the server in each configuration, run the decode benchmark
# (instruct and thinking mode) and the coding check, keep every raw log, stop exactly that server.
#
#   GPU=1 PORT=8001 llm/sweep.sh            # uses the second GPU so a running server is untouched
#
# Logs: /data/saurav/logs/sweep/<config>/{server.log,bench_instruct.txt,bench_thinking.txt,quality.txt,quality.json}
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-1}"; PORT="${PORT:-8001}"
OUT="${OUT:-/data/saurav/logs/sweep}"
PIDFILE="$OUT/server.pid"
mkdir -p "$OUT"

# name  EAGER  MTP  MTP_TOKENS
CONFIGS=(
  "1_eager_nospec 1 0 0"
  "2_graphs_nospec 0 0 0"
  "3_graphs_mtp1 0 1 1"
  "4_graphs_mtp2 0 1 2"
  "5_graphs_mtp3 0 1 3"
  "6_graphs_mtp4 0 1 4"
  "7_graphs_mtp5 0 1 5"
)

for c in "${CONFIGS[@]}"; do
  read -r name eager mtp ntok <<<"$c"
  d="$OUT/$name"; mkdir -p "$d"
  echo "=== $name  eager=$eager mtp=$mtp tokens=$ntok  $(date -u +%FT%TZ)" | tee -a "$OUT/sweep.log"
  GPU="$GPU" PORT="$PORT" EAGER="$eager" MTP="$mtp" MTP_TOKENS="${ntok:-3}" PIDFILE="$PIDFILE" \
    setsid nohup llm/serve_qwen38.sh > "$d/server.log" 2>&1 < /dev/null &
  ok=0
  for _ in $(seq 1 120); do
    sleep 5
    curl -s -m 3 "localhost:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$(cat "$PIDFILE" 2>/dev/null || echo 0)" 2>/dev/null || break
  done
  if [ "$ok" != 1 ]; then echo "  FAILED to start (see $d/server.log)" | tee -a "$OUT/sweep.log"; PIDFILE="$PIDFILE" llm/stop_qwen38.sh >/dev/null 2>&1; continue; fi
  python3 llm/bench_decode.py --port "$PORT" --max-tokens 1024 --rounds 2 > "$d/bench_instruct.txt" 2>&1
  python3 llm/bench_decode.py --port "$PORT" --max-tokens 2048 --thinking --warmup 0 > "$d/bench_thinking.txt" 2>&1
  python3 llm/quality_check.py --port "$PORT" --samples 4 --out "$d/quality.json" > "$d/quality.txt" 2>&1
  tail -2 "$d/bench_instruct.txt" | tee -a "$OUT/sweep.log"
  tail -1 "$d/quality.txt" | tee -a "$OUT/sweep.log"
  PIDFILE="$PIDFILE" llm/stop_qwen38.sh | tee -a "$OUT/sweep.log"
  sleep 5
done
echo "=== sweep done $(date -u +%FT%TZ)" | tee -a "$OUT/sweep.log"
