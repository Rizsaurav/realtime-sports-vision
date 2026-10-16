#!/usr/bin/env bash
# Stop the server started by serve_qwen38.sh, by its recorded pid only.
set -euo pipefail
PIDFILE="${PIDFILE:-/data/saurav/logs/qwen38-app.pid}"
[ -f "$PIDFILE" ] || { echo "no pidfile at $PIDFILE"; exit 1; }
PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null && grep -q "vllm" "/proc/$PID/cmdline" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  echo "stopped $PID"
else
  echo "pid $PID is not a running vllm server"
fi
rm -f "$PIDFILE"
