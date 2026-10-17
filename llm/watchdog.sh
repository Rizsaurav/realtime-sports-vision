#!/usr/bin/env bash
# Keep the two model replicas alive. Run on the GPU box as the project user.
#   setsid nohup llm/watchdog.sh > /data/saurav/logs/watchdog.log 2>&1 &
# Every minute: if a replica's process (from its pid file) is gone, start it again with the same settings.
# A replica that is still loading (process alive, port not answering yet) is left alone.
# Stop it with: kill $(cat /data/saurav/logs/watchdog.pid)
set -u
cd "$(dirname "$0")/.."
echo $$ > /data/saurav/logs/watchdog.pid
REPLICAS=("0 8000 /data/saurav/logs/qwen38-app.pid /data/saurav/logs/qwen38-a.log"
          "1 8001 /data/saurav/logs/qwen38-b.pid /data/saurav/logs/qwen38-b.log")
while true; do
  for spec in "${REPLICAS[@]}"; do
    set -- $spec
    gpu=$1; port=$2; pidfile=$3; log=$4
    pid=$(cat "$pidfile" 2>/dev/null || echo 0)
    if ! kill -0 "$pid" 2>/dev/null || ! grep -q vllm "/proc/$pid/cmdline" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) replica on GPU $gpu (port $port) is not running; restarting"
      setsid nohup env GPU=$gpu PORT=$port PIDFILE=$pidfile MAX_LEN=262144 MEM_UTIL=0.62 MAX_SEQS=8 \
        llm/serve_qwen38.sh >> "$log" 2>&1 < /dev/null &
      sleep 5
    fi
  done
  sleep 60
done
