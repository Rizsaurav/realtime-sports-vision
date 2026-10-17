#!/usr/bin/env bash
# Start the supervisor so it survives this terminal, and keep the Mac awake for a fixed time.
#   llm/orchestrator/run_night.sh llm/orchestrator/plan_webapp.json [hours]      (default 7 hours)
# The wake-lock is a separate process with its own fixed length, so it does not depend on the supervisor.
# Keep the Mac PLUGGED IN and the lid OPEN (a closed lid sleeps the Mac unless it is on power with an external display).
# Stop the supervisor with: touch <root>/STOP   (it finishes the current attempts first).
set -euo pipefail
PLAN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
HOURS="${2:-7}"
ROOT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['root'])" "$PLAN")
mkdir -p "$ROOT"
rm -f "$ROOT/STOP"
cd "$(dirname "$0")"
# -d display, -i idle, -m disk, -s system (on power), -u user activity; -t seconds
nohup caffeinate -dimsu -t $((HOURS * 3600)) > /dev/null 2>&1 &
echo $! > "$ROOT/caffeinate.pid"
nohup python3 orchestrate.py "$PLAN" > "$ROOT/supervisor.log" 2>&1 &
echo $! > "$ROOT/supervisor.pid"
echo "wake-lock pid $(cat "$ROOT/caffeinate.pid") for ${HOURS}h; supervisor pid $(cat "$ROOT/supervisor.pid")"
echo "log: $ROOT/supervisor.log   status: $ROOT/status.json"
