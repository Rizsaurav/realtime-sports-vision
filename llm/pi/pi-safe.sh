#!/usr/bin/env bash
# Pi in a macOS sandbox: it can write only inside the project folder, and talk only to the model servers and the web.
#   llm/pi/pi-safe.sh [project-dir] [pi options]       interactive screen
#   llm/pi/pi-safe.sh . -p "explain this repo"         one shot
# The project folder is the first argument if it is a directory, otherwise the current folder.
# Pi's own settings, sessions and Python tools live in ~/.pi-agent-home (sessions are saved there, so /resume works).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -gt 0 ] && [ -d "$1" ]; then PROJECT="$(cd "$1" && pwd)"; shift; else PROJECT="$(pwd)"; fi
H="${PI_SAFE_HOME:-$HOME/.pi-agent-home}"
PI_DIR="$HOME/.pi-install"; PI_BIN="$PI_DIR/node_modules/.bin/pi"
NODE_DIR="$(ls -d "$HOME"/.nvm/versions/node/v22.* 2>/dev/null | sort -V | tail -1)"
PORT="${PORT:-8001}"; HOST="${HOST:-scc}"
[ -x "$PI_BIN" ] && [ -n "$NODE_DIR" ] || { echo "Pi or Node 22 is missing"; exit 1; }
mkdir -p "$H/tmp"
# model server through the SSH forward (idempotent)
if ! curl -s -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  ssh -O forward -L "$PORT:127.0.0.1:$PORT" "$HOST" 2>/dev/null || ssh -f -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$HOST"
  sleep 1
fi
curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null || { echo "model server not reachable on localhost:$PORT"; exit 1; }
NAME="$(git -C "$PROJECT" config user.name 2>/dev/null || true)"; MAIL="$(git -C "$PROJECT" config user.email 2>/dev/null || true)"
exec sandbox-exec -f "$HERE/../orchestrator/agent_tty.sb" -D "WORK=$PROJECT" -D "AGENT_HOME=$H" -D "AGENT_TMP=$H/tmp" \
  -D "NODE_DIR=$NODE_DIR" -D "PI_DIR=$PI_DIR" \
  env HOME="$H" TMPDIR="$H/tmp" LANG=en_US.UTF-8 TERM="${TERM:-xterm-256color}" VIRTUAL_ENV="$H/venv" \
      PATH="$H/venv/bin:$NODE_DIR/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin" \
      GIT_AUTHOR_NAME="$NAME" GIT_AUTHOR_EMAIL="$MAIL" GIT_COMMITTER_NAME="$NAME" GIT_COMMITTER_EMAIL="$MAIL" \
  "$PI_BIN" --provider qwen-box --model qwen3.8-27b --thinking medium "$@"
