#!/usr/bin/env bash
# Run the Pi coding agent against the Qwen3.8-27B server on the GPU box.
#
#   cd ~/some/project && /path/to/llm/pi/pi-qwen.sh            interactive
#   llm/pi/pi-qwen.sh -p "explain this repo"                   one shot (print mode)
#   llm/pi/pi-qwen.sh --thinking off ...                       faster, no reasoning
#
# Pi runs the model's shell commands WITHOUT asking for approval, on whatever machine it runs on.
# Use it inside a project folder under git, never in your home directory, and keep secrets out of
# the environment (PI_AGENT_HOME below gives it a separate HOME for config and sessions only).
set -euo pipefail

HOST="${HOST:-scc}"
PORT="${PORT:-8000}"
PI_BIN="${PI_BIN:-$HOME/.pi-install/node_modules/.bin/pi}"
NODE_DIR="$(ls -d "$HOME"/.nvm/versions/node/v22.* 2>/dev/null | sort -V | tail -1)"
[ -n "$NODE_DIR" ] || { echo "Node 22 is required (nvm install 22)"; exit 1; }
[ -x "$PI_BIN" ] || { echo "Pi is not installed at $PI_BIN"; exit 1; }

# make sure the model server is reachable through an SSH port forward (idempotent)
if ! curl -s -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  ssh -O forward -L "$PORT:127.0.0.1:$PORT" "$HOST" 2>/dev/null \
    || ssh -f -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$HOST"
  sleep 1
  curl -s -m 5 "http://localhost:$PORT/v1/models" >/dev/null || { echo "model server not reachable on localhost:$PORT"; exit 1; }
fi

[ -n "${PI_AGENT_HOME:-}" ] && export HOME="$PI_AGENT_HOME"
exec env PATH="$NODE_DIR/bin:$PATH" "$PI_BIN" --provider qwen-box --model qwen3.8-27b "$@"
