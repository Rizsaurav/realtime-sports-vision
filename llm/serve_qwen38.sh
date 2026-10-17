#!/usr/bin/env bash
# Launch Qwen3.8-27B (FP8) on one H100 with vLLM, tuned for single-stream decode speed.
#
#   llm/serve_qwen38.sh            start in the foreground
#   MTP=0 llm/serve_qwen38.sh      baseline without speculative decoding
#   MTP_TOKENS=2 PORT=8001 ...     override any setting below
#
# What matters, in order of effect on decode speed:
#   1. CUDA graphs and torch.compile ON (no --enforce-eager): removes per-step launch overhead
#   2. MTP speculative decoding with the model's own built-in head: lossless, several tokens per step
#   3. FP8 weights: half the bytes read per token compared with bf16
# Quality: KV cache stays bf16 (only 16 of 64 layers use full attention, so it is small), and
# speculative decoding keeps the sampling distribution, so there is no quantised-KV tradeoff.
set -euo pipefail

MODEL="${MODEL:-/data/saurav/models/Qwen3.8-27B-FP8}"
VENV="${VENV:-/data/saurav/envs/qwen38}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
MAX_LEN="${MAX_LEN:-131072}"
MEM_UTIL="${MEM_UTIL:-0.55}"        # about 52 GB of 94 GB: weights 29 GB plus KV for several long agent sessions
MAX_SEQS="${MAX_SEQS:-8}"
MTP="${MTP:-1}"
MTP_TOKENS="${MTP_TOKENS:-3}"
EAGER="${EAGER:-0}"                  # 1 = --enforce-eager (no CUDA graphs, no compile): the slow old setup

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_USE_FLASHINFER_SAMPLER=0      # the FlashInfer sampler JIT needs nvcc 13; the env has only 12.9
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

SPEC=()
if [ "$MTP" = "1" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

PIDFILE="${PIDFILE:-/data/saurav/logs/qwen38-app.pid}"
echo $$ > "$PIDFILE"      # exec keeps this pid, so llm/stop_qwen38.sh can stop exactly this server

EXTRA=()
[ "$EAGER" = "1" ] && EXTRA+=(--enforce-eager)

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --gdn-prefill-backend triton \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "${EXTRA[@]}" \
  "${SPEC[@]}"
