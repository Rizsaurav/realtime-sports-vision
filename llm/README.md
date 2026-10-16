# Qwen3.8-27B on one H100: fast single-stream serving

`serve_qwen38.sh` starts vLLM with the FP8 checkpoint, CUDA graphs and the model's built-in
MTP speculative decoder. It binds to 127.0.0.1 and speaks the chat-completions HTTP API.

```bash
llm/serve_qwen38.sh                      # about 3 minutes cold (kernel compile is cached afterwards)
ssh -N -L 8000:127.0.0.1:8000 <host>     # from a laptop, then use http://localhost:8000/v1
llm/stop_qwen38.sh                       # stops exactly the server that was started
```

## Measured (H100 NVL, one stream, temperature 0.6, `llm/bench_decode.py`)

| Setting | Decode tok/s (mean) | Time to first token | MTP acceptance |
|---|---|---|---|
| Earlier attempt (`--enforce-eager`, no speculation) | under 25 | | |
| This setup, instruct mode | 202 (median 206, range 174 to 218) | 126 ms | 80% |
| This setup, thinking mode | 167 (range 144 to 193) | 160 ms | 59% |

## What changed from the slow setup

1. **CUDA graphs and torch.compile on.** `--enforce-eager` removes both, which is the largest cost
   for a 27B decode where each step is short.
2. **MTP speculative decoding, 3 draft tokens.** The checkpoint contains its own MTP head. Draft
   tokens are verified against the full model with rejection sampling, so the sampling
   distribution is preserved. Output is not bit-identical to plain greedy decoding.
3. **No JIT builds at start-up.** The FlashInfer sampler and GDN prefill kernels compile with nvcc at
   first use, and the env's torch is CUDA 13 while only the CUDA 12.9 toolkit is installed, which is
   what crashed the earlier launch. `VLLM_USE_FLASHINFER_SAMPLER=0` and `--gdn-prefill-backend triton`
   avoid it.

Kept for quality: bf16 KV cache (only 16 of 64 layers use full attention, about 65 KB per token), no
weight quantisation beyond the released FP8. The benchmarks here sampled at temperature 0.6, top-p 0.95, top-k 20,
which is not what the model card recommends (temperature 1.0 with thinking on, 0.7 with thinking off).

## Memory

`MEM_UTIL=0.45` takes about 44 GB of the 94 GB card (weights 29 GB, KV for 131K tokens, graphs).
Raise it for more concurrent long contexts; other GPU jobs that need most of the card cannot share it.

## Checks

`quality_check.py` runs 10 small problems with hidden unit tests, 4 samples each: 40 of 40 passed
with MTP on. The problems are easy, so this shows the server is healthy, not that MTP equals plain
decoding. A harder comparison (MTP on versus `MTP=0`) is the next step.
