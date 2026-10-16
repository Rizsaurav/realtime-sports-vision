# Serving Qwen3.8-27B fast on one H100: what each inference technique buys

An experimental write-up of a serving problem and what fixed it, with every number traceable to a log in
`llm/results/`. Measured on 2026-10-15 and 2026-10-16.

## Summary

1. **The slow setup was a configuration mistake, not a hardware limit.** A launch that used `--enforce-eager`
   decoded at 17 tok/s. Turning on CUDA graphs and torch.compile alone gave 77 tok/s (4.4x).
2. **Speculative decoding with the model's built-in MTP head added another 2.6x** (200 tok/s at 3 draft tokens),
   with gains that taper: 1 draft token gives 1.7x, 3 give 2.6x, 5 give 2.85x, while draft acceptance falls from 92% to 65%.
3. **Parallel agents scale almost for free.** With MTP, 4 concurrent agents produced 535 tok/s in total while each kept
   187 tok/s; 8 agents produced 939 tok/s. Single-stream speed stays near 200 tok/s.
4. **No quality difference was detectable.** On 51 SWE-bench Verified instances, MTP resolved 14 and plain decoding 9
   (McNemar exact p = 0.18). The sample is too small to show anything but a large effect, and speculative decoding is
   designed to preserve the output distribution, so the gap is most likely noise. The agent finished each task faster
   with MTP: 70 s versus 115 s on average.
5. **Docker was used under a hard disk budget** and left the host exactly as it was found (section 6).

## 1. Setup

| | |
|---|---|
| GPU | NVIDIA H100 NVL, 94 GB (two cards, one used per server) |
| Model | Qwen3.8-27B, FP8 checkpoint, hybrid architecture: Gated-DeltaNet linear attention in 3 of every 4 layers, full attention every 4th layer, one built-in MTP head |
| Engine | vLLM 0.29.0, torch 2.13 (CUDA 13.0), single GPU, bf16 KV cache |
| Serving flags | `--max-model-len 65536 --gpu-memory-utilization 0.45 --max-num-seqs 8 --gdn-prefill-backend triton --enable-prefix-caching --reasoning-parser qwen3`, `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Sampling | temperature 0.6, top-p 0.95, top-k 20 in these runs. The model card recommends temperature 1.0 with thinking on and 0.7 with thinking off, so these runs were not at the recommended settings (see section 5) |
| Agent | mini-swe-agent 2.4.6 (tool calling), 100-step limit, thinking off |
| Evaluation | swebench 5.x harness, SWE-bench Verified, official per-instance images |

`llm/serve_qwen38.sh` is the launcher. KV cache stays bf16 because only 16 of the 64 layers use full attention
(about 65 KB per token), so there is no reason to quantise it.

## 2. Why the old launch was slow and then failed

The earlier attempt ran vLLM with `--enforce-eager`, which switches off CUDA graphs and torch.compile, and then
crashed at start-up in a FlashInfer JIT compile: the environment's torch is built for CUDA 13 but only the CUDA 12.9
toolkit is installed, so `nvcc` could not build the sampler kernel. Two settings avoid it without touching the
toolchain: `VLLM_USE_FLASHINFER_SAMPLER=0` and `--gdn-prefill-backend triton`. A one-time DeepGEMM warm-up still
compiles about 1,500 FP8 kernels (about 2 minutes) and is cached afterwards.

## 3. Speed by configuration

Each configuration was started fresh and measured on six coding prompts, two rounds, one stream, 1,024 output tokens
(2,048 in thinking mode). `llm/sweep.sh` drives it and `llm/bench_decode.py` measures.

| Configuration | Instruct tok/s (mean) | median | Thinking tok/s (mean) | MTP accept. (instruct / thinking) | TTFT ms | Coding check |
|---|---|---|---|---|---|---|
| Eager, no speculation (the slow setup) | 17.4 | 17.6 | 18.0 | - / - | 144 | 40/40 |
| CUDA graphs, no speculation | 77.2 (1.00x) | 77.0 | 76.6 | - / - | 157 | 40/40 |
| CUDA graphs + MTP, 1 draft token | 130.7 (1.69x) | 131.6 | 122.0 | 92% / 80% | 156 | 40/40 |
| CUDA graphs + MTP, 2 draft tokens | 167.7 (2.17x) | 169.2 | 148.7 | 85% / 70% | 121 | 40/40 |
| CUDA graphs + MTP, 3 draft tokens | 200.5 (2.60x) | 204.7 | 166.1 | 80% / 59% | 127 | 40/40 |
| CUDA graphs + MTP, 4 draft tokens | 214.6 (2.78x) | 215.7 | 162.2 | 72% / 49% | 124 | 40/40 |
| CUDA graphs + MTP, 5 draft tokens | 219.8 (2.85x) | 220.7 | 161.7 | 65% / 43% | 125 | 40/40 |

Reading the table:

- **CUDA graphs are the largest single factor.** A 27B decode step is short, so per-step launch overhead dominates in
  eager mode.
- **More draft tokens help less each time.** Going from 3 to 5 draft tokens adds 10% in instruct mode and nothing in
  thinking mode, where acceptance is lower (reasoning text is harder to predict than code). Three is the practical
  choice and is what the server runs.
- **Thinking mode is slower under speculation** (166 versus 200 tok/s at 3 tokens) for the same reason.
- **The coding check is at its ceiling in every row** (10 small problems, hidden unit tests, 4 samples each). It shows
  the servers are healthy but cannot tell the configurations apart; the SWE-bench run in section 5 is the real
  quality comparison.

## 4. Throughput with several agents in parallel

`llm/bench_concurrency.py`: N coding prompts sent at the same moment, MTP3 on one GPU and plain decoding on the other.
Aggregate is total tokens over the time to the last token; per stream is each stream's own decode rate.

| Concurrent agents | MTP3 aggregate tok/s | MTP3 per stream | No speculation aggregate | No speculation per stream |
|---|---|---|---|---|
| 1 | 192 | 197 | 76 | 78 |
| 2 | 280 | 197 | 117 | 74 |
| 4 | 535 | 187 | 215 | 72 |
| 8 | 939 | 175 | 391 | 69 |

Decode is limited by memory bandwidth, not compute, so extra concurrent requests are nearly free until the batch gets
large. Draft acceptance stayed at 77-78% at every level. The 2-agent aggregate is low only because the two replies
ended at different times; each stream ran at about 197 tok/s.

**Limits of this measurement.** Prompts were short (about 50 tokens in), so prefill was negligible and the KV cache was
small. Real agent traffic has contexts of 10-60k tokens, which will lower per-stream speed. Treat these as best-case
numbers. A single stream cannot reach 500 tok/s on this hardware: the FP8 bandwidth ceiling is about 144 tok/s without
speculation, and MTP already gets 2.6x past it.

## 5. Quality: SWE-bench Verified, MTP versus plain decoding

51 instances sampled with a fixed seed (the run was stopped at 51 of 60 by the disk guard, section 6), the same agent
and settings against two servers on two GPUs, one sample per instance. Patches were scored by the official harness in
the official per-instance containers.

| Arm | Instances | Resolved | Rate (95% CI) | Mean agent time (s) | Mean model calls | Hit step limit | Draft acceptance |
|---|---|---|---|---|---|---|---|
| MTP, 3 draft tokens | 51 | 14 | 27.5% (17-41%) | 70 | 65.8 | 30 | 91% |
| No speculation | 51 | 9 | 17.6% (10-30%) | 115 | 78.3 | 38 | - |

The draft acceptance for the SWE runs comes from server counters that overlap across the three concurrent agents, so it is
approximate.

Paired on the 51 common instances: both resolved 7, only MTP 7, only plain 2, neither 35. McNemar exact two-sided
p = 0.18.

What this does and does not show:

- **No significant difference.** The point estimates favour MTP, but with 9 discordant pairs the data are compatible
  with no difference, which is what the method predicts: draft tokens are verified against the full model and
  accepted by rejection sampling, so sampled output follows the same distribution. Floating-point effects in the
  parallel verification still make outputs differ from plain decoding in detail (one 3090 study found 92-100% of
  greedy outputs diverge at 1,600 tokens), which is why a downstream test like this is worth running.
- **These settings are not the model's best case.** The model card reports its agentic-coding scores (for example
  SWE-bench Pro 61.7 and Terminal Bench 2.1 73.0) with thinking on, temperature 1.0 and a 256K context, and warns that
  lower reasoning effort in multi-turn agent tasks can raise failures and retries. This run used thinking off,
  temperature 0.6 and a 100-step cap, so its resolve rates should be read as a floor for this setup and not as a
  statement about the model's coding ability. Section 10 repeats the question with the recommended settings.
- **The measurement is limit-bound.** 59% and 75% of runs ended by hitting the 100-step limit, so these numbers measure
  "solved within 100 steps with thinking off", not the model's ceiling. They are far below the roughly 71% a related
  model (Qwen3-Coder-Next) reports with up to 300 turns (arXiv 2603.00729) and are not comparable to it.
- **Speed.** The agent finished tasks 1.63x faster on average with MTP (median per-instance ratio 1.34x). That is well
  below the raw 2.6x decode gain because prefill, tool execution and the 100-step cap do not speed up.
- **Not measured: run-to-run noise.** The right control is to run the same arm twice. It was not done because
  Docker Hub allows about 100 anonymous pulls per 6 hours per address, and a second pass would exceed it. Until that is
  done, the 14-versus-9 gap should not be read as an effect in either direction.

## 6. Controlled use of Docker

SWE-bench needs per-instance container images, and Docker on this host stores everything on a shared disk outside the
project directory. The runner (`llm/swe/run_swe.py`) therefore:

- pulls one image per worker at a time (3 workers) and removes each by exact name after its instance is scored;
- checks Docker's image growth and the filesystem growth against hard caps before every pull (15 GB and 20 GB) and
  stops pulling when exceeded; it tripped at 15.005 GB, which is why the run ended at 51 instances;
- runs each agent in a container limited to 12 GB of memory, 4 CPUs, 1,024 processes and no network, with no logging
  driver;
- never prunes, and touches only containers and images whose names it created;
- writes every step to `run.jsonl`, and compares the final state with the starting state.

Result: peak growth was 15.0 GB of images and 15.2 GB on the filesystem. After the run, Docker was back to exactly
7.465 GB in 4 images and 2 containers, the same as before, with no SWE-bench image or container left. Early in the run I
underestimated the peak (about 4 GB) from the first few small instances; image sizes turned out to vary widely
(a single matplotlib image is listed at 10.7 GB), and the caps are what kept it bounded.

## 7. What others report, and how to read it

Community reports of very fast Qwen 27B-class inference on a single RTX 3090 (24 GB) trace back to four ideas:
quantised weights (4-bit), speculative decoding (MTP or a DFlash drafter), a quantised KV cache, and greedy
sampling on short code prompts. The careful measurements sit around 60-70 tok/s on a 3090 with a plain baseline of
about 35-42 tok/s, a speedup of roughly 1.6-1.9x; the larger headlines are peaks from demos, such as 207 tok/s against a mean of
about 130 on the same stack. Reported speedups also fade on long outputs and open-ended text. So the ratios, not the absolute speeds, are
the comparable part. Ours (2.6x from speculation on top of CUDA graphs) sits above the better-measured 3090 ratios
(1.6-1.9x) and below the best peak claims (3.4x and more); the stacks, models, quantisation and sampling differ, so
this is context, not a like-for-like comparison.

Sources read: [HackMD Qwen3.8-27B 3090 study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090),
[Hacker News 47838788](https://news.ycombinator.com/item?id=47838788),
[3090 Ti DFlash versus MTP](https://ianlpaterson.com/blog/3090-ti-qwen-speedup-dflash-mtp/),
[EXL3 + DFlash2 repository](https://github.com/r0b0tlab/qwen38-exl3-dflash2),
[DFlash2 discussion #9](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/discussions/9).
The Qwen technical reports (arXiv 2505.09388, 2509.17765, 2511.21631, 2603.00729, 2601.21337) do not discuss consumer
GPUs or speculative-decoding acceptance; Qwen3-Omni's serving setup (vLLM with torch.compile and CUDA graphs) is
consistent with the first finding above.

## 8. Limitations

- One model, one GPU type, one engine version; single-run measurements without repeats beyond two rounds of six prompts.
- Speed benchmarks use short prompts and non-thinking output except where stated; long-context agent traffic will be slower.
- The SWE-bench arms ran on different GPUs of a shared machine, so wall-time comparisons carry some noise.
- Thinking was off and the step limit was 100; both depress the resolve rate and may change the MTP comparison.
- Not tested: DFlash2 drafters in vLLM, suffix decoding, FP8 KV cache, async scheduling, a repeat of the same arm.

## 9. Reproduce

```bash
llm/serve_qwen38.sh                                   # server on port 8000
python llm/bench_decode.py --port 8000                # speed
python llm/bench_concurrency.py --port 8000           # parallel agents
GPU=1 PORT=8001 llm/sweep.sh                          # every configuration in section 3
python llm/swe/run_swe.py --arms mtp3=8000 nospec=8001 --n 60 --seed 0 --workers 3 --cap-gb 15 --fs-cap-gb 20 --step-limit 100
llm/collect_results.sh && python llm/make_report.py   # raw logs -> tables
```

Raw logs are in `llm/results/` (benchmarks, per-instance JSONL, patches, run configs). Server logs, agent trajectories
and container logs are kept on the GPU host.
