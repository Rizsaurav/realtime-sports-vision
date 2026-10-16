## Speed and coding check by configuration

| Configuration | Instruct tok/s (mean) | median | Thinking tok/s (mean) | MTP accept. (instruct / thinking) | TTFT ms | Coding check |
|---|---|---|---|---|---|---|
| Eager, no speculation (the slow setup) | 17.4 | 17.6 | 18.0 | - / - | 144 | 40/40 |
| CUDA graphs, no speculation | 77.2 (1.00x) | 77.0 | 76.6 | - / - | 157 | 40/40 |
| CUDA graphs + MTP, 1 draft token | 130.7 (1.69x) | 131.6 | 122.0 | 92% / 80% | 156 | 40/40 |
| CUDA graphs + MTP, 2 draft tokens | 167.7 (2.17x) | 169.2 | 148.7 | 85% / 70% | 121 | 40/40 |
| CUDA graphs + MTP, 3 draft tokens | 200.5 (2.60x) | 204.7 | 166.1 | 80% / 59% | 127 | 40/40 |
| CUDA graphs + MTP, 4 draft tokens | 214.6 (2.78x) | 215.7 | 162.2 | 72% / 49% | 124 | 40/40 |
| CUDA graphs + MTP, 5 draft tokens | 219.8 (2.85x) | 220.7 | 161.7 | 65% / 43% | 125 | 40/40 |


## Throughput by number of concurrent agents

| Concurrent agents | MTP3 aggregate tok/s | MTP3 per stream | No speculation aggregate | No speculation per stream |
|---|---|---|---|---|
| 1 | 192 | 197 | 76 | 78 |
| 2 | 280 | 197 | 117 | 74 |
| 4 | 535 | 187 | 215 | 72 |
| 8 | 939 | 175 | 391 | 69 |


## SWE-bench Verified: main60

| Arm | Instances | Resolved | Rate (95% CI) | Mean agent time (s) | Mean model calls | Hit step limit | Draft acceptance |
|---|---|---|---|---|---|---|---|
| mtp3 | 51 | 14 | 27.5% (17-41%) | 70 | 65.8 | 30 | 91% |
| nospec | 51 | 9 | 17.6% (10-30%) | 115 | 78.3 | 38 | - |

Paired on 51 common instances: both 7, only mtp3 7, only nospec 2, neither 35. McNemar exact two-sided p = 0.18.

Disk control: 51 instance images pulled and each removed after its instance. Peak growth over the starting state while up to 3 instances ran at once: 15.0 GB of images, 15.2 GB on the filesystem. (Docker's per-image size from `docker image inspect` understates real use, so the totals above are the numbers to trust.)


## SWE-bench Verified: pilot6

| Arm | Instances | Resolved | Rate (95% CI) | Mean agent time (s) | Mean model calls | Hit step limit | Draft acceptance |
|---|---|---|---|---|---|---|---|
| mtp3 | 6 | 3 | 50.0% (19-81%) | 54 | 39.0 | 3 | 85% |

Disk control: 6 instance images pulled and each removed after its instance. Peak growth over the starting state while up to 3 instances ran at once: 4.4 GB of images, 4.4 GB on the filesystem. (Docker's per-image size from `docker image inspect` understates real use, so the totals above are the numbers to trust.)


## SWE-bench Verified: smoke3

| Arm | Instances | Resolved | Rate (95% CI) | Mean agent time (s) | Mean model calls | Hit step limit | Draft acceptance |
|---|---|---|---|---|---|---|---|
| mtp3 | 1 | 1 | 100.0% (21-100%) | 18 | 6.0 | 0 | 86% |

Disk control: 1 instance images pulled and each removed after its instance. Peak growth over the starting state while up to 3 instances ran at once: 0.0 GB of images, 0.0 GB on the filesystem. (Docker's per-image size from `docker image inspect` understates real use, so the totals above are the numbers to trust.)
