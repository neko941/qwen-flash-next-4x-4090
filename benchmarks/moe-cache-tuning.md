# MoE cache tuning

Tuned on four RTX 4090 cards after stopping competing inference workloads. llama.cpp's tuner loaded the model six times, generated a short synthetic workload, fit CPU/upload cost, then replayed routing traces over candidate cache sizes and CPU shares.

## Measured calibration runs

| Cache MiB | CPU share | Measured decode tok/s | Expert-cache hits |
|---:|---:|---:|---:|
| 4,736 | 0 | 27.0 | 71% |
| 4,736 | 0.5 | 33.3 | 69% |
| 4,736 | 0.95 | 26.6 | 60% |
| 14,336 | 0 | 55.0 | 93% |
| 14,336 | 0.5 | 50.9 | 92% |
| 14,336 | 0.95 | 48.0 | 89% |

## Tuner estimate and measured cache selection

The tuner estimated 47,962 MiB of total experts, 22,401 MiB free on CUDA0, and an 18,159 MiB cache cap. Routing replay predicted 64.5 tok/s at 18,112 MiB with CPU share 0, versus 57.4 tok/s at 14,336 MiB/share 0. It chose share 0 because that setting was fastest. The fit residual was 2.0%.

That recommendation was the first stable configuration. Later cold-prompt tests showed higher measured decode with a 32,768 MiB expert cache, so the active launcher now uses `--moe-cache-mib 32768 --moe-cache-cpu-share 0`. This measured setting exceeds the tuner's conservative memory cap; it was tested with two 262,144-token slots and passed the full two-slot stress with MTP enabled.

The cached-throughput estimate depends on the routing trace and fitted hardware model. Do not present the tuner estimate as a measured long-context result.

## Cold-prompt benchmark results

Prompt reuse was disabled. Prefill/decode rates are separate from end-to-end request throughput. The initial 18,112 MiB cache baseline and the 32,768 MiB MTP-off candidate used Q4_0 K/V and three samples at 8K; the 32K rows are one sample for the later cache/MTP candidates.

| Expert cache | MTP | Prompt | Samples | Median prefill tok/s | Median decode tok/s |
|---:|:---:|---:|---:|---:|---:|
| 18,112 MiB | off | 8K | 3 | 151.53 | 52.25 |
| 18,112 MiB | off | 32K | 3 | 147.16 | 50.99 |
| 32,768 MiB | off | 8K | 3 | 157.90 | 64.49 |
| 32,768 MiB | off | 32K | 1 | 155.16 | 65.14 |
| 32,768 MiB | on | 8K | 3 | 154.48 | 97.52 |
| 32,768 MiB | on | 32K | 1 | 152.11 | 114.02 |

The 32 GiB cache increased measured 8K decode by 23% over the 18 GiB baseline. With the cache held constant, MTP increased median 8K decode by 51% over the MTP-off candidate, with similar prefill. The three MTP-on 8K decode samples were 49.39, 97.52, and 109.83 tok/s; the first was a cold-start outlier. MTP draft acceptance was 64.8–68.0%, with mean accepted draft groups of about three tokens.

End-to-end generated tok/s divides output tokens by the full request time, including prefill. It is expected to be much lower for long prompts than decode tok/s and must not be reported as decode speed.

## Scheduler and split experiments

All comparisons below used MTP and the 32 GiB expert cache. The default 512 logical / 256 physical batch remained the best measured tradeoff; the alternatives did not fix the two-request decode imbalance.

| Candidate | Workload | Aggregate generated tok/s | Per-request decode tok/s | Outcome |
|---|---|---:|---:|---|
| 512 / 128 batch / ubatch | 2 × 8K | 1.36 | 1.61 / 75.94 | Rejected; lower aggregate than 512 / 256. |
| 2048 / 256 batch / ubatch | 2 × 8K | 2.39 | 4.68 / 76.33 | No meaningful gain over 512 / 256 (2.37 tok/s). |
| 512 / 256 batch / ubatch | 2 × 8K | 2.37 | 83.38 / 2.68 | Retained; higher single-request speed, but poor fairness remains. |
| Row split | startup | — | — | Unsupported by this CUDA backend (`CUDA0 does not support split buffers`); layer split retained. |

## Full-capacity stress

The checkpoint's max sequence length is 262,144 tokens per sequence. Prompt reuse was disabled; each request targeted 260,000 content tokens and generated 128 tokens. The final configuration was MTP on, 32,768 MiB expert cache, Q4_0 K/V, two slots, and layer split.

| Case | API prompt tokens | Generated | Prefill tok/s | Decode tok/s | End-to-end generated tok/s | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Two overlapping requests, near 512K aggregate | 260,057 / 260,054 | 128 each | 137.46 / 156.40 | 0.85 / 58.84 | 0.072 aggregate | 59:05 |

The run used 520,367 prompt-plus-output tokens total, 99.3% of the configured aggregate limit of 524,288. Both requests generated all 128 tokens, with no truncation, OOM, or server restart. The slots overlapped, but most long prefill work was serialized. Decode was highly uneven: one request fell to 0.85 tok/s while the other reached 58.84 tok/s. MTP does not solve this interference in the current scheduler. The 0.072 aggregate end-to-end rate includes 59 minutes of prefill and is not decode speed.

An earlier single-slot full-context run used the then-active 18,112 MiB cache with MTP off: 260,056 prompt tokens plus 128 generated, 145.39 prefill / 31.15 decode tok/s, 29:54 elapsed. It validates the per-sequence context limit but is not a measurement of the final MTP/cache configuration.

After the run, `nvidia-smi` showed 12,538–17,244 MiB used and 6,971–11,679 MiB free per GPU. Host memory showed 106 GiB available. These are observed snapshots, not instrumented high-water marks.

## Research notes

llama.cpp's MoE cache is split across GPUs; its tuner fits CPU/upload costs and replays routing traces to estimate other cache sizes. The final 32 GiB choice was confirmed by API benchmarks rather than the tuner's prediction ([MoE cache documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md)). [PDScope](https://arxiv.org/abs/2610.39216) and [Adaptive Expert Assignment](https://arxiv.org/abs/2610.23006) study phase-aware CPU/GPU expert placement and prefetching. Those techniques require scheduler/runtime changes beyond the stock build; the observed request starvation is consistent with the prefill/decode interference they target.
