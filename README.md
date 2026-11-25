# Qwen3.8 Flash Next on four RTX 4090s

Deployment and tuning record for Huihui's Qwen3.8-Flash-Next-abliterated-v2 GGUF IQ3_S on four RTX 4090s with llama.cpp and CUDA 11.8.

## Repository files

```text
qwen-flash-next-4x-4090/
├── README.md
├── .gitignore
├── environment.yml
├── benchmarks/
│   ├── README.md
│   ├── moe-cache-tuning.md
│   └── throughput.py
└── scripts/
    ├── deployment.md
    └── serve.sh
```

## Hardware and engine

- Linux host, Ubuntu 20.04, 4× RTX 4090 24 GB, CUDA toolkit 11.8, 188 GiB system RAM.
- Engine: llama.cpp commit `44d50873b56b568c28ef2ad8d6445069feaa9fbd`, built with CUDA 11.8 / SM 89.
- Model: Huihui v2 Swift IQ3_S, two GGUF shards (~83.7 GB total).
- Context limit: model config sets `max_position_embeddings` to 262,144 tokens per sequence. Serving uses two slots at 262,144 each, for 524,288 aggregate tokens across two requests; 512K is not a single sequence.
- Placement: MoE experts and PLE in host RAM; 32 GiB expert cache split across four GPUs; Q4_0 K/V KV cache offloaded to GPU.
- MTP: enabled with the separate Q4_0 Qwen3.8 draft GGUF, three draft tokens, and Q8_0 draft KV. It loaded at the full 262,144-token slot size and had 64–68% draft acceptance on short tests.
- Competing inference services were stopped before calibration.

## Current result

The active deployment uses a measured 32,768 MiB expert cache with CPU share 0. This exceeds the tuner’s original 18,112 MiB recommendation; repeated short tests supported the larger cache, and the final 512K stress passed with it.

With the same 32 GiB cache, MTP raised median 8K decode from 64.49 to 97.52 tok/s over three cold runs; prefill was similar (157.90 to 154.48 tok/s). At 32K, one sample measured 152.11 prefill / 114.02 decode tok/s with MTP, versus 155.16 / 65.14 without it. Detailed results and rejected batch-size experiments are in `benchmarks/moe-cache-tuning.md`.

The final MTP configuration passed a two-slot near-512K stress using 520,367 prompt-plus-output tokens, with no truncation, OOM, or server restart. It took 59:05; per-request decode was 0.85 and 58.84 tok/s, showing that the stock scheduler can starve one response while the other slot prefills. MTP is fastest for one active request; two slots retain context capacity but do not provide balanced long-prompt concurrency. Hermes defaults to this model alias.

The earlier 3.26 tok/s result was recorded while other services competed for the GPUs and is not comparable to the current test. The tuner calibration recorded up to 55.0 tok/s and predicted 64.5 tok/s after cache replay.

## Operational notes

The server binds to loopback and reads the API key from a host-only file. Keep credentials, machine names, private paths, and model data out of this repository. The configured model alias is `qwen3.8-flash-next-abliterated-iq3_s`.

## References

- [Qwen model config](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated/blob/main/config.json) — declares the 262,144-token maximum sequence length.
- [Inference optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) — benchmark and optimization workflow reference.
- [Strata single-4090 Qwen Flash reference](https://github.com/Tlyer233/strata-single-4090-qwen-flash) — different hardware and engine; use its workload shape as inspiration, not a directly comparable speed target.
- [llama.cpp MoE cache documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md) — cache controls and tuner used here.
- [PDScope](https://arxiv.org/abs/2610.39216) and [Adaptive Expert Assignment](https://arxiv.org/abs/2610.23006) — investigate phase-aware expert placement and transfer scheduling, which the measured stock configuration does not implement.
- [Bevy: interleaving concurrent requests for offloaded MoE inference](https://arxiv.org/abs/2611.04794) — motivates measuring aggregate throughput at concurrency 2; its scheduling gains require a modified runtime.
- [SP-MoE](https://arxiv.org/abs/2510.10302) — studies speculative expert prefetch for draft/target pairs; these runtime changes are not in this llama.cpp build.


