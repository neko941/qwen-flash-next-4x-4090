#!/usr/bin/env python3
"""Repeated prompt-length and decode-throughput benchmark for llama.cpp's OpenAI API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
import urllib.parse
import urllib.request


def request(url, key, payload, timeout=7200):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def prompt_for(target, seed, key, tokenize_url):
    passage = ("A research team compares independent archives, verifies source records, and writes a careful summary. "
               if seed % 2 else
               "The system stores expert weights in host memory and streams selected experts to the GPUs. ")
    prefix = f"Benchmark case {target}, independent sample {seed}. Explain the following passage carefully. "
    reps = max(1, target // 20)
    for _ in range(12):
        content = prefix + passage * reps
        count = len(request(tokenize_url, key, {"content": content, "add_special": False}, timeout=900)["tokens"])
        if abs(count - target) <= 8:
            return content, count
        adjusted = max(1, round(reps * target / count))
        if adjusted == reps:
            break
        reps = adjusted
    content = prefix + passage * reps
    count = len(request(tokenize_url, key, {"content": content, "add_special": False}, timeout=900)["tokens"])
    return content, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:11328/v1")
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--model", default="qwen3.8-flash-next-abliterated-iq3_s")
    p.add_argument("--prompt-tokens", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    if args.runs < 1 or args.concurrency not in (1, 2) or args.max_tokens < 1 or any(n < 1 for n in args.prompt_tokens):
        p.error("runs must be >= 1, concurrency must be 1 or 2, and token counts must be positive")
    with open(args.api_key_file, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        p.error("API key file is empty")
    base = args.base_url.rstrip("/")
    url = base + "/chat/completions"
    parsed = urllib.parse.urlsplit(base)
    tokenize_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/tokenize", "", ""))
    result = {"model": args.model, "runs": args.runs, "concurrency": args.concurrency,
              "max_tokens": args.max_tokens, "prompt_cache": False, "cases": []}
    for target in args.prompt_tokens:
        batch_rates, samples = [], []
        for run in range(args.runs):
            prepared = [prompt_for(target, run * args.concurrency + slot, key, tokenize_url)
                        for slot in range(args.concurrency)]
            if any(abs(count - target) > 8 for _, count in prepared):
                raise RuntimeError(f"prompt construction did not reach target {target}: {[n for _, n in prepared]}")
            payloads = [{"model": args.model, "messages": [{"role": "user", "content": content}],
                         "max_tokens": args.max_tokens, "temperature": 0, "stream": False, "cache_prompt": False}
                        for content, _ in prepared]
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                responses = list(pool.map(lambda body: request(url, key, body), payloads))
            elapsed = time.perf_counter() - start
            items = []
            for slot, data in enumerate(responses):
                usage = data.get("usage", {})
                generated = usage.get("completion_tokens")
                if not generated:
                    raise RuntimeError(f"API response lacks completion token usage: {data}")
                prompt_tokens = usage.get("prompt_tokens")
                if prompt_tokens is None or prompt_tokens + generated > 262144:
                    raise RuntimeError(f"request exceeded the per-sequence context limit: {usage}")
                timings = data.get("timings", {})
                items.append({"target_content_tokens": prepared[slot][1], "prompt_tokens": prompt_tokens,
                              "completion_tokens": generated,
                              "llama_prefill_tps": round(1000 * timings["prompt_n"] / timings["prompt_ms"], 3)
                              if timings.get("prompt_n") and timings.get("prompt_ms") else None,
                              "llama_decode_tps": round(1000 * timings["predicted_n"] / timings["predicted_ms"], 3)
                              if timings.get("predicted_n") and timings.get("predicted_ms") else None})
            aggregate_tokens = sum(item["prompt_tokens"] + item["completion_tokens"] for item in items)
            if args.concurrency == 2 and aggregate_tokens > 524288:
                raise RuntimeError(f"requests exceeded the aggregate context limit: {aggregate_tokens}")
            total = sum(item["completion_tokens"] for item in items)
            rate = total / elapsed
            batch_rates.append(rate)
            samples.append({"batch_elapsed_s": round(elapsed, 3), "aggregate_generated_tps": round(rate, 3), "requests": items})
        result["cases"].append({"target_prompt_tokens_per_request": target,
                                 "median_end_to_end_generated_tps": round(statistics.median(batch_rates), 3),
                                 "min_end_to_end_generated_tps": round(min(batch_rates), 3),
                                 "max_end_to_end_generated_tps": round(max(batch_rates), 3),
                                 "median_prefill_tps_per_request": round(statistics.median(x["llama_prefill_tps"] for sample in samples for x in sample["requests"] if x["llama_prefill_tps"] is not None), 3),
                                 "median_decode_tps_per_request": round(statistics.median(x["llama_decode_tps"] for sample in samples for x in sample["requests"] if x["llama_decode_tps"] is not None), 3),
                                 "samples": samples})
        print(json.dumps(result["cases"][-1]), flush=True)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
