# Benchmark protocol

Run on the model host after the tuned service is healthy and competing GPU workloads are stopped. The harness disables prompt-cache reuse, tokenizes each target with the server tokenizer, and records API prompt counts plus llama.cpp prefill and decode rates.

Default repeated comparison (three cold samples at 8K and 32K):

```bash
python3 throughput.py --api-key-file "$QWEN_API_KEY_FILE" --concurrency 1 --max-tokens 128
python3 throughput.py --api-key-file "$QWEN_API_KEY_FILE" --concurrency 2 --max-tokens 128
```

Full-capacity stress checks for the configured two 262,144-token slots:

```bash
python3 throughput.py --api-key-file "$QWEN_API_KEY_FILE" --prompt-tokens 260000 --runs 1 --concurrency 1 --max-tokens 128
python3 throughput.py --api-key-file "$QWEN_API_KEY_FILE" --prompt-tokens 260000 --runs 1 --concurrency 2 --max-tokens 128
```

The 260K target leaves room for chat-template tokens and 128 generated tokens in each slot. The two-request case exercises about 520K prompt tokens across both slots. With MTP and the 32 GiB expert cache, both slots passed this stress test without truncation or OOM, but one request decoded at only 0.85 tok/s while the other prefetched. This is a capacity/stability check (one sample), not a statistically stable concurrent-speed score. Record actual token counts, both throughput measures, GPU and host utilization, failures, and truncation. End-to-end aggregate tokens per second includes prefill time and should not be compared directly with decode tok/s.
