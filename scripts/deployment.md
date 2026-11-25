# Deployment configuration

`serve.sh` is a host-neutral launcher for the measured setup. Set `LLAMA_SERVER_BIN`, `QWEN_MODEL` (first IQ3_S shard), `QWEN_MTP_MODEL` (the Q4_0 MTP draft GGUF), and `QWEN_API_KEY_FILE` in the host environment before running it. It has no credentials or machine-specific paths.

The tested configuration enables MTP with a three-token draft, uses a 32 GiB expert cache with CPU share 0, Q4_0 K/V cache, and two 262,144-token slots. MTP loaded successfully at full slot size; accepted 64–68% of drafted tokens in short tests and raised single-request decode from 64.49 to 97.52 tok/s at 8K prompts. Two simultaneous long requests can still starve one request during the other slot's prefill.

For Hermes, use `model.provider: custom`, `model.default: qwen3.8-flash-next-abliterated-iq3_s`, and the forwarded OpenAI-compatible `/v1` endpoint. Do not select a named external provider for this local model; it will validate the ID against that provider's catalog instead of the llama.cpp server.
