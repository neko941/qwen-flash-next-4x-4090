#!/usr/bin/env bash
set -euo pipefail
: "${LLAMA_SERVER_BIN:?Set path to the llama-server binary}"
: "${QWEN_MODEL:?Set path to IQ3_S shard 1}" 
: "${QWEN_API_KEY_FILE:?Set path to the API-key file}"
: "${QWEN_MTP_MODEL:?Set path to the Qwen3.8 MTP draft GGUF}"
test -x "$LLAMA_SERVER_BIN"
test -s "$QWEN_MODEL"
test -s "$QWEN_API_KEY_FILE"
test -s "$QWEN_MTP_MODEL"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "$LLAMA_SERVER_BIN" \
  --model "$QWEN_MODEL" --alias qwen3.8-flash-next-abliterated-iq3_s \
  --host 127.0.0.1 --port 11328 --api-key-file "$QWEN_API_KEY_FILE" \
  --no-webui --slots --jinja --ctx-size 524288 --parallel 2 --cache-ram 32768 \
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 \
  --threads 8 --threads-batch 10 --batch-size 512 --ubatch-size 256 \
  --gpu-layers 999 --split-mode layer --tensor-split 1,1,1,1 --fit off \
  --cpu-moe --moe-cache-mib 32768 --moe-cache-cpu-share 0 \
  --load-mode none --lazy-mode off \
  --override-tensor '^per_layer_token_embd\.weight$=CPU' \
  --spec-type draft-mtp --spec-draft-model "$QWEN_MTP_MODEL" --spec-draft-n-max 3 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
