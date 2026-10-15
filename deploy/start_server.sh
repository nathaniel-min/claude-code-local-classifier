#!/bin/bash
# 启动 llama-server 常驻 11943（v2 微调分类器模型推理）
set -euo pipefail
cd ~/clf
if ! pgrep -f "llama-server.*11943" >/dev/null; then
  ~/clf/llama.cpp/build/bin/llama-server \
    -m ~/clf/qwen35-clf-v2-q4_k_m.gguf \
    --host 127.0.0.1 --port 11943 -c 16384 --parallel 1 \
    -ngl 999 -ctk q8_0 -ctv q8_0 \
    > ~/clf/llama-server.log 2>&1 &
  echo "llama-server started (pid $!) -> log: ~/clf/llama-server.log"
else
  echo "llama-server already running on 11943"
fi
