#!/usr/bin/env bash
# 启动/替换 llama-server（本地编译版，非 docker）
# 2026-09-24 修正：本地直接跑进程；启动前必须先杀旧进程（防重复实例叠加吃内存，
# 曾 4 实例 × 8G 把整机内存打爆触发 OOM）。内存受控：-c 16384 单实例常驻约 3-4G。
# 用法: ./run_llama.sh <models/ 下的 gguf 文件名>
set -euo pipefail
MODEL_FILE="${1:?usage: ./run_llama.sh <gguf file under models/>}"
cd "$(dirname "$0")"

# 先杀旧实例（仅本进程，勿 pkill -f 匹配到无关 bun test 进程）
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  exe=$(readlink /proc/$pid/exe 2>/dev/null || true)
  if [ "$exe" = "$(pwd)/llama.cpp/build/bin/llama-server" ]; then
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 1

nohup ./llama.cpp/build/bin/llama-server -m "models/$MODEL_FILE" \
  --host 127.0.0.1 --port 11941 -c 16384 --parallel 2 \
  > llama_server.log 2>&1 &

echo "llama-server starting: model=$MODEL_FILE port=11941 (mem 3-4G, parallel=2, ctx 16K)"
