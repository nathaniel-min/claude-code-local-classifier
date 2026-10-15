#!/usr/bin/env python3
"""merge_quantize.py — 合并 LoRA adapter → HF 全量 → GGUF → Q4_K_M 量化。

一次性脚本，服务器 CPU 端跑（需临时 venv 装 torch-cpu + llama.cpp 转换工具链）。
魔搭端只产出 adapter，不做 merge/量化。

流程：
  1. 加载底座 Qwen3.5-4B（HF，剥视觉编码器） + adapter（PEFT）
  2. merge_and_unload → 保存 HF 全量到 models/qwen3.5-4b-clf/
  3. llama.cpp convert_hf_to_gguf.py 转 GGUF（llama.cpp 已支持 Qwen3.5）
  4. llama.cpp quantize Q4_K_M → models/qwen3.5-4b-clf-v1-q4_k_m.gguf

用法：
  python3 merge_quantize.py --adapter models/qwen3.5-4b-clf/ \
      --out models/qwen3.5-4b-clf-v1-q4_k_m.gguf
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 2026-09-20 升级：微调基座改为 Qwen3.5-4B（SemIf 验证过的最强底座）
MODEL_ID = "Qwen/Qwen3.5-4B"

# llama.cpp 工具路径（可在容器内跑；docker 镜像 ghcr.io/ggml-org/llama.cpp 自带）
LLAMA_CPP_IMG = "ghcr.io/ggml-org/llama.cpp:server"


def find_llama_tools():
    """尝试定位 convert_hf_to_gguf.py / quantize。"""
    for p in ("convert_hf_to_gguf.py", "llama-quantize", "quantize"):
        if shutil.which(p):
            return p
    return None


def load_causal_model(source, **kw):
    """Qwen3.5 多模态需剥视觉编码器（SemIf 同款），只加载语言部分。"""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    cls = AutoModelForCausalLM
    if getattr(config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(__import__("transformers"), "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("transformers 版本过低，缺 Qwen3_5ForCausalLM（需 >=5.17）")
        config = config.get_text_config()
    kw.setdefault("trust_remote_code", True)
    return cls.from_pretrained(source, config=config, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter 目录")
    ap.add_argument("--out", default=str(HERE / "models" / "qwen3.5-4b-clf-v1-q4_k_m.gguf"))
    ap.add_argument("--hf-out", default=str(HERE / "models" / "qwen3.5-4b-clf-hf"))
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    hf_out = Path(args.hf_out)
    out = Path(args.out)
    if not (adapter / "adapter_model.safetensors").exists():
        sys.exit(f"adapter 目录里没有 adapter_model.safetensors: {adapter}")

    # ---------- 1+2: merge ----------
    print("合并 LoRA → HF 全量 ...", flush=True)
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = load_causal_model(args.model, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(adapter), device_map="cpu")
    model = model.merge_and_unload()
    hf_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(hf_out), safe_serialization=True)
    tokenizer.save_pretrained(str(hf_out))
    del model, base
    import gc
    gc.collect()
    print(f"HF 全量 -> {hf_out}", flush=True)

    # ---------- 3: HF → GGUF ----------
    tool = find_llama_tools()
    if tool:
        # 本机有工具（venv 里装的）
        subprocess.run([tool, str(hf_out), "--outfile", str(out.with_suffix(".f16.gguf")),
                        "--outtype", "f16"], check=True)
    else:
        # 用 llama.cpp docker 容器（本地镜像已有）
        subprocess.run([
            "docker", "run", "--rm", "-v", f"{HERE}:/work",
            LLAMA_CPP_IMG, "-m", f"/work/{hf_out.name}",
            "--outfile", f"/work/{out.with_suffix('.f16.gguf').name}",
            "--outtype", "f16",
        ], check=True)
    print(f"F16 GGUF -> {out.with_suffix('.f16.gguf')}", flush=True)

    # ---------- 4: quantize Q4_K_M ----------
    if tool and tool.endswith("convert_hf_to_gguf.py"):
        quant = "llama-quantize" if shutil.which("llama-quantize") else "quantize"
        subprocess.run([quant, str(out.with_suffix(".f16.gguf")), str(out),
                        "Q4_K_M"], check=True)
    else:
        subprocess.run([
            "docker", "run", "--rm", "-v", f"{HERE}:/work",
            LLAMA_CPP_IMG, "-m", f"/work/{out.with_suffix('.f16.gguf').name}",
            "-o", f"/work/{out.name}", "-q", "Q4_K_M",
        ], check=True)
    print(f"Q4_K_M -> {out}", flush=True)
    print("验收: 用 run_llama.sh 起服务后跑 bench/jev_replay_v2.py")


if __name__ == "__main__":
    sys.exit(main())
