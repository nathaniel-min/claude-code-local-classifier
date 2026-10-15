#!/usr/bin/env python3
"""eval_encoder.py — encoder 副线 holdout 评测（魔搭端 + 服务器端通用）。

加载底座 + head.pt（+ lora adapter），对 holdout 逐条推理 P(yes)，指标与
eval_adapter.py 完全同流程（metrics.py 共享）：raw / temperature 校准后两组
（deny 召回、误拦、sep、AUC、ECE），temperature.json 写入 ckpt 目录。

逐条 batch=1 推理并记录延迟——服务器 CPU 跑即直接产出 p50/p95 延迟验收
（不起 llama.cpp，进程内推理）。@0.5 阈值判定用 temperature 校准后概率
（与 ACCEPTANCE-v1.md 阈值说明一致）。

用法：
  # 魔搭端（训练后）
  python train/eval_encoder.py --ckpt output/encoder-lora \
      --holdout datasets/train-v2-enc/holdout-test.jsonl

  # 服务器端 CPU 验收（int8 动态量化加速）
  python train/eval_encoder.py --ckpt output/encoder-lora --device cpu --int8 \
      --holdout datasets/train-v2-enc/holdout-test.jsonl
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from metrics import fit_temperature, summarize          # noqa: E402  同目录
from train_encoder import EncoderClassifier            # noqa: E402  同目录

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def load_classifier(ckpt, model_id, device, int8=False):
    """底座 + head.pt（+ adapter）→ EncoderClassifier。"""
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    dtype = torch.float32 if (device == "cpu" or int8) else torch.bfloat16
    base = AutoModel.from_pretrained(
        cfg.get("base_model", model_id), torch_dtype=dtype,
        trust_remote_code=True)
    if (Path(ckpt) / "adapter").exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(Path(ckpt) / "adapter"))
    if int8:
        base = torch.ao.quantization.quantize_dynamic(
            base, {torch.nn.Linear}, dtype=torch.qint8)
    model = EncoderClassifier(base, cfg["hidden"])
    model.head.load_state_dict(torch.load(Path(ckpt) / "head.pt",
                                           map_location="cpu"))
    model = model.to(device).eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train_encoder.py 输出目录")
    ap.add_argument("--holdout", default=str(BASE / "datasets" / "train-v2-enc" / "holdout-test.jsonl"))
    ap.add_argument("--model", default="Alibaba-NLP/gte-multilingual-base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU 线程数；threads 对照实测 4 最优（8 过订阅慢 28%%）")
    ap.add_argument("--int8", action="store_true", help="CPU int8 动态量化（延迟验收用）")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)   # Phase A 实测：4 最优，默认 8 过订阅慢 28%

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"ckpt={args.ckpt} device={args.device} int8={args.int8}  "
          f"holdout: {len(rows)} (yes={sum(1 for r in rows if r['label']=='yes')} "
          f"no={sum(1 for r in rows if r['label']=='no')})", flush=True)

    model, cfg = load_classifier(args.ckpt, args.model, args.device, args.int8)
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("base_model", args.model),
                                              trust_remote_code=True)

    results = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            enc = tokenizer(r["text"], truncation=True,
                            max_length=cfg.get("max_length", 8192),
                            return_tensors="pt").to(args.device)
            t1 = time.monotonic()
            logits = model(**{k: v for k, v in enc.items()})[0].float()
            dt = time.monotonic() - t1
            diff = (logits[1] - logits[0]).item()      # z_yes − z_no
            p_yes = 1.0 / (1.0 + math.exp(-diff))
            results.append({"case_id": r.get("case_id"), "label": r["label"],
                            "p_yes": round(p_yes, 4), "logit_diff": round(diff, 4),
                            "latency": round(dt, 3)})
            if i % 25 == 0:
                el = time.monotonic() - t0
                print(f"  {i}/{len(rows)}  {el/i:.2f}s/条", flush=True)

    # ---------- 指标（与 eval_adapter.py 同流程） ----------
    diffs = [r["logit_diff"] for r in results]
    ys = [1 if r["label"] == "yes" else 0 for r in results]
    deny_raw = [r["p_yes"] for r in results if r["label"] == "yes"]
    allow_raw = [r["p_yes"] for r in results if r["label"] == "no"]
    raw = summarize(deny_raw, allow_raw)

    T, nll = fit_temperature(diffs, ys)
    calibrated = None
    if T is not None:
        p_cal = [1.0 / (1.0 + math.exp(-d / T)) for d in diffs]
        deny_cal = [p for p, r in zip(p_cal, results) if r["label"] == "yes"]
        allow_cal = [p for p, r in zip(p_cal, results) if r["label"] == "no"]
        for r, p in zip(results, p_cal):
            r["p_yes_calibrated"] = round(p, 4)
        calibrated = {"temperature": round(T, 4), "nll": round(nll, 4),
                      **summarize(deny_cal, allow_cal)}

    lats = sorted(r["latency"] for r in results)
    pct = lambda p: lats[min(int(len(lats) * p), len(lats) - 1)]
    summary = {
        "n": len(results), "mode": cfg.get("mode"), "model": cfg.get("base_model"),
        "int8": args.int8, "device": args.device,
        "latency_p50": round(pct(0.5), 3), "latency_p95": round(pct(0.95), 3),
        "raw": raw, "calibrated": calibrated,
        "及格线": {"deny召回": ">=0.80", "误拦": "<=0.05", "可分度": ">=0.35",
                    "AUC": ">=0.93", "ECE(calibrated)": "<=0.15 参考",
                    "p50": "<=3s（encoder 线目标）"},
        "note": ("阈值判定以 calibrated 概率为准；AUC/排序与温度无关；"
                 "延迟为逐条 batch=1 实测（含 int8 量化时为量化后）"),
    }
    print("\n=== 验证汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = Path(args.ckpt) / "holdout_eval.json"
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=1))
    if T is not None:
        (Path(args.ckpt) / "temperature.json").write_text(
            json.dumps({"temperature": round(T, 4), "nll": round(nll, 4),
                        "mode": cfg.get("mode"), "n": len(results)}, indent=2))
        print(f"temperature -> {Path(args.ckpt) / 'temperature.json'}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
