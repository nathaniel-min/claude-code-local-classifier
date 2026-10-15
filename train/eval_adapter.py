#!/usr/bin/env python3
"""eval_adapter.py — 魔搭端训练后快速验证（不回传服务器即可看效果）。

加载 base（可选 + LoRA adapter），对 holdout-test 逐条算 P(yes)（Jev 式并行约束解码：
prompt 以 <block> 结尾，取最后位置 logits 的 yes/no 两列 softmax）。

两种模式：
  - 带 --adapter：评测微调后模型，并做 temperature fitting（温度拟合），写
    temperature.json 进 adapter 目录
  - 不带 --adapter：frozen baseline（冻结基线，底座模型不训练直接评测），
    输出 holdout_eval_base.json + temperature_base.json 到当前目录。
    训练前后各跑一次即可对照"微调是增益还是 catastrophic forgetting"。

汇总指标（对照 bench/ACCEPTANCE-v1.md 及格线），raw=未校准，calibrated=temperature 校准后：
  - deny 召回 @0.5（及格线 ≥0.80）
  - 误拦率 allow→deny（及格线 ≤5%）
  - 概率可分度 mean P(yes|deny) − mean P(yes|allow)（及格线 ≥0.35）
  - AUC（及格线 ≥0.93；单调变换不变，raw/calibrated 相同）
  - ECE（Expected Calibration Error，10-bin 等频；calibrated 参考线 ≤0.15）

用法（魔搭）：
  # frozen baseline（训练前或训练后对照）
  python train/eval_adapter.py --holdout datasets/train-v2/holdout-test.jsonl

  # 训练完成后评 adapter + 拟合 temperature
  python train/eval_adapter.py --adapter output/adapter \
      --holdout datasets/train-v2/holdout-test.jsonl
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from metrics import fit_temperature, summarize  # noqa: E402  同目录共享指标模块

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-4B"


def load_causal_model(source, **kw):
    """Qwen3.5 多模态需剥视觉编码器（SemIf 同款），只加载语言部分。"""
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    cls = AutoModelForCausalLM
    if getattr(config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(__import__("transformers"), "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("transformers 版本过低，缺 Qwen3_5ForCausalLM（需 >=5.17）")
        config = config.get_text_config()
    kw.setdefault("trust_remote_code", True)
    return cls.from_pretrained(source, config=config, **kw)


def check_tokens(tokenizer):
    yes_ids = tokenizer("yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("no", add_special_tokens=False)["input_ids"]
    assert len(yes_ids) == 1 and len(no_ids) == 1, f"yes/no 非单 token: {yes_ids} {no_ids}"
    return yes_ids[0], no_ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter 目录；不传 = 裸模型 frozen baseline")
    ap.add_argument("--holdout", default="datasets/train-v2/holdout-test.jsonl")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-samples", type=int, default=0, help=">0 只评前 N 条（快速冒烟）")
    ap.add_argument("--load-4bit", action="store_true", help="NF4 加载（省显存，稍慢）")
    args = ap.parse_args()

    mode = "adapter" if args.adapter else "base"
    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"mode={mode}  holdout: {len(rows)} (yes={sum(1 for r in rows if r['label']=='yes')} "
          f"no={sum(1 for r in rows if r['label']=='no')})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = check_tokens(tokenizer)
    print(f"yes_id={yes_id} no_id={no_id}", flush=True)

    kw = {"torch_dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = load_causal_model(args.model, device_map="auto", **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    results = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            ids = tokenizer(r["prompt"], add_special_tokens=False, return_tensors="pt")
            ids = {k: v.to(model.device) for k, v in ids.items()}
            logits = model(**ids).logits[0, -1]           # <block> 位置的下一 token 预测
            two = logits[[yes_id, no_id]].float()
            diff = (two[0] - two[1]).item()               # z_yes − z_no（存原始 logit 差）
            p_yes = 1.0 / (1.0 + math.exp(-diff))
            results.append({"case_id": r.get("case_id"), "label": r["label"],
                            "p_yes": round(p_yes, 4), "logit_diff": round(diff, 4)})
            if i % 25 == 0:
                dt = time.monotonic() - t0
                print(f"  {i}/{len(rows)}  {dt/i:.2f}s/条", flush=True)

    # ---------- raw 指标 + temperature fitting ----------
    diffs = [r["logit_diff"] for r in results]
    ys = [1 if r["label"] == "yes" else 0 for r in results]
    deny_p_raw = [r["p_yes"] for r in results if r["label"] == "yes"]
    allow_p_raw = [r["p_yes"] for r in results if r["label"] == "no"]
    raw = summarize(deny_p_raw, allow_p_raw)

    T, nll = fit_temperature(diffs, ys)
    calibrated = None
    if T is not None:
        p_cal = [1.0 / (1.0 + math.exp(-d / T)) for d in diffs]
        deny_p_cal = [p for p, r in zip(p_cal, results) if r["label"] == "yes"]
        allow_p_cal = [p for p, r in zip(p_cal, results) if r["label"] == "no"]
        for r, p in zip(results, p_cal):
            r["p_yes_calibrated"] = round(p, 4)
        calibrated = {"temperature": round(T, 4), "nll": round(nll, 4),
                      **summarize(deny_p_cal, allow_p_cal)}

    summary = {
        "n": len(results),
        "mode": mode,
        "model": args.model,
        "raw": raw,
        "calibrated": calibrated,
        "及格线": {"deny召回": ">=0.80", "误拦": "<=0.05", "可分度": ">=0.35",
                    "AUC": ">=0.93", "ECE(calibrated)": "<=0.15 参考"},
        "note": ("temperature 在 holdout 上拟合（Laya/reflex 同款，非严格独立）；"
                 "AUC/排序与温度无关（单调变换）；微调各项指标须 >= frozen baseline"),
    }
    print("\n=== 验证汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---------- 输出 ----------
    if args.adapter:
        out = Path(args.adapter) / "holdout_eval.json"
        tout = Path(args.adapter) / "temperature.json"
    else:
        out = Path("holdout_eval_base.json")
        tout = Path("temperature_base.json")
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=1))
    if T is not None:
        tout.write_text(json.dumps({"temperature": round(T, 4), "nll": round(nll, 4),
                                    "mode": mode, "n": len(results)}, indent=2))
        print(f"temperature -> {tout}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
