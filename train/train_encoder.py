#!/usr/bin/env python3
"""train_encoder.py — encoder + decision head 二分类训练（B 计划副线）。

底座 gte-multilingual-base（303M，8192 上下文，多语言）+ 极简 decision head
（CLS 末层 hidden → dropout → linear(hidden, 2)）。吃 train-v2-enc 纯文本数据
（build_encoder_dataset.py 产物：蒸馏 policy + transcript，≤3.5K token 工作点）。

三档训练模式（encoder 路线的天然阶梯）：
  --mode head  frozen encoder 只训 head（几分钟；encoder 表征下限测试）
  --mode lora  peft LoRA r=32/alpha=64 全线性层 + head（默认档）
  --mode full  全参 + head（560M 在 24G 可行，备选）

loss：标准 CE（idx 约定 logits[:,1]=yes）× 样本权重 × label_smoothing。
与 train_rlcd.py 同风格（权重/超参/Trainer 骨架），但序列无自回归 target，
pad 到批内最长即可。

用法（魔搭 GPU）：
  python train/train_encoder.py --mode head \
      --train datasets/train-v2-enc/train.jsonl --val datasets/train-v2-enc/val.jsonl \
      --out output/encoder-head
  python train/train_encoder.py --mode lora ... --out output/encoder-lora

CPU 冒烟（几分钟）：
  python train/train_encoder.py --mode head --no-bf16 --cpu \
      --max-samples 8 --max-length 1024 --epochs 1 ...

产出：output/<name>/（head.pt + config.json + train_log.json；lora 另有 adapter/）
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn
from transformers import (
    AutoModel, AutoTokenizer, TrainingArguments, Trainer,
)

HERE = Path(__file__).resolve().parent
# Phase A 闸门实测（2026-09-21，bench/results/encoder_latency.json + threads 对照）：
# gte-multilingual-base 303M + int8 + threads=4：3072 tok=2.59s（bge-m3 560M 全面劣于 gte，
# 已弃用）。工作点 max_length=3500（token 实测 p50 3162 / max 3471，零截断全容纳，
# 内插延迟 p50≈2.7s / max≈3.06s，仍在 3s 闸门内；真实 p50 由 eval_encoder 实测终判）
MODEL_ID = "Alibaba-NLP/gte-multilingual-base"
TARGET_MODULES = ["query", "key", "value", "dense"]   # BERT/gte 系全线性层（peft 后缀匹配）


# ---------- 模型 ----------
class EncoderClassifier(nn.Module):
    """encoder + decision head。logits[:,1]=yes(deny) / logits[:,0]=no(allow)。"""

    def __init__(self, base, hidden, dropout=0.1):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # labels 形参仅为兼容 Trainer prediction_step（loss 在 WeightedTrainer.compute_loss）
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]        # XLM-R <s> token
        return self.head(cls)

    @property
    def config(self):
        return self.base.config


class WeightedTrainer(Trainer):
    """CE × 样本权重 × label_smoothing（与 train_rlcd.py 的 RLCDTrainer 同思路）。"""

    def __init__(self, label_smoothing=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")            # (B,) 0=no 1=yes
        weights = inputs.pop("weights")          # (B,)
        logits = model(**inputs)                 # (B, 2)
        ce = nn.functional.cross_entropy(
            logits.float(), labels, reduction="none",
            label_smoothing=self.label_smoothing)
        loss = (ce * weights).mean()
        return (loss, logits) if return_outputs else loss


@dataclass
class EncoderCollator:
    """pad 到批内最长 + labels/weights。"""

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 已分词，只做 pad（token id 列表必须走 tokenizer.pad，不能走 __call__）
        enc = self.tokenizer.pad({
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }, padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
            "weights": torch.tensor([f["weight"] for f in features], dtype=torch.float),
        }


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def tokenize_rows(rows, tokenizer, max_length):
    out = []
    for r in rows:
        enc = tokenizer(r["text"], truncation=True, max_length=max_length)
        out.append({"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "label": 1 if r["label"] == "yes" else 0,
                    "weight": float(r.get("weight", 1.0))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "lora", "full"], default="lora")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default=str(HERE / "output" / "encoder-lora"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-length", type=int, default=3500,
                    help="token 上限；Phase A 工作点 3500（见模块头注释）")
    ap.add_argument("--lr", type=float, default=None,
                    help="默认按 mode：head 1e-3 / lora 2e-4 / full 1e-5")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=None, help="默认 head 4 / 其余 2")
    ap.add_argument("--grad-accum", type=int, default=None, help="默认凑有效 batch 16")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0, help=">0 冒烟用")
    ap.add_argument("--cpu", action="store_true", help="CPU 模式（冒烟）")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    lr = args.lr or {"head": 1e-3, "lora": 2e-4, "full": 1e-5}[args.mode]
    batch = args.batch or (4 if args.mode == "head" else 2)
    grad_accum = args.grad_accum or (16 // batch)
    use_bf16 = (not args.no_bf16) and (not args.cpu)
    device_map = None if args.cpu else "auto"

    print(f"mode={args.mode} lr={lr} batch={batch}x{grad_accum} "
          f"bf16={use_bf16} cpu={args.cpu}", flush=True)

    # ---------- 底座 + head ----------
    print(f"加载底座 {args.model} ...", flush=True)
    base = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=device_map, trust_remote_code=True)
    hidden = base.config.hidden_size

    if args.mode == "head":
        for p in base.parameters():
            p.requires_grad = False
    elif args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        base = get_peft_model(base, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=TARGET_MODULES, lora_dropout=0.1,
            bias="none", task_type="FEATURE_EXTRACTION"))
    # full：什么都不做，全参训练

    model = EncoderClassifier(base, hidden)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_train/1e3:.1f}K / 总 {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)
    if args.mode != "head" and not args.cpu:
        model.base.gradient_checkpointing_enable()
        model.base.config.use_cache = False

    # ---------- 数据 ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
        val_rows = val_rows[:max(1, args.max_samples // 4)]
    print(f"train {len(train_rows)} / val {len(val_rows)}", flush=True)
    from datasets import Dataset
    train_ds = Dataset.from_list(tokenize_rows(train_rows, tokenizer, args.max_length))
    val_ds = Dataset.from_list(tokenize_rows(val_rows, tokenizer, args.max_length))

    # ---------- 训练 ----------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch,
        eval_strategy="steps", eval_steps=0.2,
        save_strategy="steps", save_steps=0.2, logging_steps=20,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=use_bf16, gradient_checkpointing=False,  # 已按模式手动开
        optim="adamw_torch" if args.cpu else "paged_adamw_8bit",
        max_grad_norm=1.0, seed=42, report_to=[],
        remove_unused_columns=False, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        use_cpu=args.cpu,
    )
    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=EncoderCollator(tokenizer=tokenizer),
        label_smoothing=args.label_smoothing,
    )
    trainer.train()

    # ---------- 保存：head.pt + config + (lora) adapter ----------
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    json.dump({"mode": args.mode, "base_model": args.model, "hidden": hidden,
               "max_length": args.max_length, "label_idx": {"no": 0, "yes": 1}},
              open(out_dir / "config.json", "w"), indent=2)
    if args.mode == "lora":
        model.base.save_pretrained(str(out_dir / "adapter"))
    log = {"mode": args.mode, "model": args.model, "lr": lr, "epochs": args.epochs,
           "max_length": args.max_length, "train_samples": len(train_rows),
           "train_loss": trainer.state.log_history[-1].get("loss")}
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print("训练完成 ->", out_dir, flush=True)


if __name__ == "__main__":
    sys.exit(main())
