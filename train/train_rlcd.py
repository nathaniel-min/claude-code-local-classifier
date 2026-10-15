#!/usr/bin/env python3
"""train_rlcd.py — 概率校准微调（RLCD）训练入口。

目标：让 Qwen 学会"给定 system 安全策略 + 会话 transcript 尾部，输出
yes(拦截)/no(放行) 的概率"。不用完整 next-token 生成 loss，只优化最后一个位置
yes/no 两个 token 上的 softmax 交叉熵（Jev 式并行约束解码的对齐训练）。

配置（魔搭社区 16G GPU / 100h，QLoRA NF4 4bit）：
  - 底座: Qwen/Qwen3.5-4B（HF 自动下载，原生 262K 上下文，纯文本）
  - LoRA: r=32 / alpha=64，全线性层，dropout 0.1
  - bf16 + gradient_checkpointing + paged_adamw_8bit
  - batch=1 × grad_accum=16，lr 2e-4 cosine，epochs 3，label_smoothing 0.05
  - seq_len 8192（16G 显存够；原生 262K 上下文无截断压力；OOM 降级 6144/4096）

数据：build_train_dataset.py 产物 datasets/train-v2/{train,val}.jsonl
  每行 {"prompt": 烧死成品文本, "label": "yes"/"no", "weight": float}

用法（魔搭/GPU）：
  pip install -r requirements-win.txt
  python train_rlcd.py --train datasets/train-v2/train.jsonl \
      --val datasets/train-v2/val.jsonl --out output/adapter

产出：output/adapter/（LoRA adapter + trainer_state）+ train_log.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

HERE = Path(__file__).resolve().parent
# 2026-09-20 升级：魔搭社区 24G GPU + 36h → Qwen3.5-4B（SemIf 验证过的最强底座）
MODEL_ID = "Qwen/Qwen3.5-4B"


def load_causal_model(source, quant_config, device_map="auto", trust_remote_code=True):
    """加载底座（Qwen3.5 多模态需剥视觉编码器，SemIf 同款）。

    Qwen3.5-4B 是 Image-Text-to-Text 混合架构，config.model_type in {"qwen3_5",
    "qwen3_5_text"}。用 config.get_text_config() 拿纯文本子配置，配
    Qwen3_5ForCausalLM 只加载语言部分（省显存，推理时 logits 位置与纯文本一致）。
    """
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    cls = AutoModelForCausalLM
    if getattr(config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(__import__("transformers"), "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("transformers 版本过低，缺 Qwen3_5ForCausalLM（需 >=5.17）")
        config = config.get_text_config()
    return cls.from_pretrained(
        source, config=config, quantization_config=quant_config,
        device_map=device_map, trust_remote_code=trust_remote_code,
    )


# ---------- token 检查 ----------
def check_tokens(tokenizer):
    """断言 yes/no 单 token 化 + chatml 答案边界 round-trip 精确。"""
    yes_ids = tokenizer("yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("no", add_special_tokens=False)["input_ids"]
    assert len(yes_ids) == 1, f"yes 非单 token: {yes_ids}"
    assert len(no_ids) == 1, f"no 非单 token: {no_ids}"
    yes_tok = tokenizer.convert_ids_to_tokens(yes_ids[0])
    no_tok = tokenizer.convert_ids_to_tokens(no_ids[0])
    print(f"yes -> token {yes_tok!r} (id {yes_ids[0]}), no -> token {no_tok!r} (id {no_ids[0]})")
    # SemIf 式 round-trip：<|im_start|>assistant\n 后 yes/no 不得合并/分裂
    for w, tid in (("yes", yes_ids[0]), ("no", no_ids[0])):
        base = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        full = tokenizer("<|im_start|>assistant\n" + w, add_special_tokens=False)["input_ids"]
        assert full == base + [tid], (
            f"答案边界改变分词：{w!r} 在 assistant 头后 non-round-trip "
            f"({base + [tid]} != {full})")
    print("round-trip 通过：assistant 头后 yes/no 单 token 精确")
    return yes_ids[0], no_ids[0]


# ---------- RLCD 自定义 loss ----------
class RLCDTrainer(Trainer):
    """只优化最后位置的 yes/no 二分类 CE，乘以样本权重。"""

    def __init__(self, yes_id, no_id, label_smoothing=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_id = int(yes_id)
        self.no_id = int(no_id)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")          # (B,) sample weight
        outputs = model(**inputs)                # logits (B, S, V)（weights 已弹出）
        logits = outputs.logits
        # 序列 = [pad..., prompt..., <block>, target]（右对齐，全定长 seq_len）
        # causal LM: logits[i] 预测 token[i+1]。预测 target 需要 <block> 位置
        # 的输出 = logits[:, -2]（logits[:, -1] 是 target 之后的位置，错位！）
        last = logits[:, -2, :]                  # (B, V)
        # 目标 id：labels 最后一列（yes/no id 或 -100）
        tgt = labels[:, -1].clone()
        valid = tgt != -100
        if valid.sum() == 0:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

        # 只取 yes/no 两列的 logits
        sel = last[valid][:, [self.yes_id, self.no_id]]     # (B', 2)
        tgt_sel = tgt[valid]
        tgt_idx = torch.where(tgt_sel == self.yes_id,
                              torch.zeros_like(tgt_sel),
                              torch.ones_like(tgt_sel))     # 0=yes 1=no
        w = weights[valid] if weights is not None else torch.ones_like(tgt_sel, dtype=torch.float)
        # softmax CE（两列）
        ce = torch.nn.functional.cross_entropy(
            sel, tgt_idx, weight=None, reduction="none",
            label_smoothing=self.label_smoothing)
        loss = (ce * w).mean()
        if return_outputs:
            return (loss, outputs)
        return loss


# ---------- 自定义 collator（保留 labels 与 weights） ----------
@dataclass
class RLCDDataCollator:
    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # features 已按 seq_len 右对齐，直接取 input_ids/attention_mask/labels/weights
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            "weights": torch.tensor([f["weights"] for f in features], dtype=torch.float),
        }


# ---------- 数据加载 ----------
def load_jsonl(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def tokenize_rows(rows, tokenizer, seq_len, yes_id, no_id, max_chars=24000):
    """右对齐：保 policy 头部 + 尾部答案，裁 transcript 中段（字符预算）。

    与 eval/build_train_dataset.build_prompt_for_case 的 max_chars 截断逻辑对齐：
    从头截断会砍 policy 核心（HARD BLOCK 在 policy 中部），必须保头部 policy。

    序列 = [pad..., <|im_start|>system\n{policy}<|im_end|>\n<|im_start|>user\n
            <transcript>…(中段裁剪)…\n{action}\n</transcript>\n\n{judgement}
            <|im_end|>\n<|im_start|>assistant\n, target]
    labels 仅答案位置有效，其余标 -100（RLCD 只算最后位置）。
    """
    eos = tokenizer.eos_token_id
    out = []
    for r in rows:
        text = r["prompt"]
        label = r["label"]
        weight = float(r.get("weight", 1.0))
        target_id = yes_id if label == "yes" else no_id
        # 字符预算裁剪：<transcript> 中段向后裁，保 policy 头 + 尾部答案
        # prompt 由 build_train_dataset 已按 max_chars 烧死；此处按 seq_len 折算字符上限兜底
        budget = min(len(text), max_chars)
        if len(text) > budget:
            # 找 <transcript> 块边界，从后往前保留尾部行
            marker = "<transcript>\n"
            mid_start = text.index(marker) + len(marker)
            mid_end = text.index("\n</transcript>", mid_start)
            head = text[:mid_start]
            tail = text[mid_end:]
            lines = text[mid_start:mid_end].split("\n")
            keep, used = [], 0
            for ln in reversed(lines):
                if used + len(ln) + 1 > budget - len(head) - len(tail):
                    break
                keep.append(ln)
                used += len(ln) + 1
            text = head + "\n".join(reversed(keep)) + tail
        # token 化（不 pad）
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            # 极端兜底：仍超 seq_len，裁 token（保尾部答案）
            ids = ids[: seq_len - 1]
        n_real = len(ids)
        ids = ids + [target_id]                       # 答案 token
        n_total = n_real + 1
        # 右对齐：左侧 pad（pad 处 attention_mask=0）
        if n_total < seq_len:
            ids = [eos] * (seq_len - n_total) + ids
            labels = [-100] * (seq_len - n_total) + [-100] * (n_total - 1) + [target_id]
            attn = [0] * (seq_len - n_total) + [1] * n_total
        else:
            labels = [-100] * (n_total - 1) + [target_id]
            attn = [1] * n_total
        out.append({
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "weights": weight,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.jsonl 路径")
    ap.add_argument("--val", required=True, help="val.jsonl 路径")
    ap.add_argument("--out", default=str(HERE / "output" / "adapter"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0,
                    help=">0 只取前 N 条训练（CPU 冒烟用，验证 collator/loss 不报错）")
    args = ap.parse_args()

    # ---------- 底座 ----------
    print(f"加载底座 {args.model} ...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = check_tokens(tokenizer)
    model = load_causal_model(args.model, bnb)
    model = prepare_model_for_kbit_training(model)
    # LoRA：全线性层
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                      "up_proj", "down_proj"]
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=0.1,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # ---------- 数据 ----------
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
        val_rows = val_rows[:max(1, args.max_samples // 5)]
        print(f"[冒烟] 数据子集 train={len(train_rows)} val={len(val_rows)}", flush=True)
    print(f"train {len(train_rows)} / val {len(val_rows)}", flush=True)
    train_enc = tokenize_rows(train_rows, tokenizer, args.seq_len, yes_id, no_id)
    val_enc = tokenize_rows(val_rows, tokenizer, args.seq_len, yes_id, no_id)
    train_ds = Dataset.from_list(train_enc)
    val_ds = Dataset.from_list(val_enc)

    # ---------- 训练 ----------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch,
        eval_strategy="steps",   # transformers>=4.46 改名（旧名 evaluation_strategy 已移除）
        eval_steps=0.2,
        # save/eval 策略必须一致（load_best_model_at_end 要求，transformers>=4.46 强校验）
        save_strategy="steps", save_steps=0.2,
        logging_steps=20,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=42,
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = RLCDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=RLCDDataCollator(tokenizer=tokenizer),
        yes_id=yes_id,
        no_id=no_id,
        label_smoothing=args.label_smoothing,
    )
    trainer.train()

    # ---------- 保存 ----------
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log = {
        "model": args.model, "seq_len": args.seq_len, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "lr": args.lr, "epochs": args.epochs,
        "train_samples": len(train_rows), "val_samples": len(val_rows),
        "yes_token_id": yes_id, "no_token_id": no_id,
        "train_loss": trainer.state.log_history[-1].get("loss"),
    }
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print("训练完成 ->", out_dir, flush=True)


if __name__ == "__main__":
    sys.exit(main())
