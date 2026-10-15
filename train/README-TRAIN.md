# RLCD 微调训练指南（魔搭社区 24G GPU，36h）

微调 **Qwen3.5-4B**（多模态底座，纯文本部分原生 262K 上下文）做安全分类器：
根据安全策略 + 会话历史输出 yes(拦截)/no(放行) 的概率。训练产出 LoRA adapter
（~80MB），回服务器合并量化（llama.cpp 已支持 Qwen3.5 架构）。

**底座选型**：SemIf（Jev 本地复刻，2.2k star）用 Qwen3.5-4B 裸模型即达通用决策
0.813 / Jev 对齐 0.845 → 4B 级判断力足够。2026-09-20 拍板换掉 Qwen3-4B-Instruct-2507。

**Prompt 格式**：chatml（`<|im_start|>`/`<|im_end|>`），与 SemIf 对齐。模型在
`<|im_start|>assistant\n` 后预测第一个 token = yes/no。训练/推理共用
`eval/prompt_assembly.py` 真源，数据在 `datasets/train-v2/`。

## 0. 前置

- 魔搭社区 GPU 环境：24G 显存（T4/A10 级），36 小时配额
- Python 3.10+
- 数据：`datasets/train-v2/`（train.jsonl / val.jsonl / holdout-test.jsonl），
  由服务器 `build_train_dataset.py`（chatml 版）产出，prompt 字段烧死成品文本

## 1. 环境准备

```bash
pip install -r train/requirements-win.txt
# 含: torch==2.10, transformers==5.17（Qwen3.5 原生支持必需）, peft, bitsandbytes,
#     accelerate, datasets, sentencepiece
```

> 版本基准参考 SemIf requirements.txt。Qwen3.5 需 transformers>=5.17 +
> torch>=2.10（`Qwen3_5ForCausalLM` 原生模型）。魔搭若网络受限无法访问 HF：
> `export HF_ENDPOINT=https://hf-mirror.com` 或用 `modelscope download --model Qwen/Qwen3.5-4B`。

## 2. 检查 yes/no 分词

```bash
python train/check_tokens.py --model Qwen/Qwen3.5-4B
```

必须看到 `yes` 和 `no` 都 `[OK]`（单 token）+ round-trip 通过（`assistant\n` 后
精确拼接）。若 `FAIL(multi-token)` 或 round-trip FAIL，RLCD loss 会算错——不要硬跑。

## 3. 训练

```bash
python train/train_rlcd.py \
    --train datasets/train-v2/train.jsonl \
    --val datasets/train-v2/val.jsonl \
    --out output/adapter
```

默认参数（24G 显存 OK）：
- 底座 Qwen3.5-4B（`Qwen3_5ForCausalLM` + `config.get_text_config()` 剥视觉编码器）
- QLoRA NF4 4bit，LoRA r=32/alpha=64，全线性层
- seq_len 8192（chatml prompt 平均 ~5K token；24G 可试 16384）
- batch 1 × grad_accum 16，lr 2e-4 cosine，epochs 3，label_smoothing 0.05

预计耗时：3295 条 × 3 epochs，T4 级 ~11h（36h 配额内，余量够 2-3 轮重训）。

## 4. 训练后立即验证（魔搭端，不用回传服务器）

```bash
python train/eval_adapter.py --adapter output/adapter \
    --holdout datasets/train-v2/holdout-test.jsonl
```

输出 deny 召回 / 误拦率 / 可分度 / AUC（及格线：召回 ≥0.80、误拦 ≤5%、
可分度 ≥0.35、AUC ≥0.93）。结果存 `output/adapter/holdout_eval.json`。

- 先用 `--max-samples 50` 冒烟（~2 分钟），确认跑通再全量 298 条（~10 分钟）
- 不达标 → 先看 mean_p_yes 分布是否可分；可分但阈值不优 → 调 threshold；
  完全不可分 → 训练有问题（看 train loss 是否收敛）

## 5. 产出回传

`output/adapter/`：LoRA adapter + tokenizer + train_log.json + holdout_eval.json（~80MB）
把整个 `output/adapter/` 拷回服务器（scp/网盘均可）。

**GPU 端不要做 merge/量化**（服务器 CPU 端用 `merge_quantize.py` 做）。

## 5. 显存不足 / OOM 降级（24G 一般不会）

| 步骤 | 改法 |
|---|---|
| 1 | `--seq-len 6144` |
| 2 | `--lora-r 16 --lora-alpha 32` |
| 3 | `--batch 1 --grad-accum 8` |

## 6. 训练结束回传

1. `output/adapter/` 拷回服务器，放 `models/qwen3.5-4b-clf/`
2. 服务器 `python merge_quantize.py --adapter models/qwen3.5-4b-clf/`
3. `python bench/jev_replay_v2.py` 验收（对照 ACCEPTANCE-v1.md）

## 常见问题

- **transformers 版本不够**：Qwen3.5 需 5.17+（`Qwen3_5ForCausalLM` 原生）。
  报 `缺 Qwen3_5ForCausalLM` → 升级 transformers
- **HF 下载失败**：`export HF_ENDPOINT=https://hf-mirror.com` 或魔搭手动下载
- **bitsandbytes 装不上**：魔搭 CUDA 版 pip 源，用最新版
- **`weights` 传进模型报错**：已修复，collator 单独收 weights，不传给 forward
- **CPU 训练不推荐**：4B 无 GPU 不可行，只有魔搭 24G GPU 值得跑
