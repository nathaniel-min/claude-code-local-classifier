# 复刻指南：从数据到本地安全分类器

用微调 Qwen3.5-4B 做 Claude Code 的本地安全分类器。整条链路：数据 → 训练 → 验证 → 量化 → 部署 → 接入。每一步含命令、参数、耗时和思路。

```
数据 → 训练(RLCD+LoRA) → 验证(holdout) → merge → GGUF f16 → 量化(Q4_K_M) → llama-server → 路由代理
```

---

## 1. 数据

**格式**：chatml 三段式 prompt（由 `eval/prompt_assembly.py` 统一组装）：
`policy_summary`（system）+ `<transcript>`（会话历史）+ 待判定 action + 裁决指令，assistant 头后跟一个 token（`yes`=deny / `no`=allow）。

**来源**：合成 RLCD case + 真实会话（敏感，不入库，见 `datasets/README.md` 声明）。仓库只放 `datasets/samples.jsonl` 6 条合成样本演示格式。

**为什么这么做**：三段式 prompt 是训练/推理/验收的同一真源，保证逐字符一致。模型只学最后位置一个 token 的概率，不生成理由——推理读 logprob 当概率，阈值可调。

## 2. 训练

**命令**（完整细节见 `train/README-TRAIN.md`）：
```bash
pip install -r train/requirements-win.txt   # transformers>=5.17, torch>=2.10

python train/check_tokens.py --model Qwen/Qwen3.5-4B   # yes/no 必须单 token

python train/train_rlcd.py \
    --train datasets/train-v2/train.jsonl \
    --val datasets/train-v2/val.jsonl \
    --out output/adapter
```

**配置**：底座 `Qwen/Qwen3.5-4B`（多模态版，`get_text_config()` 剥视觉编码器只留语言部分）；QLoRA NF4 4bit，LoRA r=32/alpha=64，全线性层；seq_len 8192；batch 1 × grad_accum 16；lr 2e-4 cosine；epochs 3；label_smoothing 0.05。

**数据量**：train 3,295 条（yes 1,308）/ val 109 / holdout-test 298（GT-deny 28 条 + deny_human 84 + allow 186）。

**耗时**：约 11 小时（T4 级，估算值；魔搭社区 24G 显存 / 36h 配额内）。训练日志在魔搭端，未回传本地，实际耗时以魔搭端为准。训练速度曾出现三连跳（490→208→78.6s/步），是环境修复过程中的波动。

**为什么这么做**：
- **Qwen3.5-4B**：Jev 本地复刻项目（SemIf）裸模型即达决策 0.813/对齐 0.845，4B 级判断力足够
- **LoRA 而非全量**：省显存（170M adapter vs 全量）、易发布、可替换
- **RLCD 而非生成式**：分类任务只需一个 token 的概率，不生成 reason，快且无幻觉

## 3. 验证

```bash
python train/eval_adapter.py --adapter output/adapter \
    --holdout datasets/train-v2/holdout-test.jsonl
```

及格线（对照 `train/README-TRAIN.md` §4）：deny 召回 ≥0.80 / 误拦 ≤5% / sep ≥0.35 / auc ≥0.93。
不达标先看 mean_p_yes 分布是否可分；完全不可分 → 训练问题（看 loss 收敛）。

**裸基线对照**：裸 Qwen3.5-4B auc 0.423（不可分）→ 微调后 auc 0.979，证明微调必要。

## 4. 量化

**命令**（`deploy/merge_quantize.py`，CPU 端跑）：
```bash
python3 merge_quantize.py --adapter output/adapter --out models/qwen35-clf-q4_k_m.gguf
# 内部：load_causal_model(剥视觉) → PeftModel.merge_and_unload → convert_hf_to_gguf.py --outtype f16 → llama-quantize Q4_K_M
```

**耗时**：
- merge（LoRA→HF 全量）：约 1.5 分钟（本地 CPU 实测：base 7s + adapter 8s + merge 41s + save 48s）
- convert f16 / quantize Q4_K_M：未精确计时（服务器 CPU 端）

**产物**：f16 GGUF ~8.4G → Q4_K_M ~2.5G（BPW 5.13，省约 3 倍磁盘/内存）。

**为什么这么做**：
- **先 f16 再量化**：f16 是无损中转格式，稳定中间态
- **Q4_K_M**：4-bit 主量 + 高重要分量保精度，量化后指标与 f16 训练环境几乎一致（deny_recall 0.866 ≈ 0.884），概率判定不因量化失真
- **⚠️ 不带 `--no-nextn`**：本模型无 MTP 层，带 `--no-nextn` 会改变 GGUF 元数据处理 → llama.cpp 推理 logits 系统性压缩（deny p_yes 0.85→0.47，deny_recall 0.214）。以 `deploy/merge_quantize.py` 为准。

## 5. 部署

**启动**（`deploy/start_server.sh`）：
```bash
llama-server -m qwen35-clf-v2-q4_k_m.gguf --host 127.0.0.1 --port 11943 \
  -c 16384 --parallel 1 -ngl 999 -ctk q8_0 -ctv q8_0
```

**调用**（`/completion` 读概率，不读生成文本）：
```bash
curl 127.0.0.1:11943/completion -H 'Content-Type: application/json' -d '{
  "prompt": "<|im_start|>system\n{policy_summary}<|im_end|>\n<|im_start|>user\n<transcript>...{action}</transcript>...<|im_end|>\n<|im_start|>assistant\n",
  "n_predict": 1, "n_probs": 60, "temperature": 0, "cache_prompt": true}'
# 取 completion_probabilities[0].top_logprobs 的 yes/no logprob → softmax → p_yes
# block 判定：p_yes >= 0.5（阈值可调）
```

**为什么走 `/completion`**：Qwen3.5 是思考型模型，llama-server 里会无限输出 thinking 循环，`--reasoning off`/`enable_thinking:false` 均无效。绕开方式 = 读 logprob 不读生成文本——即使 argmax 是 thinking token，yes/no 的 logit 信号仍在 top-N 里。

**端口架构**（勿混）：
- **11940 = 路由代理入口**：Claude Code 连这里（Anthropic API + 哨兵判定 + 路由透传）
- **11943 = llama-server**：只给代理内部 `/completion` 用，不暴露给 Claude Code（无 /v1/messages、不路由、不跑主对话模型）

## 6. 接入路由代理

```bash
cp router/config.example.json router/config.json  # 填你的网关 URL
python router/server.py                            # 监听 11940
```

代理行为：哨兵路径（请求 model=分类器）→ 摊平成训练同构 chatml → 调本地 `/completion` 读概率 → 合成 `<block>yes/no</block>`；其他模型按前缀透传上游网关。

## 7. 常见坑

| 坑 | 现象 | 解法 |
|---|---|---|
| `--no-nextn` | deny p_yes 0.85→0.47，deny_recall 0.214 | convert 不带 `--no-nextn` |
| policy 失配 | 哨兵 system 139K 完整 rules 送本地，概率失真 | 用固定 `policy/policy_summary.txt`（15K）作 system，不取请求体 |
| 截断丢开标签 | 长 transcript 截尾后不以 `<transcript>` 开头，不同构 | 截断后补回 `<transcript>\n[...truncated...]\n` + 尾部 |
| thinking 循环 | 输出 thinking 循环，关不掉 | 走 `/completion` 读概率，不读文本 |
| yes/no 变体 | top_logprobs token 含前导空格/大小写 | `.strip().lower()` 后精确匹配 |

## 8. 指标（298 holdout）

| 指标 | 本地 v2 (RTX 4050 GPU) | 服务器纯 CPU | transformers 训练环境 |
|---|---|---|---|
| deny_recall@0.5 | 0.866 | 0.866 | 0.884 |
| false_block@0.5 | 0.005 | 0.005 | 0.005 |
| auc | 0.945 | — | 0.979 |
| sep | 0.799 | 0.726 | 0.819 |
| latency | 1.8s/条 | p50 33ms（KV 命中后） | — |

> 延迟口径说明：`1.8s/条` 是本地 GPU 单条完整判定（含 prompt eval + 1 token）；服务器 `p50 33ms` 是 KV 缓存命中后的增量请求；未命中冷请求约 13s。

## 检查清单

1. [ ] `check_tokens.py` 确认 yes/no 单 token（多 token 会导致 RLCD loss 算错）
2. [ ] 训练达标（deny 召回 ≥0.80 / 误拦 ≤5% / auc ≥0.93）
3. [ ] `merge_quantize.py` 不带 `--no-nextn`
4. [ ] llama-server 起在 11943，health ok
5. [ ] `bench/jev_replay_v2.py` 验收（对照基线 `bench/results/jev_v2_20260924-022529.json`）
