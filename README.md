# claude-code-local-classifier（基于 Jev 思路）

用微调 Qwen3.5-4B 做 Claude Code 的本地安全分类器（prompt injection 哨兵）。数据不出域、阈值概率可调、本地 GPU 推理。

**思路来源：Jev**（TypeSafe AI 的快速推理模型）——不生成文本、只做 `true/false` 二分类决策。本模型沿此原理：每次只回答"这个动作该不该拦"（yes/no 概率），不生成理由，快且无幻觉。

## 为什么

Claude Code 的安全哨兵默认走云端 haiku——费钱且会话数据出域。本项目微调一个 4B 小模型在本地做分类器：deny_recall 0.866（接近云端），数据不出域，单条 1.8s（GPU）。

## 怎么选出来的（简版）

完整尝试史见[选型博客](https://daily-ai-practice.vercel.app/posts/从云端哨兵到本地分类器-为什么试了那么多模型/)。简表：

| 阶段 | 尝试 | 为什么放弃/保留 |
|---|---|---|
| 1 | 云端 haiku 哨兵 | 费钱 + 数据出域 → 本地化 |
| 2 | 现成小模型 / 云端 fallback | 全拦/全放/格式失败 + deny 召回 0 → 不可靠 |
| 3 | encoder 路线（gte/bge-m3） | 判断力随机 + transformers 版本问题 → 弃用 |
| 4 | Qwen3-4B-Instruct-2507（v1） | 正样本太少，判断力不足 → 换底座 |
| 5 | **Qwen3.5-4B（v2，当前）** | Jev 本地复刻（SemIf）裸模型决策 0.813/对齐 0.845，4B 判断力足够 |

另：裸模型基线 auc 0.423（不可分）→ 证明必须微调。

## 指标（298 holdout）

| 指标 | 本地 v2 (RTX 4050) | 服务器纯 CPU | transformers 训练环境 |
|---|---|---|---|
| deny_recall@0.5 | 0.866 | 0.866 | 0.884 |
| false_block@0.5 | 0.005 | 0.005 | 0.005 |
| auc | 0.945 | — | 0.979 |
| sep | 0.799 | 0.726 | 0.819 |
| latency | 1.8s/条 | p50 33ms¹ | — |

¹ 服务器 p50 33ms 是 KV 缓存命中后的增量请求；本地 1.8s/条是含 prompt eval 的完整判定。

## 架构

```
Claude Code ──► 11940 (classifier proxy: 哨兵判定 + 路由透传)
                    │ 哨兵 model=ruvltra-classifier
                    ▼
                 11943 (llama-server: /completion 读 yes/no logprob)
```

- **11940 = classifier proxy 入口**：收 Anthropic Messages API；哨兵(model=ruvltra-classifier)走本地概率判定；其他 model 按前缀透传上游网关。
- **11943 = llama-server**：只给 classifier 内部 `/completion` 概率判定用，不暴露给 Claude Code（不实现 /v1/messages、不路由、不跑主对话模型）。

## 仓库内容

| 目录 | 内容 |
|---|---|
| `train/` | 训练管线（Qwen3.5-4B + LoRA r=32，RLCD） |
| `eval/` | `prompt_assembly.py`(prompt 真源) + `eval_holdout_full.py` + `verify_local.py` |
| `deploy/` | `merge_quantize.py` + `start_server.sh` + `run_llama.sh` |
| `router/` | `server.py`(classifier proxy) + `config.example.json`(脱敏) |
| `bench/` | `jev_replay_v2.py`(验收回放) + `results/`(基线) |
| `policy/` | `policy_summary.txt`(训练 prompt 真源，15K) |
| `datasets/` | `samples.jsonl`(6 合成 case) + 格式说明 |

## 快速开始

```bash
# 1. 起 llama-server（11943，需 GGUF，见 HuggingFace 模型卡）
bash deploy/start_server.sh

# 2. 配 classifier proxy
cp router/config.example.json router/config.json  # 填你的网关 URL
python router/server.py                            # 监听 11940

# 3. 评测（smoke，用合成样本）
python eval/eval_holdout_full.py datasets/samples.jsonl
```

## 文档

- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — 复刻指南（数据→训练→量化→部署→踩坑，含耗时和思路）
- 📝 技术博客（Jev 思路 + 训练 + 踩坑 + 性能）：https://daily-ai-practice.vercel.app/posts/微调一个-4b-模型当本地安全分类器/
- 📝 选型博客（尝试史：为什么试了那么多模型）：https://daily-ai-practice.vercel.app/posts/从云端哨兵到本地分类器-为什么试了那么多模型/
- 🤗 HuggingFace 模型（LoRA adapter + GGUF + Model Card）：https://huggingface.co/43rt/claude-code-local-classifier

## 免责

- `datasets/samples.jsonl` 是合成通用 case，仅演示格式；原始 298 holdout 含真实会话历史，**不入库**。
- 本分类器是 prompt injection 辅助哨兵，不替代完整安全审查。
- 训练用的 policy_summary 是通用 AI 安全规则摘要；原始完整 rules 不随仓库发布。
