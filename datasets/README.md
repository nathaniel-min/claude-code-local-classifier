# Datasets

## 原始 holdout 数据（不入库）

298 条 holdout（`holdout-test.jsonl`）含真实 Claude Code 会话历史（用户名、内部项目路径、工具调用内容），属敏感数据，**不放入本仓库**。HuggingFace 模型卡同样不放原始会话。

## samples.jsonl（合成，演示格式）

6 条合成通用 case（3 危险 Bash + 3 安全命令），格式与训练/评测的 holdout 完全一致（由 `eval/prompt_assembly.py` 的 `build_prompt()` 生成），用于：

- 演示 prompt 格式（policy_summary + `<transcript>` + 裁决指令 chatml）
- 让 `eval/eval_holdout_full.py` 能 smoke 跑通链路

## jsonl 格式

每行一个 JSON 对象：

```json
{
  "case_id": "danger-rmrf",
  "label": "yes",
  "tool": "Bash",
  "prompt": "<|im_start|>system\n{policy_summary}<|im_end|>\n<|im_start|>user\n<transcript>\n{\"user\":\"...\"}\n{\"Bash\":{...}}\n</transcript>\n\n{裁决指令}<|im_end|>\n<|im_start|>assistant\n"
}
```

- `label`: `yes`=应拒绝(block) / `no`=应放行(allow)
- `prompt`: 完整 chatml，由 `build_prompt(case, policy, judgement)` 生成 = policy_summary 全文 + `<transcript>`(对话历史+action) + 裁决指令 + `<|im_start|>assistant\n`
- 模型在 assistant 头后预测下一个 token = `yes`(deny) / `no`(allow)

## 自建 holdout

如需复现完整评测，按上述格式自建 jsonl：每条含真实/合成 case 的 `prompt`（用 `build_prompt`）+ `label`。`case` 结构见 `eval/prompt_assembly.py` 的 `build_messages()`：`{"tool":..., "input":..., "transcript":[{"user":...}|{"tool":..., "input":...}]}`。
