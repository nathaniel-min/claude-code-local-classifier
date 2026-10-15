#!/usr/bin/env python3
"""prompt_assembly.py — 全项目唯一 prompt 格式真源。

训练（魔搭）、验收回放（服务器 CPU）、未来服务化三端共用此模块，保证
prompt 格式完全一致。任何格式改动只改这里。

格式（Qwen3.5 chat template / chatml，SemIf 同款）：
  <|im_start|>system
  {policy_summary}
  <|im_end|>
  <|im_start|>user
  <transcript>
  {"user":"..."}
  {"Bash":{...}}
  ...
  {action_line}
  </transcript>

  {裁决指令（含 <block> 结尾）}
  <|im_end|>
  <|im_start|>assistant

模型在 assistant 头之后预测下一个 token = yes(deny)/no(allow)。logits 取
最后位置（eval 无 target 用 logits[:, -1]；训练序列含 target 用 logits[:, -2]）。

chatml 标签为确定性文本，服务器端手工组装；魔搭端 check_tokens.py 用
tokenizer.apply_chat_template() 验证逐字符一致。

用法（模块内）：
  from eval.prompt_assembly import build_prompt, load_policy
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
POLICY_FILE = HERE / "policy_summary.txt"
CAPTURE = BASE / "logs" / "classifier-capture" / "20260916-123255-40dad7.json"

# chatml 模板常量（与 Qwen3.5 tokenizer apply_chat_template 输出逐字符对齐）
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CHATML_SYSTEM = f"{IM_START}system\n"          # 后接 system 内容
CHATML_USER = f"\n{IM_END}\n{IM_START}user\n"  # 连接 system→user
CHATML_ASSISTANT = f"\n{IM_END}\n{IM_START}assistant\n"  # 尾部生成头

# 由 make_policy_summary.py 生成，训练/验收都读它（一次生成，全局固定）
_policy_cache = None


def load_policy(path=None):
    global _policy_cache
    if _policy_cache is None:
        _policy_cache = Path(path or POLICY_FILE).read_text()
    return _policy_cache


def load_judgement_instruction(capture=None):
    """取 capture 的 m1 最后一块（裁决指令，含 <block> 结尾）。"""
    cap = json.loads(Path(capture or CAPTURE).read_text())
    m1 = cap["messages"][1]["content"]
    return m1[-1]["text"]


def build_messages(case, policy, judgement):
    """组装 chatml messages（list[dict]），与 Qwen3.5 apply_chat_template 输入一致。

    返回 [{"role":"system","content":policy},
          {"role":"user","content": body}]，body 含 <transcript>+action+裁决指令(<block>)。
    """
    tool = case.get("tool", "Bash")
    inp = case.get("input") or {}
    action_line = json.dumps({tool: inp}, ensure_ascii=False)

    trans = case.get("transcript") or []
    lines = []
    for x in trans:
        if x.get("user"):
            lines.append(json.dumps({"user": x["user"]}, ensure_ascii=False))
        elif x.get("tool"):
            lines.append(json.dumps({x["tool"]: x["input"]}, ensure_ascii=False))
    if not lines:
        desc = inp.get("description") or "perform the requested action"
        lines.append(json.dumps({"user": desc}, ensure_ascii=False))

    body = ("<transcript>\n"
            + "\n".join(lines)
            + "\n" + action_line
            + "\n</transcript>\n\n"
            + judgement)
    return [{"role": "system", "content": policy},
            {"role": "user", "content": body}]


def build_prompt(case, policy=None, judgement=None, policy_path=None):
    """组装一条训练/推理 prompt（chatml 文本）。

    case: 含 transcript（list of {"user":str} 或 {"tool":name,"input":...}）
          和 tool/input（最后 action）。若无 transcript，用 description 行兜底。
    policy: 预加载的策略文本（传 policy_path 则惰性加载）
    """
    pol = policy if policy is not None else load_policy(policy_path)
    jud = judgement if judgement is not None else load_judgement_instruction()
    msgs = build_messages(case, pol, jud)
    return (CHATML_SYSTEM + msgs[0]["content"]
            + CHATML_USER + msgs[1]["content"]
            + CHATML_ASSISTANT)


def build_answer(label, include_reason=False):
    """根据 label 生成期望答案（RLCD 目标：yes/no 单 token）。

    label: deny/deny_human -> yes（拦截），allow -> no（放行）
    include_reason: 是否带 category/reason（当前训练只学 yes/no 概率，不生成 reason）
    """
    if label in ("deny", "deny_human"):
        return "yes"
    return "no"


if __name__ == "__main__":
    # 自检：打印样例 prompt 结构（不出具体值）
    p = load_policy()
    j = load_judgement_instruction()
    print(f"policy: {len(p)} 字符  judgement: {len(j)} 字符")
    sample = {"tool": "Bash", "input": {"command": "example", "description": "d"},
              "transcript": [{"user": "hello"}, {"tool": "Read", "input": {"file_path": "/x"}}]}
    full = build_prompt(sample)
    print(f"示例 prompt 总长: {len(full)} 字符")
    print("结尾 120 字符:", full[-120:])
    assert full.endswith(CHATML_ASSISTANT), "prompt 必须以 assistant 生成头结尾"
    assert "<transcript>" in full and "</transcript>" in full, "transcript 块缺失"
    assert "<block>" in full, "<block> 裁决标记缺失"
    print("自检通过：chatml 格式正确")
