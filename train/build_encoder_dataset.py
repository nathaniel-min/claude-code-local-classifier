#!/usr/bin/env python3
"""build_encoder_dataset.py — train-v2（chatml 烧死）→ encoder 纯文本数据集。

机械转换：去掉 3 个 chatml 标记（真源 eval/prompt_assembly.py 的 CHATML_*
常量，此处硬编码保持包内自包含），label/weight/case_id/session 原样保留，
prompt 字段改存 text。与 Qwen 线逐字符同信息——两线对照实验公平性的关键。

输出：datasets/train-v2-enc/{train,val,holdout-test}.jsonl
  每行: {"text": 纯文本(policy+transcript+action+judgement), "label", "weight",
         "case_id", "session", "user", "split", "tool", "input"}

可选 --max-chars：Phase A 延迟闸门要求更短输入时二次裁剪（保 policy 头 +
尾部 action/judgement，裁 transcript 中段——与 build_prompt_for_case 同款逻辑）。

用法（包根目录）：
  python train/build_encoder_dataset.py                 # 原长转换
  python train/build_encoder_dataset.py --max-chars 8000  # 二次裁剪
"""
import argparse
import json
import sys
from pathlib import Path

# 与 eval/prompt_assembly.py 逐字符对齐（真源在那里；魔搭包内自包含故硬编码）
CHATML_SYSTEM = "<|im_start|>system\n"
CHATML_USER = "\n<|im_end|>\n<|im_start|>user\n"
CHATML_ASSISTANT = "\n<|im_end|>\n<|im_start|>assistant\n"

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
IN_DIR = BASE / "datasets" / "train-v2"
OUT_DIR = BASE / "datasets" / "train-v2-enc"
SPLITS = ("train", "val", "holdout-test")


def to_encoder_text(prompt):
    """chatml 烧死 prompt → encoder 纯文本。全量 assert 每个标记恰出现一次。"""
    for marker in (CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT):
        n = prompt.count(marker)
        assert n == 1, f"chatml 标记出现 {n} 次（应为 1）：{marker!r}"
    return (prompt
            .replace(CHATML_SYSTEM, "")
            .replace(CHATML_USER, "\n\n")
            .replace(CHATML_ASSISTANT, ""))


def trim_middle(text, max_chars):
    """超长时裁 <transcript> 中段（保 policy 头 + 尾部 action/judgement）。"""
    if len(text) <= max_chars:
        return text
    marker = "<transcript>\n"
    mid_start = text.index(marker) + len(marker)
    mid_end = text.index("\n</transcript>", mid_start)
    head, tail = text[:mid_start], text[mid_end:]
    lines = text[mid_start:mid_end].split("\n")
    keep, used = [], 0
    for ln in reversed(lines):
        if used + len(ln) + 1 > max_chars - len(head) - len(tail):
            break
        keep.append(ln)
        used += len(ln) + 1
    return head + "\n".join(reversed(keep)) + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(IN_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-chars", type=int, default=0,
                    help=">0 时二次裁剪到该字符预算（Phase A 工作点）")
    ap.add_argument("--swap-policy", default=None,
                    help="用该 policy 文件替换 prompt 中烧死的原 policy（encoder 专用蒸馏版），"
                         "原文件为 eval/policy_summary.txt")
    args = ap.parse_args()

    orig_policy = swap_policy = None
    if args.swap_policy:
        orig_policy = (BASE / "eval" / "policy_summary.txt").read_text(encoding="utf-8")
        swap_policy = Path(args.swap_policy).read_text(encoding="utf-8")
        print(f"policy 替换: {len(orig_policy)} -> {len(swap_policy)} 字符", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trimmed = 0
    for split in SPLITS:
        src = Path(args.in_dir) / f"{split}.jsonl"
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_rows = []
        for r in rows:
            prompt = r["prompt"]
            if swap_policy is not None:
                assert prompt.count(orig_policy) == 1, \
                    f"case {r.get('case_id')}: 原 policy 未恰出现一次"
                prompt = prompt.replace(orig_policy, swap_policy)
            text = to_encoder_text(prompt)
            if args.max_chars:
                t = trim_middle(text, args.max_chars)
                if t != text:
                    n_trimmed += 1
                text = t
            out_rows.append({
                "text": text,
                "label": r["label"], "weight": r.get("weight", 1.0),
                "case_id": r.get("case_id"), "session": r.get("session"),
                "user": r.get("user"), "split": split,
                "tool": r.get("tool"), "input": r.get("input", {}),
            })
        dst = out_dir / f"{split}.jsonl"
        dst.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in out_rows),
                       encoding="utf-8")
        n_yes = sum(1 for i in out_rows if i["label"] == "yes")
        lens = sorted(len(i["text"]) for i in out_rows)
        print(f"{split}: {len(out_rows)} (yes={n_yes} no={len(out_rows)-n_yes}) "
              f"字符长 p50={lens[len(lens)//2]} max={lens[-1]} -> {dst}")
    if args.max_chars:
        print(f"二次裁剪: {n_trimmed} 条")
    print("转换完成。校验：标记断言已全量通过（每行 3 个标记各恰 1 次）")


if __name__ == "__main__":
    sys.exit(main())
