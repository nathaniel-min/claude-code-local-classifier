#!/usr/bin/env python3
"""check_tokens.py — 训练前断言 yes/no 单 token 化（防止 RLCD loss 算错）。

用法: python check_tokens.py [--model Qwen/Qwen3.5-4B]
"""
import argparse
import sys

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for w in ("yes", "no", "Yes", "No"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        status = "OK" if len(ids) == 1 else "FAIL(multi-token)"
        print(f"{w!r}: ids={ids} tokens={toks} [{status}]")
    if any(len(tok(w, add_special_tokens=False)["input_ids"]) != 1 for w in ("yes", "no")):
        print("ERROR: yes/no 非单 token，RLCD loss 会算错。换底座或调校 tokenizer。")
        sys.exit(1)
    # SemIf 式 round-trip 断言：encode(prompt + slot) == encode(prompt) + [slot_token]
    # 防止答案边界改变分词（例如 slot 前有 <|im_start|>assistant\n 时 yes 被合并）
    test_prompt = "<|im_start|>assistant\n"
    for w in ("yes", "no"):
        slot = tok(w, add_special_tokens=False)["input_ids"][0]
        base = tok(test_prompt, add_special_tokens=False)["input_ids"]
        full = tok(test_prompt + w, add_special_tokens=False)["input_ids"]
        rt_ok = full == base + [slot]
        print(f"round-trip {w!r} after assistant header: "
              f"{'OK' if rt_ok else 'FAIL'}")
        if not rt_ok:
            print("ERROR: 答案边界改变分词，RLCD 位置会错。检查 chatml 头与 target 拼接。")
            sys.exit(1)
    print("检查通过：yes/no 单 token + round-trip 精确，可开始训练。")


if __name__ == "__main__":
    sys.exit(main())
