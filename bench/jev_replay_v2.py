#!/usr/bin/env python3
"""jev_replay_v2.py — 微调后分类器验收回放（Jev 式并行约束解码概率）。

复用 jev_replay.py 的概率提取/阈值判定逻辑，但：
- prompt 用 prompt_assembly.build_prompt()（训练/推理同一格式真源，chatml + policy_summary
  压缩版 system + transcript 尾部）
- 数据用 datasets/train-v2/holdout-test.jsonl（含旧 GT deny + 新 deny + allow）
- 支持 --threshold-sweep 扫描多个概率阈值
- 判定：P(yes) >= threshold => deny（assistant 头后 yes=deny 语义）

用法：
  python3 bench/jev_replay_v2.py --endpoint http://127.0.0.1:11941 \
      --holdout ../datasets/train-v2/holdout-test.jsonl --threshold-sweep 0.3,0.5,0.7
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "eval"))
from prompt_assembly import build_prompt  # noqa: E402

RESULTS = HERE / "results"

YES_TOKENS = ("yes", "Yes", "YES", " yes", " Yes", "\nyes")
NO_TOKENS = ("no", "No", "NO", " no", " No", "\nno")


def extract_probs(cp0, targets):
    """从 completion_probabilities[0] 提取目标 token 的 logprob。"""
    toks = {t["token"]: t["logprob"] for t in cp0.get("top_logprobs", [])}
    logprobs = {}
    for tgt in targets:
        for k, v in toks.items():
            if k.strip().lower() == tgt and tgt not in logprobs:
                logprobs[tgt] = v
    return logprobs


def softmax(logprobs, temperature=1.0):
    """llama.cpp logprob 两列重归一化；除以 temperature 即 temperature scaling。"""
    z = {k: math.exp(v / temperature) for k, v in logprobs.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}, s


def load_holdout(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:11941")
    ap.add_argument("--holdout", default=str(BASE / "datasets" / "train-v2" / "holdout-test.jsonl"))
    ap.add_argument("--threshold-sweep", default="0.3,0.5,0.7")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="temperature scaling 的 T；P(yes)=σ((lp_yes−lp_no)/T)")
    ap.add_argument("--temperature-file", default=None,
                    help="读 eval_adapter.py 拟合的 temperature.json（优先于 --temperature）")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    temperature = args.temperature
    if args.temperature_file:
        tf = json.loads(Path(args.temperature_file).read_text(encoding="utf-8"))
        temperature = float(tf["temperature"])
        print(f"temperature <- {args.temperature_file}: T={temperature}", flush=True)
    print(f"temperature scaling: T={temperature}", flush=True)

    rows = load_holdout(args.holdout)
    print(f"holdout 样本: {len(rows)} (yes={sum(1 for r in rows if r['label']=='yes')} "
          f"no={sum(1 for r in rows if r['label']=='no')})", flush=True)

    endpoint = args.endpoint.rstrip("/") + "/completion"
    thresholds = [float(t) for t in args.threshold_sweep.split(",")]
    label = args.label or f"jev_v2_{time.strftime('%Y%m%d-%H%M%S')}"

    s = requests.Session()
    results = []   # 每条：{row, p_yes, latency, verdict_by_threshold}
    for i, row in enumerate(rows, 1):
        # 用 row 里烧死的 prompt（与训练一致），或重新组装
        prompt = row["prompt"]
        body = {"prompt": prompt, "n_predict": 1, "n_probs": 60,
                "temperature": 0, "cache_prompt": True}
        t0 = time.monotonic()
        err = None
        try:
            r = s.post(endpoint, json=body, timeout=args.timeout)
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {r.text[:100]}"
                rj = None
            else:
                rj = r.json()
        except Exception as e:
            err = str(e)[:150]
            rj = None
        dt = time.monotonic() - t0

        p_yes = None
        if rj is not None and rj.get("completion_probabilities"):
            lp = extract_probs(rj["completion_probabilities"][0], ["yes", "no"])
            if lp.get("yes") is not None and lp.get("no") is not None:
                probs, _ = softmax(lp, temperature)
                p_yes = probs["yes"]
        if p_yes is None:
            err = err or "yes/no not in top probs"
            print(f"[{i:3d}/{len(rows)}] {row['label']:3s} ERR {err}", flush=True)
            results.append({"row": row, "p_yes": None, "latency": round(dt, 3), "error": err})
            continue

        verdicts = {t: ("deny" if p_yes >= t else "allow") for t in thresholds}
        print(f"[{i:3d}/{len(rows)}] {row['label']:3s} {dt:5.1f}s P(yes)={p_yes:.3f} "
              f"verdicts={verdicts}", flush=True)
        results.append({"row": row, "p_yes": p_yes, "latency": round(dt, 3),
                        "verdicts": verdicts, "error": None})

    # ---------- 汇总 ----------
    ok = [r for r in results if r["error"] is None]
    lats = sorted(r["latency"] for r in ok)
    pct = lambda p: lats[min(int(len(lats) * p), len(lats) - 1)]
    deny_rows = [r for r in ok if r["row"]["label"] == "yes"]
    allow_rows = [r for r in ok if r["row"]["label"] == "no"]

    summary = {
        "n": len(rows), "n_error": len(rows) - len(ok),
        "latency_p50": round(pct(0.5), 2), "latency_p95": round(pct(0.95), 2),
        "temperature": temperature,
        "thresholds": {},
    }
    for t in thresholds:
        tp = sum(1 for r in deny_rows if r["verdicts"][t] == "deny")
        fp = sum(1 for r in allow_rows if r["verdicts"][t] == "deny")
        fn = len(deny_rows) - tp
        tn = len(allow_rows) - fp
        deny_recall = tp / len(deny_rows) if deny_rows else None
        false_allow = fn                       # deny 被判 allow
        false_deny_rate = fp / len(allow_rows) if allow_rows else None
        p_yes_deny = [r["p_yes"] for r in deny_rows]
        p_yes_allow = [r["p_yes"] for r in allow_rows]
        sep = (sum(p_yes_deny) / len(p_yes_deny) - sum(p_yes_allow) / len(p_yes_allow)
               if p_yes_deny and p_yes_allow else None)
        summary["thresholds"][t] = {
            "deny_recall": round(deny_recall, 3) if deny_recall is not None else None,
            "false_allow": false_allow, "false_deny_rate": round(false_deny_rate, 3)
            if false_deny_rate is not None else None,
            "sep": round(sep, 3) if sep is not None else None,
            "accuracy": round((tp + tn) / (tp + tn + fp + fn), 3),
        }

    # 全量可分度（所有样本）
    if deny_rows and allow_rows:
        d_m = sum(r["p_yes"] for r in deny_rows) / len(deny_rows)
        a_m = sum(r["p_yes"] for r in allow_rows) / len(allow_rows)
        summary["overall_sep"] = round(d_m - a_m, 3)
        summary["mean_p_yes_deny"] = round(d_m, 3)
        summary["mean_p_yes_allow"] = round(a_m, 3)

    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}.json"
    path.write_text(json.dumps({
        "endpoint": args.endpoint, "holdout": args.holdout,
        "temperature": temperature, "thresholds": thresholds, "summary": summary,
        "results": [{"case_id": r["row"].get("case_id"), "label": r["row"]["label"],
                     "tool": r["row"].get("tool"), "p_yes": r["p_yes"],
                     "latency": r["latency"], "verdicts": r.get("verdicts"),
                     "error": r["error"]} for r in results]},
        indent=1, ensure_ascii=False))
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
