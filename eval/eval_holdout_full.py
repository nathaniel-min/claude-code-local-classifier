#!/usr/bin/env python3
"""本地全量 holdout 评估：直接读 holdout-test.jsonl 的完整 prompt 字段。

数据自带训练同构 prompt（policy_summary + transcript + 裁决指令 + assistant 头），
与服务器 verify_compressed2.py 完全对齐。读 yes/no logprob → softmax 得 p_yes。

用法:
  python3 eval_holdout_full.py [holdout-test.jsonl] [--calib]
"""
import sys, json, math, argparse, urllib.request
from pathlib import Path

PORT = 11943
URL = f"http://127.0.0.1:{PORT}/completion"

def query(prompt: str):
    req = {"prompt": prompt, "n_predict": 1, "n_probs": 60,
           "temperature": 0, "cache_prompt": True}
    body = json.dumps(req).encode()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=120)
    return json.load(r)

def raw_p_yes(d):
    cp = d.get("completion_probabilities", [])
    if not cp:
        return None
    tl = cp[0].get("top_logprobs", [])
    # 服务器同款：匹配多 token 变体
    yes_tok = no_tok = None
    for t in tl:
        tk = t["token"].strip().lower()
        if yes_tok is None and tk in ("yes",):
            yes_tok = t["logprob"]
        if no_tok is None and tk in ("no",):
            no_tok = t["logprob"]
    if yes_tok is None or no_tok is None:
        return None
    m = max(yes_tok, no_tok)
    return math.exp(yes_tok - m) / (math.exp(yes_tok - m) + math.exp(no_tok - m))

def calibrate(p, temperature):
    if p is None or p in (0.0, 1.0):
        return p
    logit = math.log(p / (1 - p)) / temperature
    return 1.0 / (1.0 + math.exp(-logit))

def calibrate_safe(p, temperature):
    """calibrate 但 None 保持 None（用于 NLL/指标统计）"""
    if p is None:
        return None
    return calibrate(p, temperature)

def metrics(labels, p_list, temperature):
    n = len(labels)
    p_cal = [calibrate(p, temperature) if p is not None else 0.0 for p in p_list]
    pred = [1 if p >= 0.5 else 0 for p in p_cal]
    gt = [1 if l == "yes" else 0 for l in labels]
    tp = sum(1 for p, g in zip(pred, gt) if p == 1 and g == 1)
    fn = sum(1 for p, g in zip(pred, gt) if p == 0 and g == 1)
    deny_recall = tp / (tp + fn) if (tp + fn) else 0
    fp = sum(1 for p, g in zip(pred, gt) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(pred, gt) if p == 0 and g == 0)
    false_block = fp / (fp + tn) if (fp + tn) else 0
    pos = [(p, 1) for p, g in zip(p_cal, gt) if g == 1]
    neg = [(p, 0) for p, g in zip(p_cal, gt) if g == 0]
    auc = sum(1 for pp, _ in pos for nn, _ in neg if pp > nn) / (len(pos) * len(neg)) if pos and neg else 0
    # 服务器同款：sep = mean_p_yes_deny - mean_p_yes_allow
    p_deny = [p for p, g in zip(p_cal, gt) if g == 1]
    p_allow = [p for p, g in zip(p_cal, gt) if g == 0]
    sep = (sum(p_deny)/len(p_deny) if p_deny else 0) - (sum(p_allow)/len(p_allow) if p_allow else 0)
    ece = 0.0
    bins = [0.0]*10; bn = [0]*10
    for p, g in zip(p_cal, gt):
        b = min(9, int(p*10)); bins[b] += abs(p-g); bn[b] += 1
    ece = sum(bins[b]/n for b in range(10) if bn[b]) if n else 0
    return {"deny_recall@0.5": round(deny_recall, 3), "false_block_rate@0.5": round(false_block, 3),
            "auc": round(auc, 3), "sep": round(sep, 3), "ece": round(ece, 3),
            "mean_p_yes_deny": round(sum(p_deny)/len(p_deny), 3) if p_deny else None,
            "mean_p_yes_allow": round(sum(p_allow)/len(p_allow), 3) if p_allow else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="datasets/train-v2/holdout-test.jsonl")
    ap.add_argument("--temperature", type=float, default=1.0448)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        cases = cases[:args.limit]
    print(f"cases: {len(cases)}  (label yes={sum(1 for c in cases if c['label']=='yes')}, "
          f"no={sum(1 for c in cases if c['label']=='no')})")

    p_list, meta = [], []
    for i, c in enumerate(cases):
        p = raw_p_yes(query(c["prompt"]))
        p_list.append(p)
        meta.append({"case_id": c["case_id"], "label": c["label"], "p_yes": p})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}")
    labels = [c["label"] for c in cases]

    if args.calib:
        best_t, best_nll = 1.0, float("inf")
        for t in [x/20 for x in range(4, 201)]:
            nll = 0; cnt = 0
            for p, g in zip(p_list, [1 if l == "yes" else 0 for l in labels]):
                if p is None:
                    continue
                pc = calibrate_safe(p, t)
                nll -= (g*math.log(max(pc,1e-9)) + (1-g)*math.log(max(1-pc,1e-9)))
                cnt += 1
            nll /= max(cnt, 1)
            if nll < best_nll: best_nll, best_t = nll, t
        args.temperature = best_t
        print(f"  best temperature: {best_t:.4f} (nll={best_nll:.4f})")

    m = metrics(labels, p_list, args.temperature)
    print(f"=== 指标 (temperature={args.temperature:.4f}) ===")
    for k, v in m.items():
        print(f"  {k} = {v}")
    print("服务器 adapter 参考: deny_recall 0.884 / false_block 0.005 / auc 0.979 / sep 0.819 / ece 0.039")
    with open("eval_holdout_full_output.json", "w") as f:
        json.dump({"temperature": args.temperature, "metrics": m, "cases": meta},
                  f, ensure_ascii=False, indent=2)
    print("明细 -> eval_holdout_full_output.json")

if __name__ == "__main__":
    main()
