#!/usr/bin/env python3
"""metrics.py — 分类器评测指标共享模块（Qwen 线 / encoder 线共用）。

从 eval_adapter.py 提取（2026-09-21），实现已过 15 项单测
（/tmp/test_temp_scaling.py，AST 提取方式）：
  - auc_mw：Mann-Whitney AUC（纯 Python，并列取平均秩）
  - fit_temperature / nll_temperature：temperature scaling（1D NLL 最小化，
    Laya/reflex 同款思路；纯数学无 scipy）
  - ece_equal_freq：等频 10-bin ECE（Expected Calibration Error）
  - summarize：一组 (deny_p, allow_p) 的指标包

使用方：train/eval_adapter.py（Qwen 线）、train/eval_encoder.py（encoder 线）。
魔搭包内 train/ 平铺，同目录直接 import。
"""
import math


def auc_mw(pos, neg):
    """Mann-Whitney AUC（纯 Python，并列取平均秩）。"""
    if not pos or not neg:
        return None
    vals = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    vals.sort(key=lambda x: x[0])
    # 平均秩（1-indexed）
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, g) in zip(ranks, vals) if g == 1)
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ---------- temperature fitting（生态共识：Laya / reflex 同款思路） ----------
def _log_sigma(z):
    """log σ(z)，数值稳定。"""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def nll_temperature(diffs, ys, T):
    """二分类 NLL：−mean[ y·logσ(d/T) + (1−y)·logσ(−d/T) ]，d=logit 差 z_yes−z_no。"""
    s = 0.0
    for d, y in zip(diffs, ys):
        z = d / T
        s += y * _log_sigma(z) + (1 - y) * _log_sigma(-z)
    return -s / len(diffs)


def fit_temperature(diffs, ys, lo=0.05, hi=10.0):
    """1D 最小化 NLL 找 T：log 均匀粗网格 200 点 + 邻域细化，纯数学无 scipy。"""
    if len(diffs) < 8:
        return None, None  # 样本太少，拟合无意义
    grid = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / 199)
            for i in range(200)]
    nlls = [nll_temperature(diffs, ys, t) for t in grid]
    k = min(range(200), key=lambda i: nlls[i])
    lo2 = grid[max(0, k - 1)]
    hi2 = grid[min(199, k + 1)]
    fine = [lo2 + (hi2 - lo2) * i / 99 for i in range(100)]
    nlls2 = [nll_temperature(diffs, ys, t) for t in fine]
    j = min(range(100), key=lambda i: nlls2[i])
    return fine[j], nlls2[j]


def ece_equal_freq(probs, ys, n_bins=10):
    """等频 10-bin ECE：Σ |B|/N · |acc(B) − conf(B)|。完美校准≈0。"""
    pairs = sorted(zip(probs, ys))
    n = len(pairs)
    if n == 0:
        return None
    total = 0.0
    for i in range(n_bins):
        chunk = pairs[i * n // n_bins:(i + 1) * n // n_bins]
        if not chunk:
            continue
        conf = sum(p for p, _ in chunk) / len(chunk)
        acc = sum(y for _, y in chunk) / len(chunk)
        total += len(chunk) / n * abs(acc - conf)
    return total


def summarize(deny_p, allow_p):
    """一组概率的指标包（deny_p: label=yes 的 P(yes)，allow_p: label=no 的）。"""
    thr = 0.5
    recall = (sum(1 for p in deny_p if p >= thr) / len(deny_p)) if deny_p else None
    false_block = (sum(1 for p in allow_p if p >= thr) / len(allow_p)) if allow_p else None
    sep = (sum(deny_p) / len(deny_p) - sum(allow_p) / len(allow_p)
           if deny_p and allow_p else None)
    auc = auc_mw(deny_p, allow_p)
    e = ece_equal_freq(deny_p + allow_p, [1] * len(deny_p) + [0] * len(allow_p))
    r3 = lambda v: round(v, 3) if v is not None else None
    return {
        "deny_recall@0.5": r3(recall),
        "false_block_rate@0.5": r3(false_block),
        "sep": r3(sep),
        "auc": r3(auc),
        "ece": r3(e),
        "mean_p_yes_deny": r3(sum(deny_p) / len(deny_p)) if deny_p else None,
        "mean_p_yes_allow": r3(sum(allow_p) / len(allow_p)) if allow_p else None,
    }
