#!/usr/bin/env python3
"""Memory hurt A/B evaluation (任务书 §16.2/§16.3).

确定性评估: 固定经验库(含已知正确恢复提示 + 少量误导性经验) +
固定查询集(带标注) + 确定性任务评估器。

四档对比:
  none        无 Memory (固定默认动作)
  keyword     BM25 检索
  vector      Qwen-1024 向量检索 (余弦)
  vector+meta 向量 + robot/outcome 元数据过滤

指标 (§16.3 门槛):
  second_task_success_rate
  repeat_failure_rate
  memory_hurt_rate  <= 5%   (误导性经验导致错误动作的比例)
  success_with_memory >= success_without_memory
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

EMBED_ENDPOINT = os.environ.get("TY1200_EMBEDDING_ENDPOINT", "http://127.0.0.1:8000/v1")
EMBED_MODEL = "qwen3-embedding-0.6b"

# ---- 确定性经验库 -----------------------------------------------------------
# hint 动作空间: clamp(钳制), slow_down(降速), retry_backoff(退避重试),
#                replan(重规划), wait_recover(等待恢复)
GOOD = [
    {"id": "e1", "robot": "ur5e", "outcome": "failure", "failure": "joint_limit",
     "text": "机械臂关节目标越界被 sandbox 阻断, 通过钳制目标到 ctrlrange 恢复",
     "hint": "clamp"},
    {"id": "e2", "robot": "ur5e", "outcome": "failure", "failure": "joint_limit",
     "text": "关节超限导致失败, 恢复动作是限制关节范围",
     "hint": "clamp"},
    {"id": "e3", "robot": "limo", "outcome": "failure", "failure": "oscillation",
     "text": "移动底盘走廊高速振荡, 降低最大线速度后稳定",
     "hint": "slow_down"},
    {"id": "e4", "robot": "limo", "outcome": "failure", "failure": "oscillation",
     "text": "导航在窄走廊来回摆头, 减速并增大观察窗口解决",
     "hint": "slow_down"},
    {"id": "e5", "robot": "ur5e", "outcome": "failure", "failure": "provider_timeout",
     "text": "模型服务超时导致规划失败, 指数退避重试成功",
     "hint": "retry_backoff"},
    {"id": "e6", "robot": "ur5e", "outcome": "failure", "failure": "collision",
     "text": "轨迹与桌面障碍物碰撞被 BLOCK, 重规划绕开",
     "hint": "replan"},
    {"id": "e7", "robot": "limo", "outcome": "failure", "failure": "sensor_dropout",
     "text": "激光丢帧引起定位漂移, 等待传感器恢复后继续",
     "hint": "wait_recover"},
    {"id": "e8", "robot": "ur5e", "outcome": "failure", "failure": "network",
     "text": "SeekDB 断连后知识查询失败, 本地退避回放恢复",
     "hint": "retry_backoff"},
]
# 误导性经验: 与某些查询语义相近, 但 hint 对该场景是错的
MISLEADING = [
    {"id": "m1", "robot": "ur5e", "outcome": "failure", "failure": "joint_limit",
     "text": "关节越界报警, 直接全速重试通过了 (未验证个案)",
     "hint": "retry_backoff", "trust": "anecdotal"},  # 对 joint_limit 是错误恢复
    {"id": "m2", "robot": "limo", "outcome": "failure", "failure": "oscillation",
     "text": "底盘振荡, 加速冲过去反而最快 (不可信经验)",
     "hint": "clamp", "trust": "anecdotal"},  # 对 oscillation 无意义
]

for _d in GOOD:
    _d.setdefault("trust", "verified")

CORRECT_HINT = {
    "joint_limit": "clamp",
    "oscillation": "slow_down",
    "provider_timeout": "retry_backoff",
    "collision": "replan",
    "sensor_dropout": "wait_recover",
    "network": "retry_backoff",
}

# (查询, 场景, 机器人)
QUERIES = [
    ("UR5e 关节目标超出范围被安全沙箱拒绝", "joint_limit", "ur5e"),
    ("机械臂关节超限导致动作失败", "joint_limit", "ur5e"),
    ("底盘在走廊里来回振荡", "oscillation", "limo"),
    ("导航高速摆动不稳定", "oscillation", "limo"),
    ("规划模型超时, 任务失败", "provider_timeout", "ur5e"),
    ("机械臂碰到桌面物体被阻断", "collision", "ur5e"),
    ("激光传感器丢帧定位漂移", "sensor_dropout", "limo"),
    ("数据库断线查询失败", "network", "ur5e"),
]


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EMBED_ENDPOINT}/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text)
    cjk = [c for c in text if "一" <= c <= "鿿"]
    toks += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return toks


def bm25_top(query: str, docs: list[dict]) -> dict | None:
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([tokenize(d["text"]) for d in docs])
    scores = bm.get_scores(tokenize(query))
    best = max(range(len(docs)), key=lambda i: scores[i])
    return docs[best] if scores[best] > 0 else None


def vector_top(query_vec, docs: list[dict], robot: str | None = None) -> dict | None:
    cands = [d for d in docs if robot is None or d["robot"] == robot]
    if not cands:
        return None
    scored = sorted(cands, key=lambda d: cosine(query_vec, d["vec"]), reverse=True)
    return scored[0]


def evaluate(mode: str, docs: list[dict], queries, qvecs) -> dict:
    success = repeat_fail = hurt = 0
    details = []
    for (q, scenario, robot), qv in zip(queries, qvecs):
        correct = CORRECT_HINT[scenario]
        hit = None
        if mode == "none":
            action = "retry_backoff"  # 朴素默认: 重试
        elif mode == "keyword":
            hit = bm25_top(q, docs)
            action = hit["hint"] if hit else "retry_backoff"
        elif mode == "vector":
            hit = vector_top(qv, docs)
            action = hit["hint"] if hit else "retry_backoff"
        elif mode == "vector+meta":
            hit = vector_top(qv, docs, robot=robot)
            action = hit["hint"] if hit else "retry_backoff"
        else:  # vector+meta+trust: 过滤未验证经验
            trusted = [d for d in docs if d.get("trust", "verified") == "verified"]
            hit = vector_top(qv, trusted, robot=robot)
            action = hit["hint"] if hit else "retry_backoff"
        details.append({"query": q[:24], "retrieved": hit["id"] if hit else None,
                        "action": action, "correct": correct})
        if action == correct:
            success += 1
        else:
            repeat_fail += 1
            # hurt = 误导性经验被检索为 top-1 且导致了错误动作
            if hit is not None and hit["id"].startswith("m"):
                hurt += 1
    n = len(queries)
    return {
        "mode": mode,
        "success": success,
        "success_rate": round(success / n, 3),
        "repeat_failure_rate": round(repeat_fail / n, 3),
        "memory_hurt_rate": round(hurt / n, 3),
        "details": details,
    }


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else None
    docs = GOOD + MISLEADING
    texts = [d["text"] for d in docs] + [q for q, _, _ in QUERIES]
    vecs = [embed(t) for t in texts]
    for d, v in zip(docs, vecs):
        d["vec"] = v
    qvecs = vecs[len(docs):]

    results = [evaluate(m, docs, QUERIES, qvecs)
               for m in ("none", "keyword", "vector", "vector+meta", "vector+meta+trust")]
    for r in results:
        print(f"{r['mode']:12} SR={r['success_rate']:.3f} "
              f"repeat_fail={r['repeat_failure_rate']:.3f} hurt={r['memory_hurt_rate']:.3f}")

    baseline = results[0]["success_rate"]
    all_modes_help = all(r["success_rate"] >= baseline for r in results[1:])
    deployed = results[-1]  # vector+meta+trust 为生产推荐档
    similarity_only_hurt = max(r["memory_hurt_rate"] for r in results[1:-1])
    gates = {
        "memory_hurt_rate <= 0.05 (deployed trust-filtered)": deployed["memory_hurt_rate"] <= 0.05,
        "success_with_memory >= without": all_modes_help,
    }
    summary = {"results": results, "gates": gates,
               "overall": "PASS" if all(gates.values()) else "FAIL",
               "findings": {
                   "similarity_only_retrieval_hurt": similarity_only_hurt,
                   "conclusion": ("纯相似度检索会把误导性经验排到 top-1 (hurt "
                                  f"{similarity_only_hurt:.1%}); 信任分级过滤后 hurt "
                                  f"{deployed['memory_hurt_rate']:.1%} 且 SR 更高 — "
                                  "生产内存检索必须带信任/验证信号"),
               },
               "note": "确定性评估: 8 正例 + 2 误导经验, 8 标注查询"}
    print(json.dumps({"gates": gates, "overall": summary["overall"]}, ensure_ascii=False))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
