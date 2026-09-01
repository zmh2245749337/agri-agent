# backend/eval/route_accuracy_eval.py
"""
统一主图「路由准确率」评测。

测什么
------
只测 AgriGraph 的第一步——路由器：给定用户这句话（及最近对话），它把请求分到
direct / react / planning / clarify 哪条线、需要哪些农业能力。**不跑下游工具**，
所以一条样本只产生一次模型调用，两百多条样本在 flash 级模型上成本可以忽略。

为什么只测路由器而不测端到端
----------------------------
路由错了后面全错，它是整张图误差的源头；端到端评测会把工具返回质量、外部服务
可用性、生成措辞一起混进来，指标动了也说不清是哪一层的问题。先把源头这一层
量化住，是更干净的做法。端到端评测是后续方向，不在这个脚本范围内。

指标口径（与 Gemma Tool-Use 项目保持同一套设计思路）
--------------------------------------------------
- 语义层：整体路由准确率、分类别准确率、四类混淆矩阵；能力识别 micro P/R/F1
- 安全层：**无能力轮误触发率** —— 期望 direct/clarify（不需要任何农业能力）
  却被判成 react/planning 的比例。这是路由层对应「不该调工具却调了」的指标
- 工程层：路由模型异常触发保守兜底的次数

运行前提
--------
仓库根目录的 .env 里配好 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL（脚本会自动加载），
并已在项目的 conda 环境里装好 backend/requirements.txt。
先用 --limit 20 试跑确认环境，再跑全量。

用法（PowerShell，在项目根目录下）
----------------------------------
    conda activate agri-agent
    cd backend
    python eval\build_route_eval.py                    # 生成数据集（已生成过就不必重跑）
    python eval\route_accuracy_eval.py --limit 20      # 冒烟，确认能连上模型
    python eval\route_accuracy_eval.py                 # 主集全量
    python eval\route_accuracy_eval.py --challenge     # 难例集

conda activate 报「不是内部或外部命令」时，改用绝对路径直接调环境里的解释器：
    C:\anaconda\envs\agri-agent\python.exe eval\route_accuracy_eval.py --limit 20
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
sys.path.append(str(BACKEND / "src"))

# 显式加载仓库根目录的 .env。项目里只有 api/main.py 和各 agent 的 __main__ 块调了
# load_dotenv()，直接 import AgriGraphAgent 的脚本不会自动拿到 LLM_API_KEY 等变量，
# 不补这一步会在实例化时报「必须提供 model / api_key / base_url」。
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from langchain_core.messages import AIMessage, HumanMessage

from agri_agent.agents.agri_graph import AgriGraphAgent

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROUTES = ["direct", "react", "planning", "clarify"]


class RecordingRouter:
    """
    包一层路由模型：失败时自动重试，并把失败原因记下来。

    为什么需要：AgriGraph 在路由模型异常时会走 _fallback_decision 保守兜底，
    而兜底只按关键词猜能力、不提取 crop/city/region 等槽位，必填字段随即全空、
    路由被下游改判成 clarify。如果不把这类样本单独标出来，评测测到的就不是
    「路由器准不准」，而是「这轮网络稳不稳」——两者必须分开。
    """

    def __init__(self, inner, retries=2, backoff=2.0):
        self.inner = inner
        self.retries = retries
        self.backoff = backoff
        self.failures = []
        self.retry_count = 0

    def invoke(self, messages):
        last = None
        for attempt in range(self.retries + 1):
            try:
                result = self.inner.invoke(messages)
                if result is None:
                    raise ValueError("路由模型返回空结果")
                return result
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    self.retry_count += 1
                    time.sleep(self.backoff * (attempt + 1))
        self.failures.append(f"{type(last).__name__}: {last}")
        raise last


def load_rows(path, limit=None):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def build_state(row):
    """把一条样本还原成 AgriState：历史对话 + 本轮用户消息。"""
    messages = []
    for turn in row.get("history", []):
        content = turn.get("content", "")
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=row["query"]))
    return {"messages": messages, "mode": "auto", "capabilities": [],
            "missing_fields": [], "tool_rounds": 0, "task_status": {}}


def evaluate(rows, agent, sleep=0.0):
    records, fallback_hits = [], 0
    for i, row in enumerate(rows, 1):
        if sleep and i > 1:
            time.sleep(sleep)
        state = build_state(row)
        t0 = time.time()
        try:
            updates = agent._route_node(state)
            error = None
        except Exception as exc:          # 单条失败不中断整轮评测
            updates, error = {}, f"{type(exc).__name__}: {exc}"
        latency = time.time() - t0

        pred_route = updates.get("route")
        pred_caps = sorted(updates.get("capabilities") or [])
        gold_caps = sorted(row.get("expect_capabilities") or [])
        # 置信度为兜底默认值时记一次（_fallback_decision 固定给 0.7）
        used_fallback = updates.get("confidence") == 0.7
        if used_fallback:
            fallback_hits += 1

        records.append({
            "id": row["id"], "kind": row.get("kind"), "source": row.get("source"),
            "query": row["query"], "has_history": bool(row.get("history")),
            "gold_route": row["expect_route"], "pred_route": pred_route,
            "gold_capabilities": gold_caps, "pred_capabilities": pred_caps,
            "route_correct": pred_route == row["expect_route"],
            "capabilities_exact": pred_caps == gold_caps,
            "confidence": updates.get("confidence"),
            "used_fallback": used_fallback,
            "latency_s": round(latency, 3), "error": error,
        })
        if i % 20 == 0:
            acc = sum(r["route_correct"] for r in records) / len(records)
            fb = sum(r["used_fallback"] for r in records)
            print(f"  [{i}/{len(rows)}] 路由准确率 {acc:.2%}（其中 {fb} 条走了保守兜底）")
    return records, fallback_hits


def summarize(records, fallback_hits):
    n = len(records)
    route_acc = sum(r["route_correct"] for r in records) / n
    ok = [r for r in records if not r["used_fallback"]]
    route_acc_ok = round(sum(r["route_correct"] for r in ok) / len(ok), 4) if ok else None

    per_route = {}
    for route in ROUTES:
        sub = [r for r in records if r["gold_route"] == route]
        if sub:
            per_route[route] = {"n": len(sub),
                                "accuracy": round(sum(x["route_correct"] for x in sub) / len(sub), 4)}

    confusion = defaultdict(Counter)
    for r in records:
        confusion[r["gold_route"]][r["pred_route"] or "ERROR"] += 1

    # 能力识别 micro P/R/F1
    tp = fp = fn = 0
    for r in records:
        g, p = set(r["gold_capabilities"]), set(r["pred_capabilities"])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # 安全层：无能力轮误触发率
    no_cap = [r for r in records if not r["gold_capabilities"]]
    false_trigger = [r for r in no_cap if r["pred_capabilities"] or r["pred_route"] in ("react", "planning")]

    return {
        "samples": n,
        "route_accuracy": round(route_acc, 4),
        "route_accuracy_excl_fallback": route_acc_ok,
        "fallback_free_samples": len(ok),
        "per_route_accuracy": per_route,
        "confusion_matrix": {g: dict(c) for g, c in confusion.items()},
        "capability_precision": round(prec, 4),
        "capability_recall": round(rec, 4),
        "capability_f1": round(f1, 4),
        "capability_exact_rate": round(sum(r["capabilities_exact"] for r in records) / n, 4),
        "no_capability_turns": len(no_cap),
        "false_trigger_count": len(false_trigger),
        "false_trigger_rate": round(len(false_trigger) / len(no_cap), 4) if no_cap else None,
        "fallback_hits": fallback_hits,
        "fallback_reasons": [],
        "error_count": sum(1 for r in records if r["error"]),
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / n, 3),
    }


def write_report(summary, records, tag):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"route_accuracy_{tag}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    L = [f"# 统一主图路由准确率评测（{tag}）", "",
         "> 只评测路由器本身，不触发下游工具；每条样本一次模型调用。", "",
         "## 总体", "", "| 指标 | 数值 |", "| --- | ---: |",
         f"| 样本数 | {summary['samples']} |",
         f"| 路由准确率（含兜底样本） | {summary['route_accuracy']:.2%} |",
         f"| 路由准确率（仅模型成功返回的 {summary['fallback_free_samples']} 条） | "
         f"{summary['route_accuracy_excl_fallback']:.2%} |"
         if summary["route_accuracy_excl_fallback"] is not None else "| 路由准确率（排除兜底） | n/a |",
         f"| 能力识别 F1 | {summary['capability_f1']:.2%} |",
         f"| 能力集合完全匹配率 | {summary['capability_exact_rate']:.2%} |",
         f"| 无能力轮误触发率 ↓ | {summary['false_trigger_rate']:.2%} "
         f"（{summary['false_trigger_count']}/{summary['no_capability_turns']}） |"
         if summary["false_trigger_rate"] is not None else "| 无能力轮误触发率 | n/a |",
         f"| 保守兜底触发次数 | {summary['fallback_hits']} |",
         f"| 异常样本数 | {summary['error_count']} |",
         f"| 平均单次路由耗时 | {summary['avg_latency_s']}s |", "",
         "## 分类别准确率", "", "| 路由 | 样本数 | 准确率 |", "| --- | ---: | ---: |"]
    for route, v in summary["per_route_accuracy"].items():
        L.append(f"| {route} | {v['n']} | {v['accuracy']:.2%} |")

    L += ["", "## 混淆矩阵（行=期望，列=预测）", "",
          "| 期望 \\ 预测 | " + " | ".join(ROUTES) + " |",
          "| --- | " + " | ".join(["---:"] * len(ROUTES)) + " |"]
    for g in ROUTES:
        row = summary["confusion_matrix"].get(g, {})
        L.append(f"| {g} | " + " | ".join(str(row.get(p, 0)) for p in ROUTES) + " |")

    wrong = [r for r in records if not r["route_correct"]][:15]
    if wrong:
        L += ["", "## 典型错误样例", ""]
        for r in wrong:
            L.append(f"- `{r['id']}`（{r['kind']}）「{r['query']}」"
                     f"期望 **{r['gold_route']}**，预测 **{r['pred_route']}**")

    path = RESULTS_DIR / f"route_accuracy_report_{tag}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", action="store_true", help="评测难例集而非主集")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条，用于冒烟")
    ap.add_argument("--sleep", type=float, default=0.0, help="每条之间的间隔秒数，限流严重时调大")
    args = ap.parse_args()

    tag = "challenge" if args.challenge else "main"
    path = DATA_DIR / ("route_eval_challenge.jsonl" if args.challenge else "route_eval.jsonl")
    if not path.exists():
        raise SystemExit(f"数据集不存在：{path}\n请先运行 python eval/build_route_eval.py")

    rows = load_rows(path, args.limit)
    print(f"评测集：{path.name}，共 {len(rows)} 条")

    # 快速失败并给出可操作的提示，避免直接抛 pydantic 的 "model Input should be a
    # valid string"——那个报错看不出真正原因是环境变量没读到。
    import os
    missing = [k for k in ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL") if not os.getenv(k)]
    if missing:
        env_path = REPO_ROOT / ".env"
        raise SystemExit(
            f"缺少环境变量：{', '.join(missing)}\n"
            f"脚本已尝试从 {env_path} 加载（存在：{env_path.exists()}）。\n"
            "请检查：\n"
            "  1. 仓库根目录下有 .env（不是 .env.example），且包含上述变量；\n"
            "  2. 当前环境装了 python-dotenv：pip install python-dotenv；\n"
            "  3. 或在 PowerShell 里临时设置后重试：\n"
            '     $env:LLM_MODEL_ID="glm-4-flash-250414"\n'
            '     $env:LLM_API_KEY="你的key"\n'
            '     $env:LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"'
        )

    agent = AgriGraphAgent()
    agent.router = RecordingRouter(agent.router)
    records, fallback_hits = evaluate(rows, agent, sleep=args.sleep)
    summary = summarize(records, fallback_hits)
    summary["fallback_reasons"] = sorted(set(agent.router.failures))[:10]
    summary["router_retries"] = agent.router.retry_count

    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {write_report(summary, records, tag)}")


if __name__ == "__main__":
    main()
