"""运行 AgriAgent 端到端场景评测并生成可审计报告。

与只调用 ``_route_node`` 的路由评测不同，本脚本通过公开 ``run`` 接口执行整张
LangGraph：路由、追问、多轮状态、ReAct ToolNode、Planning 子 Agent 和最终回答
都会真实运行。外部天气和模型服务的失败也会计入结果，不从指标中剔除。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.append(str(BACKEND / "src"))

ERROR_MARKERS = (
    "发生错误",
    "执行失败",
    "本次没有子模块生成有效建议",
    "工具调用次数已达到上限",
    "请求超时",
)


def load_tasks(path: Path, limit: int | None = None, categories: set[str] | None = None) -> list[dict]:
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if categories:
        tasks = [task for task in tasks if task.get("category") in categories]
    return tasks[:limit] if limit else tasks


def sample_per_category(tasks: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    seen: Counter = Counter()
    for task in tasks:
        category = task.get("category")
        if seen[category] >= count:
            continue
        selected.append(task)
        seen[category] += 1
    return selected


def _compact_location(value: object) -> str:
    text = str(value or "").strip()
    for suffix in ("壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省", "市"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _context_matches(actual: dict, expected: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, wanted in expected.items():
        got = actual.get(key)
        if key in {"city", "region"}:
            matched = _compact_location(got) == _compact_location(wanted)
        elif isinstance(wanted, bool):
            matched = got is wanted
        else:
            matched = str(got or "").strip() == str(wanted).strip()
        if not matched:
            failures.append(f"context.{key}: expected={wanted!r}, actual={got!r}")
    return not failures, failures


def _tool_names(history: list[dict]) -> list[str]:
    names: list[str] = []
    for message in history:
        for call in message.get("tool_calls") or []:
            name = call.get("name") or (call.get("function") or {}).get("name")
            if name:
                names.append(str(name))
    return names


def _score_turn(expect: dict, snapshot: dict, history: list[dict], answer: str) -> dict:
    expected_route = expect.get("route")
    expected_caps = sorted(expect.get("capabilities") or [])
    expected_tools = sorted(expect.get("tools") or [])
    expected_tasks = sorted(expect.get("tasks") or [])
    expected_missing = sorted(expect.get("missing_fields") or [])

    actual_route = snapshot.get("route")
    actual_caps = sorted(snapshot.get("capabilities") or [])
    actual_tools = sorted(set(_tool_names(history)))
    task_status = snapshot.get("task_status") or {}
    actual_tasks = sorted(task_status)
    actual_missing = sorted(snapshot.get("missing_fields") or [])

    context_ok, context_failures = _context_matches(
        snapshot.get("context") or {}, expect.get("context") or {}
    )
    route_ok = actual_route == expected_route
    capabilities_ok = actual_caps == expected_caps
    tools_ok = actual_tools == expected_tools
    tasks_selected_ok = actual_tasks == expected_tasks
    tasks_completed_ok = all(
        (task_status.get(name) or {}).get("status") == "completed"
        for name in expected_tasks
    )
    missing_ok = actual_missing == expected_missing
    answer_text = str(answer or "").strip()
    output_ok = bool(answer_text) and not any(marker in answer_text for marker in ERROR_MARKERS)

    failures = list(context_failures)
    checks = {
        "route": route_ok,
        "capabilities": capabilities_ok,
        "context": context_ok,
        "missing_fields": missing_ok,
        "tools": tools_ok,
        "tasks_selected": tasks_selected_ok,
        "tasks_completed": tasks_completed_ok,
        "output": output_ok,
    }
    detail = {
        "route": (expected_route, actual_route),
        "capabilities": (expected_caps, actual_caps),
        "missing_fields": (expected_missing, actual_missing),
        "tools": (expected_tools, actual_tools),
        "tasks": (expected_tasks, actual_tasks),
    }
    for name, passed in checks.items():
        if not passed and name != "context":
            failures.append(f"{name}: expected={detail.get(name, 'ok')!r}")

    return {
        "success": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "actual": {
            "route": actual_route,
            "capabilities": actual_caps,
            "context": snapshot.get("context") or {},
            "missing_fields": actual_missing,
            "tools": actual_tools,
            "task_status": task_status,
            "answer": answer_text,
        },
    }


def run_task(agent, task: dict, run_id: str, retries: int = 1, sleep: float = 0.0) -> dict:
    last_error: str | None = None
    for attempt in range(retries + 1):
        thread_id = f"e2e-{run_id}-{task['id']}-attempt-{attempt}"
        turn_records: list[dict] = []
        try:
            for turn_index, turn in enumerate(task["turns"], 1):
                before_history = agent.get_history(thread_id)
                started = time.perf_counter()
                answer = agent.run(turn["user"], thread_id=thread_id)
                latency = time.perf_counter() - started
                snapshot = agent.get_snapshot(thread_id)
                all_history = agent.get_history(thread_id)
                new_history = all_history[len(before_history):]
                scored = _score_turn(turn["expect"], snapshot, new_history, answer)
                turn_records.append({
                    "turn": turn_index,
                    "user": turn["user"],
                    "expect": turn["expect"],
                    "latency_s": round(latency, 3),
                    **scored,
                })
                if sleep:
                    time.sleep(sleep)
            return {
                "id": task["id"],
                "category": task["category"],
                "source": task.get("source"),
                "success": all(turn["success"] for turn in turn_records),
                "attempts": attempt + 1,
                "error": None,
                "turns": turn_records,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(max(1.0, sleep))
                continue
            return {
                "id": task["id"],
                "category": task["category"],
                "source": task.get("source"),
                "success": False,
                "attempts": attempt + 1,
                "error": last_error,
                "turns": turn_records,
            }
    raise AssertionError("unreachable")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def summarize(records: list[dict]) -> dict:
    turns = [turn for record in records for turn in record.get("turns", [])]
    latencies = [turn["latency_s"] for turn in turns]

    tool_tp = tool_fp = tool_fn = 0
    expected_subtasks = completed_subtasks = 0
    no_action_turns = false_action_turns = 0
    expected_context_fields = correct_context_fields = 0

    for turn in turns:
        expected = turn["expect"]
        actual = turn["actual"]
        expected_tools = set(expected.get("tools") or [])
        actual_tools = set(actual.get("tools") or [])
        tool_tp += len(expected_tools & actual_tools)
        tool_fp += len(actual_tools - expected_tools)
        tool_fn += len(expected_tools - actual_tools)

        for name in expected.get("tasks") or []:
            expected_subtasks += 1
            if (actual.get("task_status", {}).get(name) or {}).get("status") == "completed":
                completed_subtasks += 1

        if not expected_tools and not (expected.get("tasks") or []):
            no_action_turns += 1
            if actual_tools or actual.get("task_status"):
                false_action_turns += 1

        expected_context_fields += len(expected.get("context") or {})
        actual_context = actual.get("context") or {}
        for key, wanted in (expected.get("context") or {}).items():
            if key in {"city", "region"}:
                correct = _compact_location(actual_context.get(key)) == _compact_location(wanted)
            else:
                correct = str(actual_context.get(key) or "").strip() == str(wanted).strip()
            correct_context_fields += int(correct)

    tool_precision = _safe_ratio(tool_tp, tool_tp + tool_fp)
    tool_recall = _safe_ratio(tool_tp, tool_tp + tool_fn)
    if tool_precision is not None and tool_recall is not None and tool_precision + tool_recall:
        tool_f1 = round(2 * tool_precision * tool_recall / (tool_precision + tool_recall), 4)
    else:
        tool_f1 = 0.0

    category_summary: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    for category, subset in sorted(grouped.items()):
        subset_turns = [turn for record in subset for turn in record.get("turns", [])]
        category_summary[category] = {
            "tasks": len(subset),
            "task_success_rate": _safe_ratio(sum(r["success"] for r in subset), len(subset)),
            "turn_success_rate": _safe_ratio(sum(t["success"] for t in subset_turns), len(subset_turns)),
        }

    return {
        "tasks": len(records),
        "turns": len(turns),
        "task_success_rate": _safe_ratio(sum(record["success"] for record in records), len(records)),
        "turn_success_rate": _safe_ratio(sum(turn["success"] for turn in turns), len(turns)),
        "route_accuracy": _safe_ratio(sum(turn["checks"]["route"] for turn in turns), len(turns)),
        "capability_exact_rate": _safe_ratio(
            sum(turn["checks"]["capabilities"] for turn in turns), len(turns)
        ),
        "context_field_accuracy": _safe_ratio(correct_context_fields, expected_context_fields),
        "clarification_accuracy": _safe_ratio(
            sum(
                turn["checks"]["route"] and turn["checks"]["missing_fields"]
                for turn in turns
                if turn["expect"].get("route") == "clarify"
            ),
            sum(1 for turn in turns if turn["expect"].get("route") == "clarify"),
        ),
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "tool_f1": tool_f1,
        "subagent_completion_rate": _safe_ratio(completed_subtasks, expected_subtasks),
        "no_action_false_trigger_rate": _safe_ratio(false_action_turns, no_action_turns),
        "output_success_rate": _safe_ratio(sum(turn["checks"]["output"] for turn in turns), len(turns)),
        "runtime_error_count": sum(1 for record in records if record.get("error")),
        "average_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_s": _percentile(latencies, 0.95),
        "category_breakdown": category_summary,
    }


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def write_reports(summary: dict, records: list[dict], tag: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"e2e_agent_{tag}.json"
    markdown_path = RESULTS_DIR / f"e2e_agent_report_{tag}.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "真实执行 LangGraph 主流程；场景数据不代表线上用户流量",
        "summary": summary,
        "records": records,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        f"# AgriAgent 端到端场景评测（{tag}）",
        "",
        "> 通过公开 `run` 接口真实执行 LangGraph 主流程；任务由仓库知识槽位与农业场景模板构造，不代表线上用户流量。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 场景任务数 | {summary['tasks']} |",
        f"| 对话轮次数 | {summary['turns']} |",
        f"| 端到端任务完成率 | {_fmt_rate(summary['task_success_rate'])} |",
        f"| 单轮执行成功率 | {_fmt_rate(summary['turn_success_rate'])} |",
        f"| 路由准确率 | {_fmt_rate(summary['route_accuracy'])} |",
        f"| 能力集合完全匹配率 | {_fmt_rate(summary['capability_exact_rate'])} |",
        f"| 状态字段准确率 | {_fmt_rate(summary['context_field_accuracy'])} |",
        f"| 缺参追问正确率 | {_fmt_rate(summary['clarification_accuracy'])} |",
        f"| 工具调用 F1 | {_fmt_rate(summary['tool_f1'])} |",
        f"| Planning 子任务完成率 | {_fmt_rate(summary['subagent_completion_rate'])} |",
        f"| 无动作场景误触发率 ↓ | {_fmt_rate(summary['no_action_false_trigger_rate'])} |",
        f"| 非错误输出率 | {_fmt_rate(summary['output_success_rate'])} |",
        f"| 平均端到端延迟 | {summary['average_latency_s']}s |",
        f"| P95 端到端延迟 | {summary['p95_latency_s']}s |",
        f"| 运行时异常数 | {summary['runtime_error_count']} |",
        "",
        "## 分场景结果",
        "",
        "| 场景 | 任务数 | 任务完成率 | 单轮成功率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, item in summary["category_breakdown"].items():
        rows.append(
            f"| {category} | {item['tasks']} | {_fmt_rate(item['task_success_rate'])} | "
            f"{_fmt_rate(item['turn_success_rate'])} |"
        )

    failed = [record for record in records if not record["success"]]
    if failed:
        rows += ["", "## 失败样本（前 20 条）", ""]
        for record in failed[:20]:
            reasons = []
            for turn in record.get("turns", []):
                reasons.extend(turn.get("failures") or [])
            reason_text = "；".join(reasons[:4]) or record.get("error") or "未知"
            rows.append(f"- `{record['id']}`（{record['category']}）：{reason_text}")

    rows += [
        "",
        "## 指标边界",
        "",
        "- 任务完成要求该任务全部轮次同时通过路由、能力、状态、工具/子 Agent 轨迹与非错误输出检查。",
        "- 该评测衡量系统执行正确性，不使用 LLM-as-Judge 判断最终建议的农业专业质量。",
        "- 外部模型、天气和 A2A 服务异常不会从分母中剔除。",
    ]
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "e2e_tasks.jsonl")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--replace-from", type=Path, default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=True)
    missing = [
        key
        for key in ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")
        if not os.getenv(key)
    ]
    if missing:
        raise SystemExit(
            f"缺少环境变量：{', '.join(missing)}；已尝试读取 {args.env_file}"
        )

    from agri_agent.agents.agri_graph import AgriGraphAgent

    tasks = load_tasks(args.dataset, None, set(args.category) or None)
    if args.task_id:
        wanted_ids = set(args.task_id)
        tasks = [task for task in tasks if task["id"] in wanted_ids]
    if args.per_category:
        tasks = sample_per_category(tasks, max(1, args.per_category))
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("没有匹配的评测任务")

    print(f"加载 {len(tasks)} 个待执行任务，开始真实执行 AgriGraph 主流程")
    run_id = uuid4().hex[:8]
    agent = AgriGraphAgent()
    partial_path = RESULTS_DIR / f"e2e_agent_{args.tag}.partial.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    if args.replace_from:
        previous = json.loads(args.replace_from.read_text(encoding="utf-8"))
        records = previous.get("records", previous)
        replacement_ids = {task["id"] for task in tasks}
        records = [record for record in records if record["id"] not in replacement_ids]
        print(f"替换既有评测中的 {len(replacement_ids)} 个任务")
    elif args.resume_from:
        previous = json.loads(args.resume_from.read_text(encoding="utf-8"))
        records = previous.get("records", previous)
        completed_ids = {record["id"] for record in records}
        tasks = [task for task in tasks if task["id"] not in completed_ids]
        print(f"复用既有评测：已完成 {len(records)} 个，剩余 {len(tasks)} 个")
    elif args.resume and partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8"))
        completed_ids = {record["id"] for record in records}
        tasks = [task for task in tasks if task["id"] not in completed_ids]
        print(f"断点续跑：已完成 {len(records)} 个，剩余 {len(tasks)} 个")

    total_tasks = len(records) + len(tasks)

    for offset, task in enumerate(tasks, 1):
        record = run_task(
            agent,
            task,
            run_id=run_id,
            retries=max(0, args.retries),
            sleep=max(0.0, args.sleep),
        )
        records.append(record)
        partial_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        completed = len(records)
        if completed % 5 == 0 or offset == len(tasks):
            current = sum(item["success"] for item in records) / len(records)
            print(f"[{completed}/{total_tasks}] 当前任务完成率 {current:.2%}", flush=True)

    summary = summarize(records)
    json_path, markdown_path = write_reports(summary, records, args.tag)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
