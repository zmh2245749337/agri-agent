"""运行 AgriAgent V2 结果导向挑战集。

与 ``e2e_agent_eval.py`` 的区别：

* 不要求内部 route、capability 列表或工具顺序与预设轨迹完全相同；
* 以最终上下文、实际完成的业务领域、可核验证据和非错误输出判定成功；
* 多轮缺参场景额外检查第一轮是否发现缺失字段、是否避免提前调用业务工具；
* 默认使用冻结天气响应，避免实时天气变化污染可复现性。

自然语言建议的“专业水平”不由脚本自动打分。该指标只表示可客观核验的业务目标
是否完成，不能解释成线上用户满意度或农业诊断准确率。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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

TOOL_DOMAIN = {
    "diagnose_crop_disease": "diagnosis",
    "get_weather_forecast": "weather",
    "search_subsidy_policy": "policy",
    "web_search_policy": "policy",
}

FROZEN_WEATHER = {
    "长沙": "长沙：小雨，18-24℃，东南风3级；有降雨，不建议露天喷药。",
    "广州": "广州：多云转阵雨，24-31℃，南风3级；午后可能降雨。",
    "南京": "南京：晴到多云，19-28℃，东北风2级；白天适合常规田间作业。",
    "成都": "成都：阴，17-23℃，风力2级；空气湿度较高。",
    "武汉": "武汉：中雨，20-26℃，北风4级；不适合喷药和收割。",
    "郑州": "郑州：晴，16-27℃，西风2级；未来24小时无明显降水。",
    "济南": "济南：多云，15-25℃，南风3级；降雨概率较低。",
    "合肥": "合肥：小雨转阴，18-25℃，东风3级；上午不宜喷药。",
    "南昌": "南昌：雷阵雨，22-29℃，阵风5级；暂停露天施药。",
    "西安": "西安：晴，14-26℃，东风2级；昼夜温差较大。",
    "昆明": "昆明：阵雨，15-22℃，西南风3级；注意短时降雨。",
    "哈尔滨": "哈尔滨：多云，8-18℃，西北风4级；早晚温度较低。",
    "沈阳": "沈阳：晴，10-21℃，北风3级；天气较干燥。",
    "福州": "福州：中雨，23-30℃，南风3级；不建议当天施药。",
    "贵阳": "贵阳：阴有小雨，14-20℃，东北风2级；田间湿度偏高。",
    "太原": "太原：晴，9-23℃，西北风3级；无明显降水。",
    "兰州": "兰州：多云，10-24℃，东风2级；空气偏干。",
    "长春": "长春：小雨，7-16℃，北风4级；注意低温和降雨。",
    "南宁": "南宁：阵雨，24-31℃，东南风3级；午后避免施药。",
    "呼和浩特": "呼和浩特：晴，6-20℃，西北风4级；风力偏大。",
    "乌鲁木齐": "乌鲁木齐：晴，7-19℃，西北风3级；天气干燥。",
    "青岛": "青岛：多云，15-22℃，东南风4级；沿海风力较大。",
}


def load_scenarios(path: Path, limit: int | None = None) -> list[dict]:
    scenarios = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return scenarios[:limit] if limit else scenarios


def _dataset_hash(scenarios: list[dict]) -> str:
    payload = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in scenarios
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_location(value: object) -> str:
    text = str(value or "").strip()
    for suffix in ("壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省", "市"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _same_value(key: str, actual: object, expected: object) -> bool:
    if key in {"city", "region"}:
        return _compact_location(actual) == _compact_location(expected)
    return str(actual or "").strip() == str(expected or "").strip()


def _tool_calls(history: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for message in history:
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = item.get("name") or function.get("name")
            args = item.get("args") if "args" in item else function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": str(name or ""), "args": args or {}})
    return calls


def _tool_outputs(history: list[dict]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in history
        if message.get("role") == "tool"
    ]


def _parse_tool_output(text: str) -> object:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return text


def _subagent_outputs(agent) -> tuple[dict, list[str]]:
    planning_agent = getattr(agent, "planning_agent", None)
    tasks = getattr(planning_agent, "last_tasks", {}) or {}
    statuses: dict[str, dict] = {}
    outputs: list[str] = []
    for name, task in tasks.items():
        status = getattr(task, "status", None)
        status_value = getattr(status, "value", status)
        statuses[name] = {
            "status": str(status_value or ""),
            "protocol": getattr(task, "protocol", None),
            "fallback_reason": getattr(task, "fallback_reason", None),
        }
        result_text = getattr(task, "result_text", None)
        if result_text:
            outputs.append(str(result_text))
        evidence = getattr(task, "evidence", None)
        if evidence is not None:
            outputs.append(json.dumps(evidence, ensure_ascii=False, default=str))
    return statuses, outputs


def _action_domains(tool_calls: list[dict], task_status: dict) -> set[str]:
    domains = {
        TOOL_DOMAIN[call["name"]]
        for call in tool_calls
        if call.get("name") in TOOL_DOMAIN
    }
    for name, item in (task_status or {}).items():
        status = str((item or {}).get("status") or "").lower()
        if name in {"diagnosis", "weather", "policy"} and status in {"completed", "success"}:
            domains.add(name)
    return domains


def _output_ok(answer: str) -> bool:
    text = str(answer or "").strip()
    return bool(text) and not any(marker in text for marker in ERROR_MARKERS)


def _score_checkpoint(checkpoint: dict, snapshot: dict, action_domains: set[str]) -> dict:
    failures: list[str] = []
    required_missing = set(checkpoint.get("missing_contains") or [])
    actual_missing = set(snapshot.get("missing_fields") or [])
    pending_slot = snapshot.get("pending_slot")
    missing_ok = required_missing.issubset(actual_missing | ({pending_slot} if pending_slot else set()))
    if not missing_ok:
        failures.append(
            f"缺参追问：expected contains={sorted(required_missing)!r}, "
            f"actual={sorted(actual_missing)!r}, pending={pending_slot!r}"
        )
    no_early_action_ok = not checkpoint.get("forbid_domain_actions") or not action_domains
    if not no_early_action_ok:
        failures.append(f"缺参前提前执行业务能力：{sorted(action_domains)!r}")
    return {
        "success": missing_ok and no_early_action_ok,
        "checks": {
            "missing_detected": missing_ok,
            "no_early_action": no_early_action_ok,
        },
        "failures": failures,
    }


def _score_goal(goal: dict, turns: list[dict]) -> dict:
    failures: list[str] = []
    final = turns[-1]
    final_context = final["snapshot"].get("context") or {}
    context_checks = {
        key: _same_value(key, final_context.get(key), expected)
        for key, expected in (goal.get("context") or {}).items()
    }
    if not all(context_checks.values()):
        for key, passed in context_checks.items():
            if not passed:
                failures.append(
                    f"最终上下文 {key}: expected={goal['context'][key]!r}, "
                    f"actual={final_context.get(key)!r}"
                )

    all_domains: set[str] = set()
    evidence_parts: list[str] = []
    for turn in turns:
        all_domains.update(turn["action_domains"])
        evidence_parts.extend(turn.get("tool_outputs") or [])
        evidence_parts.extend(turn.get("subagent_outputs") or [])
    evidence_text = "\n".join(evidence_parts)

    required_domains = set(goal.get("required_domains") or [])
    domains_ok = required_domains.issubset(all_domains)
    if not domains_ok:
        failures.append(
            f"业务能力未完成：expected={sorted(required_domains)!r}, actual={sorted(all_domains)!r}"
        )

    evidence_checks: dict[str, bool] = {}
    for domain, alternatives in (goal.get("evidence_any") or {}).items():
        # 同一领域列出的词采用 any-of；例如政策标题和来源任一能够核验即可。
        passed = any(str(term) in evidence_text for term in alternatives)
        evidence_checks[domain] = passed
        if not passed:
            failures.append(f"{domain} 缺少可核验证据，候选={alternatives!r}")

    no_action = bool(goal.get("no_action"))
    no_action_ok = not no_action or not all_domains
    if not no_action_ok:
        failures.append(f"边界场景发生不必要业务调用：{sorted(all_domains)!r}")

    output_ok = _output_ok(final.get("answer") or "")
    if not output_ok:
        failures.append("最终输出为空或包含错误标记")

    checks = {
        "context": all(context_checks.values()),
        "business_outcome": domains_ok,
        "evidence": all(evidence_checks.values()),
        "no_unnecessary_action": no_action_ok,
        "output": output_ok,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "actual": {
            "context": final_context,
            "domains": sorted(all_domains),
            "evidence_checks": evidence_checks,
            "answer": final.get("answer") or "",
        },
    }


def run_scenario(agent, scenario: dict, run_id: str, retries: int = 0) -> dict:
    last_error: str | None = None
    for attempt in range(retries + 1):
        thread_id = f"outcome-{run_id}-{scenario['id']}-attempt-{attempt}"
        turn_records: list[dict] = []
        try:
            for index, turn in enumerate(scenario["turns"], 1):
                planning_agent = getattr(agent, "planning_agent", None)
                if planning_agent is not None:
                    planning_agent.last_tasks = {}
                before = agent.get_history(thread_id)
                started = time.perf_counter()
                answer = agent.run(turn["user"], thread_id=thread_id)
                latency = time.perf_counter() - started
                snapshot = agent.get_snapshot(thread_id)
                history = agent.get_history(thread_id)[len(before):]
                calls = _tool_calls(history)
                subagent_status, subagent_outputs = _subagent_outputs(agent)
                merged_status = dict(snapshot.get("task_status") or {})
                for name, item in subagent_status.items():
                    merged_status.setdefault(name, item)
                domains = _action_domains(calls, merged_status)
                checkpoint_score = None
                if turn.get("checkpoint"):
                    checkpoint_score = _score_checkpoint(
                        turn["checkpoint"], snapshot, domains
                    )
                turn_records.append({
                    "turn": index,
                    "user": turn["user"],
                    "answer": answer,
                    "latency_s": round(latency, 3),
                    "snapshot": snapshot,
                    "tool_calls": calls,
                    "tool_outputs": _tool_outputs(history),
                    "parsed_tool_outputs": [
                        _parse_tool_output(text) for text in _tool_outputs(history)
                    ],
                    "subagent_status": subagent_status,
                    "subagent_outputs": subagent_outputs,
                    "action_domains": sorted(domains),
                    "checkpoint": checkpoint_score,
                })

            goal_score = _score_goal(scenario["goal"], turn_records)
            checkpoint_ok = all(
                turn.get("checkpoint") is None or turn["checkpoint"]["success"]
                for turn in turn_records
            )
            failures = list(goal_score["failures"])
            for turn in turn_records:
                if turn.get("checkpoint") and not turn["checkpoint"]["success"]:
                    failures.extend(turn["checkpoint"]["failures"])
            return {
                "id": scenario["id"],
                "category": scenario["category"],
                "interaction": scenario["interaction"],
                "style": scenario["style"],
                "source": scenario.get("source"),
                "goal": scenario["goal"],
                "success": goal_score["success"] and checkpoint_ok,
                "goal_score": goal_score,
                "checkpoint_success": checkpoint_ok,
                "failures": failures,
                "turns": turn_records,
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                continue
            return {
                "id": scenario["id"],
                "category": scenario["category"],
                "interaction": scenario["interaction"],
                "style": scenario["style"],
                "source": scenario.get("source"),
                "goal": scenario["goal"],
                "success": False,
                "goal_score": None,
                "checkpoint_success": False,
                "failures": [last_error],
                "turns": turn_records,
                "attempts": attempt + 1,
                "error": last_error,
            }
    raise AssertionError("unreachable")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[max(0, math.ceil(percentile * len(values)) - 1)], 3)


def summarize(records: list[dict]) -> dict:
    turns = [turn for record in records for turn in record.get("turns", [])]
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_interaction: dict[str, list[dict]] = defaultdict(list)
    by_style: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
        by_interaction[record["interaction"]].append(record)
        by_style[record["style"]].append(record)

    def breakdown(groups: dict[str, list[dict]]) -> dict:
        return {
            name: {
                "tasks": len(items),
                "success_rate": _safe_ratio(sum(item["success"] for item in items), len(items)),
            }
            for name, items in sorted(groups.items())
        }

    goal_records = [record for record in records if record.get("goal_score")]
    check_names = ("context", "business_outcome", "evidence", "no_unnecessary_action", "output")
    check_rates = {
        name: _safe_ratio(
            sum(record["goal_score"]["checks"].get(name, False) for record in goal_records),
            len(goal_records),
        )
        for name in check_names
    }
    checkpoint_records = [
        record for record in records
        if any(turn.get("checkpoint") for turn in record.get("turns", []))
    ]
    latencies = [turn["latency_s"] for turn in turns]
    return {
        "tasks": len(records),
        "turns": len(turns),
        "outcome_success_rate": _safe_ratio(sum(r["success"] for r in records), len(records)),
        "checkpoint_success_rate": _safe_ratio(
            sum(r["checkpoint_success"] for r in checkpoint_records),
            len(checkpoint_records),
        ),
        "check_rates": check_rates,
        "runtime_error_count": sum(bool(r.get("error")) for r in records),
        "average_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_s": _percentile(latencies, 0.95),
        "by_category": breakdown(grouped),
        "by_interaction": breakdown(by_interaction),
        "by_style": breakdown(by_style),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def write_reports(summary: dict, records: list[dict], tag: str, dataset_hash: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"outcome_agent_{tag}.json"
    markdown_path = RESULTS_DIR / f"outcome_agent_report_{tag}.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "scope": "结果导向的合成挑战集；不代表线上用户流量或农业建议专业准确率",
        "summary": summary,
        "records": records,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        f"# AgriAgent 结果导向挑战评测（{tag}）",
        "",
        "> 评测最终业务状态、能力执行和可核验证据，不要求固定内部路由；场景为合成挑战集，不代表线上流量。",
        "",
        f"- 数据集 SHA-256：`{dataset_hash}`",
        f"- 场景任务：{summary['tasks']} 项；对话：{summary['turns']} 轮",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 业务结果成功率 | {_fmt(summary['outcome_success_rate'])} |",
        f"| 缺参检查点通过率 | {_fmt(summary['checkpoint_success_rate'])} |",
        f"| 最终上下文正确率 | {_fmt(summary['check_rates']['context'])} |",
        f"| 所需业务能力完成率 | {_fmt(summary['check_rates']['business_outcome'])} |",
        f"| 可核验证据通过率 | {_fmt(summary['check_rates']['evidence'])} |",
        f"| 边界场景无误调用率 | {_fmt(summary['check_rates']['no_unnecessary_action'])} |",
        f"| 非错误输出率 | {_fmt(summary['check_rates']['output'])} |",
        f"| 平均单轮延迟 | {summary['average_latency_s']}s |",
        f"| P95 单轮延迟 | {summary['p95_latency_s']}s |",
        f"| 运行时异常数 | {summary['runtime_error_count']} |",
    ]

    for title, key in (("按业务类别", "by_category"), ("按交互状态", "by_interaction"), ("按表达风格", "by_style")):
        rows += ["", f"## {title}", "", "| 分组 | 任务数 | 成功率 |", "| --- | ---: | ---: |"]
        for name, item in summary[key].items():
            rows.append(f"| {name} | {item['tasks']} | {_fmt(item['success_rate'])} |")

    failed = [record for record in records if not record["success"]]
    if failed:
        rows += ["", "## 失败样本（前30项）", ""]
        for record in failed[:30]:
            rows.append(
                f"- `{record['id']}`：{'；'.join(record.get('failures') or ['未知失败'])}"
            )

    rows += [
        "",
        "## 指标边界",
        "",
        "- 业务结果成功要求最终上下文、所需能力执行、可核验证据、边界约束和非错误输出同时通过。",
        "- route、节点名称和工具顺序只保存在明细中，不作为主成功条件。",
        "- 天气默认使用冻结响应；政策与诊断使用仓库中的真实本地知识数据。",
        "- 不使用 LLM-as-Judge 自动评价自由文本建议的农业专业水平。",
    ]
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def install_frozen_weather() -> None:
    """将实时 MCP 天气替换为确定性快照；只影响当前评测进程。"""
    from agri_agent.agents import weather_agent
    from agri_agent.tools import agent_tools

    async def frozen_query(city: str) -> str:
        return FROZEN_WEATHER.get(
            city,
            f"{city}：多云，16-25℃，风力2级；冻结测试响应。",
        )

    weather_agent.query_weather = frozen_query
    agent_tools.query_weather = frozen_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "outcome_challenge_v2.jsonl")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--interaction", action="append", default=[])
    parser.add_argument("--style", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--live-weather", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=True)
    # 评测依赖本机已经缓存的BGE模型；禁止HuggingFace在每次启动时联网探测配置，
    # 避免“模型明明已缓存，却因HEAD请求失败等待多轮重试”。这不影响业务LLM API。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    missing = [
        key for key in ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")
        if not os.getenv(key)
    ]
    if missing:
        raise SystemExit(f"缺少环境变量：{', '.join(missing)}；已尝试读取 {args.env_file}")

    from agri_agent.agents.agri_graph import AgriGraphAgent

    scenarios = load_scenarios(args.dataset)
    full_hash = _dataset_hash(scenarios)
    if args.category:
        wanted = set(args.category)
        scenarios = [s for s in scenarios if s["category"] in wanted]
    if args.interaction:
        wanted = set(args.interaction)
        scenarios = [s for s in scenarios if s["interaction"] in wanted]
    if args.style:
        wanted = set(args.style)
        scenarios = [s for s in scenarios if s["style"] in wanted]
    if args.task_id:
        wanted = set(args.task_id)
        scenarios = [s for s in scenarios if s["id"] in wanted]
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("没有匹配的挑战场景")

    if not args.live_weather:
        install_frozen_weather()
    agent = AgriGraphAgent()
    # 结果导向主评测默认只测本地确定性政策链路；A2A故障/回退另由故障注入测试验证。
    if hasattr(agent, "planning_agent"):
        agent.planning_agent.policy_a2a_client = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = RESULTS_DIR / f"outcome_agent_{args.tag}.partial.json"
    records: list[dict] = []
    if args.resume and partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8"))
        completed = {record["id"] for record in records}
        scenarios = [scenario for scenario in scenarios if scenario["id"] not in completed]

    run_id = uuid4().hex[:8]
    total = len(records) + len(scenarios)
    print(f"加载 {len(scenarios)} 项结果导向挑战任务；数据集哈希 {full_hash[:12]}")
    for scenario in scenarios:
        record = run_scenario(agent, scenario, run_id, retries=max(0, args.retries))
        records.append(record)
        partial_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if len(records) % 5 == 0 or len(records) == total:
            rate = sum(record["success"] for record in records) / len(records)
            print(f"[{len(records)}/{total}] 当前业务结果成功率 {rate:.2%}", flush=True)

    summary = summarize(records)
    json_path, markdown_path = write_reports(summary, records, args.tag, full_hash)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
