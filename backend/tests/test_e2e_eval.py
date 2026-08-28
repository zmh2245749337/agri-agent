import sys
from collections import Counter
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND / "eval"))

from build_e2e_eval import build_tasks
from e2e_agent_eval import _score_turn, summarize


def test_e2e_dataset_has_140_tasks_and_required_coverage():
    tasks = build_tasks()
    categories = Counter(task["category"] for task in tasks)

    assert len(tasks) == 140
    assert len({tuple(turn["user"] for turn in task["turns"]) for task in tasks}) == 140
    assert sum(len(task["turns"]) for task in tasks) == 164
    assert categories["single_tool_diagnosis"] == 20
    assert categories["single_tool_weather"] == 20
    assert categories["single_tool_policy"] == 20
    assert sum(count for name, count in categories.items() if name.startswith("multi_agent_")) == 24
    assert sum(count for name, count in categories.items() if name.startswith("multi_turn_")) == 24


def test_turn_scoring_requires_route_state_tool_and_output():
    expect = {
        "route": "react",
        "capabilities": ["diagnosis"],
        "context": {"crop": "水稻"},
        "missing_fields": [],
        "tools": ["diagnose_crop_disease"],
        "tasks": [],
    }
    snapshot = {
        "route": "react",
        "capabilities": ["diagnosis"],
        "context": {"crop": "水稻"},
        "missing_fields": [],
        "task_status": {},
    }
    history = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "diagnose_crop_disease", "args": {}}],
    }]

    passed = _score_turn(expect, snapshot, history, "建议先检查田间排水。")
    failed = _score_turn(expect, snapshot, [], "建议先检查田间排水。")

    assert passed["success"] is True
    assert failed["success"] is False
    assert failed["checks"]["tools"] is False


def test_summary_reports_task_and_tool_metrics():
    record = {
        "id": "demo",
        "category": "single_tool_diagnosis",
        "success": True,
        "error": None,
        "turns": [{
            "success": True,
            "latency_s": 1.2,
            "checks": {
                "route": True,
                "capabilities": True,
                "context": True,
                "missing_fields": True,
                "tools": True,
                "tasks_selected": True,
                "tasks_completed": True,
                "output": True,
            },
            "expect": {
                "route": "react",
                "capabilities": ["diagnosis"],
                "context": {"crop": "水稻"},
                "missing_fields": [],
                "tools": ["diagnose_crop_disease"],
                "tasks": [],
            },
            "actual": {
                "route": "react",
                "capabilities": ["diagnosis"],
                "context": {"crop": "水稻"},
                "missing_fields": [],
                "tools": ["diagnose_crop_disease"],
                "task_status": {},
                "answer": "ok",
            },
        }],
    }

    summary = summarize([record])

    assert summary["task_success_rate"] == 1.0
    assert summary["tool_f1"] == 1.0
    assert summary["context_field_accuracy"] == 1.0
    assert summary["p95_latency_s"] == 1.2
