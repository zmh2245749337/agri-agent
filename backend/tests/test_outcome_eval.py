import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND / "eval"))

from build_outcome_eval import FAMILIES, INTERACTIONS, STYLES, build_scenarios
from outcome_agent_eval import (
    _dataset_hash,
    _subagent_outputs,
    _score_checkpoint,
    _score_goal,
    run_scenario,
    summarize,
)


def test_subagent_outputs_include_structured_evidence_for_auditability():
    task = SimpleNamespace(
        status="completed",
        protocol="local",
        fallback_reason=None,
        result_text="未来几天有雨，暂缓施药。",
        evidence={"city": "长沙", "weather_data": "长沙：小雨，北风3级"},
    )
    agent = SimpleNamespace(planning_agent=SimpleNamespace(last_tasks={"weather": task}))

    statuses, outputs = _subagent_outputs(agent)

    assert statuses["weather"]["status"] == "completed"
    assert any("长沙" in output and "北风3级" in output for output in outputs)


def test_three_domain_missing_scenarios_omit_city_not_inferable_region():
    scenarios = build_scenarios()
    missing = next(
        item for item in scenarios
        if item["id"] == "v2_diagnosis_weather_policy_missing_standard"
    )

    assert missing["turns"][0]["checkpoint"]["missing_contains"] == ["city"]
    assert missing["goal"]["context"]["region"] in missing["turns"][0]["user"]
    assert missing["goal"]["context"]["city"] not in missing["turns"][0]["user"]


def test_outcome_dataset_has_coverage_driven_120_scenarios():
    scenarios = build_scenarios()
    categories = Counter(item["category"] for item in scenarios)
    interactions = Counter(item["interaction"] for item in scenarios)
    styles = Counter(item["style"] for item in scenarios)

    assert len(scenarios) == 120
    assert sum(len(item["turns"]) for item in scenarios) == 204
    assert len({item["id"] for item in scenarios}) == 120
    assert len({tuple(turn["user"] for turn in item["turns"]) for item in scenarios}) == 120
    assert categories["boundary_no_action"] == 8
    for family in FAMILIES:
        assert categories[family] == len(INTERACTIONS) * len(STYLES) == 16
    for interaction in INTERACTIONS:
        assert interactions[interaction] == len(FAMILIES) * len(STYLES) == 28
    for style in STYLES:
        assert styles[style] == len(FAMILIES) * len(INTERACTIONS) == 28


def test_outcome_dataset_has_a_stable_content_hash():
    first = _dataset_hash(build_scenarios())
    second = _dataset_hash(build_scenarios())

    assert first == second
    assert len(first) == 64


def test_primary_goal_scoring_uses_outcome_and_evidence_not_route():
    goal = {
        "context": {"crop": "水稻", "city": "长沙"},
        "required_domains": ["diagnosis", "weather"],
        "evidence_any": {
            "diagnosis": ["稻瘟病"],
            "weather": ["长沙"],
        },
        "no_action": False,
    }
    turns = [{
        "snapshot": {
            # route 故意不放进目标评分；只要业务终态正确就能通过。
            "route": "custom-route",
            "context": {"crop": "水稻", "city": "长沙"},
        },
        "action_domains": ["diagnosis", "weather"],
        "tool_outputs": ["命中稻瘟病", "长沙：小雨"],
        "subagent_outputs": [],
        "answer": "建议避开降雨时段，并根据诊断结果处理。",
    }]

    scored = _score_goal(goal, turns)

    assert scored["success"] is True
    assert scored["checks"]["business_outcome"] is True
    assert scored["checks"]["evidence"] is True


def test_primary_goal_fails_when_action_runs_without_gold_evidence():
    goal = {
        "context": {"crop": "水稻"},
        "required_domains": ["diagnosis"],
        "evidence_any": {"diagnosis": ["稻瘟病"]},
        "no_action": False,
    }
    turns = [{
        "snapshot": {"context": {"crop": "水稻"}},
        "action_domains": ["diagnosis"],
        "tool_outputs": ["没有找到匹配条目"],
        "subagent_outputs": [],
        "answer": "暂时没有找到。",
    }]

    scored = _score_goal(goal, turns)

    assert scored["success"] is False
    assert scored["checks"]["business_outcome"] is True
    assert scored["checks"]["evidence"] is False


def test_missing_checkpoint_requires_slot_and_blocks_early_action():
    checkpoint = {"missing_contains": ["city"], "forbid_domain_actions": True}

    passed = _score_checkpoint(
        checkpoint,
        {"missing_fields": ["city"], "pending_slot": "city"},
        set(),
    )
    failed = _score_checkpoint(
        checkpoint,
        {"missing_fields": [], "pending_slot": None},
        {"weather"},
    )

    assert passed["success"] is True
    assert failed["success"] is False
    assert failed["checks"]["missing_detected"] is False
    assert failed["checks"]["no_early_action"] is False


def test_summary_reports_outcome_dimensions():
    record = {
        "id": "demo",
        "category": "diagnosis",
        "interaction": "complete",
        "style": "colloquial",
        "success": True,
        "checkpoint_success": True,
        "error": None,
        "goal_score": {
            "checks": {
                "context": True,
                "business_outcome": True,
                "evidence": True,
                "no_unnecessary_action": True,
                "output": True,
            }
        },
        "turns": [{"latency_s": 1.0, "checkpoint": None}],
    }

    summary = summarize([record])

    assert summary["outcome_success_rate"] == 1.0
    assert summary["check_rates"]["evidence"] == 1.0
    assert summary["by_category"]["diagnosis"]["success_rate"] == 1.0


def test_runner_executes_public_agent_interface_and_scores_business_outcome():
    class PlanningStub:
        last_tasks = {}

    class AgentStub:
        def __init__(self):
            self.planning_agent = PlanningStub()
            self.histories = {}

        def get_history(self, thread_id):
            return list(self.histories.get(thread_id, []))

        def run(self, user_message, thread_id):
            self.histories.setdefault(thread_id, []).extend([
                {"role": "user", "content": user_message},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "diagnose_crop_disease",
                            "arguments": '{"crop":"水稻","symptom_text":"褐斑"}',
                        }
                    }],
                },
                {"role": "tool", "content": '{"causes":["稻瘟病"]}'},
                {"role": "assistant", "content": "知识库结果提示可能是稻瘟病。"},
            ])
            return "知识库结果提示可能是稻瘟病。"

        def get_snapshot(self, thread_id):
            return {
                "route": "react",
                "context": {"crop": "水稻"},
                "missing_fields": [],
                "pending_slot": None,
                "task_status": {},
            }

    scenario = {
        "id": "runner_demo",
        "category": "diagnosis",
        "interaction": "complete",
        "style": "colloquial",
        "source": "unit_test",
        "turns": [{"user": "水稻叶上有褐色梭形斑，帮我判断。"}],
        "goal": {
            "context": {"crop": "水稻"},
            "required_domains": ["diagnosis"],
            "evidence_any": {"diagnosis": ["稻瘟病"]},
            "no_action": False,
        },
    }

    record = run_scenario(AgentStub(), scenario, "test")

    assert record["success"] is True
    assert record["goal_score"]["checks"]["evidence"] is True
