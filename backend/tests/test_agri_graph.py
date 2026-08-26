"""统一AgriGraph的路由、槽位记忆、ReAct和Planning分支测试。"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LLM_API_KEY", "test-dummy-key")
os.environ.setdefault("LLM_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LLM_MODEL_ID", "test-dummy-model")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage

from agri_agent.agents.agri_graph import AgriGraphAgent, RouteDecision


class _ScriptedLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.received_messages = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.received_messages.append(messages)
        return self.responses.pop(0)


class _ScriptedRouter:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.received_messages = []

    def invoke(self, messages):
        self.received_messages.append(messages)
        return self.decisions.pop(0)


class _NeverRouter:
    def invoke(self, messages):
        raise AssertionError("结构化planning模式不应该调用意图路由模型")


class _FakePlanningAgent:
    def __init__(self):
        self.calls = []
        self.last_tasks = {}

    def run(self, **kwargs):
        self.calls.append(kwargs)
        self.last_tasks = {
            capability: SimpleNamespace(status="completed", error=None)
            for capability in kwargs.get("capabilities", [])
        }
        return "统一规划结果"


class _FakeVisionLLM:
    def invoke(self, messages):
        return "作物：水稻；症状：叶片发黄"


def _make_agent(router, llm=None, planning_agent=None):
    return AgriGraphAgent(
        llm=llm or _ScriptedLLM(),
        router=router,
        planning_agent=planning_agent or _FakePlanningAgent(),
        vision_llm=_FakeVisionLLM(),
    )


def test_direct_route_answers_without_tools_or_planning():
    router = _ScriptedRouter([
        RouteDecision(route="direct", confidence=0.95),
    ])
    llm = _ScriptedLLM([AIMessage(content="你好，我是智慧农事助手。")])
    planning = _FakePlanningAgent()
    agent = _make_agent(router, llm, planning)

    answer = agent.run("你好", thread_id="direct-thread")
    snapshot = agent.get_snapshot("direct-thread")

    assert answer == "你好，我是智慧农事助手。"
    assert snapshot["route"] == "direct"
    assert snapshot["capabilities"] == []
    assert planning.calls == []
    assert len(router.received_messages[0]) == 2
    assert "最近对话如下" in router.received_messages[0][-1].content
    assert "不得输出工具名" in llm.received_messages[0][0].content
    assert "调用工具的规则" not in llm.received_messages[0][0].content
    print("测试通过：普通寒暄走direct，不调用工具或Planning节点")


def test_empty_router_result_uses_keyword_fallback():
    router = _ScriptedRouter([None])
    llm = _ScriptedLLM([AIMessage(content="你好，我可以帮你处理农事问题。")])
    agent = _make_agent(router, llm)

    answer = agent.run("你好", thread_id="empty-router-thread")
    snapshot = agent.get_snapshot("empty-router-thread")

    assert answer == "你好，我可以帮你处理农事问题。"
    assert snapshot["route"] == "direct"
    assert snapshot["capabilities"] == []
    print("测试通过：结构化路由返回空结果时自动回退，不中断普通对话")


def test_followup_explanation_uses_recent_dialogue_instead_of_clarifying():
    router = _ScriptedRouter([
        RouteDecision(route="direct", confidence=0.95),
        RouteDecision(
            route="clarify",
            capabilities=["weather"],
            missing_fields=["city"],
            confidence=0.5,
        ),
    ])
    llm = _ScriptedLLM([
        AIMessage(content="当前回答里没有政策建议。"),
        AIMessage(content="因为你上一轮只询问了天气，没有提出政策需求。"),
    ])
    agent = _make_agent(router, llm)

    agent.run("最近必须采取什么措施", thread_id="followup-thread")
    answer = agent.run("为什么没有建议", thread_id="followup-thread")
    snapshot = agent.get_snapshot("followup-thread")
    second_router_input = router.received_messages[1][-1].content

    assert answer.startswith("因为你上一轮")
    assert snapshot["route"] == "direct"
    assert "助手：当前回答里没有政策建议" in second_router_input
    assert "用户：为什么没有建议" in second_router_input
    assert "立即结合已确认的目标和阶段补充3到5条" in llm.received_messages[1][0].content
    assert "不得只解释原因后让用户重新选择" in llm.received_messages[1][0].content
    print("测试通过：解释性追问会读取最近对话，不会脱离上下文重新追问意图")


def test_action_advice_followup_prefers_conversation_context():
    router = _ScriptedRouter([
        RouteDecision(route="direct", crop="水稻", region="江苏", confidence=0.95),
        RouteDecision(
            route="react",
            capabilities=["weather"],
            need="最近必须要采取的措施",
            confidence=0.8,
        ),
    ])
    llm = _ScriptedLLM([
        AIMessage(content="江苏当前是否适合种水稻，需要结合稻作类型和当地农时判断。"),
        AIMessage(content="近期先确认稻作类型和生育阶段，再决定田间措施。"),
    ])
    planning = _FakePlanningAgent()
    agent = _make_agent(router, llm, planning)

    agent.run("我想在江苏种水稻，现在适合种吗", thread_id="advice-followup-thread")
    answer = agent.run("你帮我看看最近必须要采取的措施", thread_id="advice-followup-thread")
    snapshot = agent.get_snapshot("advice-followup-thread")

    assert answer.startswith("近期先确认")
    assert snapshot["route"] == "direct"
    assert snapshot["context"]["crop"] == "水稻"
    assert snapshot["context"]["region"] == "江苏"
    assert "need" not in snapshot["context"]
    assert planning.calls == []
    print("测试通过：承接上文询问措施时直接利用上下文，不被误路由到缺字段流程")


def test_preplanting_answer_fills_pending_stage_and_blocks_diagnosis():
    router = _ScriptedRouter([
        RouteDecision(
            route="direct",
            crop="水稻",
            city="扬州",
            region="江苏",
            confidence=0.95,
        ),
        # 故意模拟概率路由器误判。程序必须用pending_slot和否定事实纠正它。
        RouteDecision(
            route="react",
            capabilities=["diagnosis"],
            confidence=0.8,
        ),
        RouteDecision(
            route="clarify",
            capabilities=["diagnosis"],
            missing_fields=["symptom"],
            confidence=0.8,
        ),
    ])
    llm = _ScriptedLLM([
        AIMessage(content="我先给你通用准备步骤。请问您目前处于哪个阶段？"),
        AIMessage(content="既然还没有开始种植，应先确认品种和下一季农时。"),
        AIMessage(content="明白，你尚未种植且没有症状，不需要进入病虫害诊断。"),
    ])
    planning = _FakePlanningAgent()
    agent = _make_agent(router, llm, planning)

    agent.run("我想在扬州种水稻，我该干什么", thread_id="preplanting-thread")
    first_snapshot = agent.get_snapshot("preplanting-thread")
    second_answer = agent.run("我还没有开始种植", thread_id="preplanting-thread")
    second_snapshot = agent.get_snapshot("preplanting-thread")
    third_answer = agent.run("还没有种植啊，还没有症状", thread_id="preplanting-thread")
    third_snapshot = agent.get_snapshot("preplanting-thread")

    assert first_snapshot["pending_slot"] == "growth_stage"
    assert second_answer.startswith("既然还没有开始")
    assert second_snapshot["route"] == "direct"
    assert second_snapshot["pending_slot"] is None
    assert second_snapshot["context"]["growth_stage"] == "未开始种植"
    assert second_snapshot["context"]["has_symptom"] is False
    assert second_snapshot["context"]["active_goal"] == "种植准备"
    assert third_answer.startswith("明白")
    assert third_snapshot["route"] == "direct"
    assert third_snapshot["capabilities"] == []
    assert third_snapshot["missing_fields"] == []
    assert planning.calls == []
    print("测试通过：未种植/无症状作为正式状态，能够覆盖LLM的诊断误判")


def test_frontend_history_rehydrates_empty_checkpointer_after_restart():
    router = _ScriptedRouter([
        RouteDecision(
            route="clarify",
            capabilities=["weather"],
            missing_fields=["city"],
            confidence=0.5,
        ),
    ])
    llm = _ScriptedLLM([
        AIMessage(content="上一条其实给了通用措施，你具体是觉得哪部分不够明确？"),
    ])
    agent = _make_agent(router, llm)
    restored_history = [
        {"role": "user", "content": "我想在江苏种水稻"},
        {"role": "assistant", "content": "我先给你一些通用种植建议。"},
    ]

    answer = agent.run(
        "为什么没有建议",
        thread_id="rehydrated-thread",
        history=restored_history,
        context={"crop": "水稻", "region": "江苏", "active_goal": "种植咨询"},
    )
    serialized = agent.get_history("rehydrated-thread")
    router_input = router.received_messages[0][-1].content

    assert answer.startswith("上一条其实")
    assert [item["role"] for item in serialized] == ["user", "assistant", "user", "assistant"]
    assert "助手：我先给你一些通用种植建议" in router_input
    assert agent.get_snapshot("rehydrated-thread")["route"] == "direct"
    print("测试通过：后端重启后可从前端历史恢复对话，不再出现可见历史与图状态错位")


def test_single_capability_is_not_forced_into_planning():
    router = _ScriptedRouter([
        RouteDecision(
            route="planning",
            capabilities=["weather"],
            crop="水稻",
            city="南京",
            region="江苏",
            confidence=0.95,
        ),
    ])
    llm = _ScriptedLLM([AIMessage(content="这是单项天气回答。")])
    planning = _FakePlanningAgent()
    agent = _make_agent(router, llm, planning)

    answer = agent.run("南京现在的天气适合打药吗", thread_id="single-capability-thread")
    snapshot = agent.get_snapshot("single-capability-thread")

    assert answer == "这是单项天气回答。"
    assert snapshot["route"] == "react"
    assert snapshot["capabilities"] == ["weather"]
    assert planning.calls == []
    print("测试通过：单一能力不会被模型误判后硬塞进综合Planning")


def test_province_name_is_not_accepted_as_weather_city():
    router = _ScriptedRouter([
        RouteDecision(
            route="react",
            capabilities=["weather"],
            crop="水稻",
            city="江苏",
            region="江苏",
            confidence=0.95,
        ),
    ])
    agent = _make_agent(router)

    answer = agent.run("我想在江苏种水稻，现在适合种吗", thread_id="province-city-thread")
    snapshot = agent.get_snapshot("province-city-thread")

    assert "城市" in answer
    assert snapshot["route"] == "clarify"
    assert snapshot["context"]["region"] == "江苏"
    assert "city" not in snapshot["context"]
    print("测试通过：省级地区不会冒充城市去查询天气")


def test_missing_slot_is_clarified_and_completed_on_next_turn():
    router = _ScriptedRouter([
        RouteDecision(
            route="react",
            capabilities=["diagnosis"],
            symptom="叶片发黄",
            confidence=0.95,
        ),
        RouteDecision(
            # 故意模拟模型把“水稻”这种短补充误判成direct；程序应该根据上一轮
            # missing_fields发现crop已补齐，并恢复原来的diagnosis任务。
            route="direct",
            crop="水稻",
            confidence=0.95,
        ),
    ])
    llm = _ScriptedLLM([AIMessage(content="我会根据水稻叶片发黄的情况给出建议。")])
    agent = _make_agent(router, llm)

    first_answer = agent.run("叶片最近发黄", thread_id="slot-thread")
    first_snapshot = agent.get_snapshot("slot-thread")
    second_answer = agent.run("水稻", thread_id="slot-thread")
    second_snapshot = agent.get_snapshot("slot-thread")

    assert "作物" in first_answer
    assert first_snapshot["route"] == "clarify"
    assert first_snapshot["context"]["symptom"] == "叶片发黄"
    assert second_answer.startswith("我会根据水稻")
    assert second_snapshot["route"] == "react"
    assert second_snapshot["context"]["crop"] == "水稻"
    assert second_snapshot["context"]["symptom"] == "叶片发黄"
    print("测试通过：缺字段时先追问，下一轮补充后沿用之前槽位继续执行")


def test_react_route_executes_tool_and_preserves_frontend_trace():
    router = _ScriptedRouter([
        RouteDecision(
            route="react",
            capabilities=["diagnosis"],
            crop="水稻",
            symptom="叶片发黄",
            confidence=0.95,
        ),
    ])
    tool_call_id = "call-agri-graph-1"
    llm = _ScriptedLLM([
        AIMessage(content="", tool_calls=[{
            "name": "diagnose_crop_disease",
            "args": {"crop": "水稻", "symptom_text": "叶片发黄"},
            "id": tool_call_id,
        }]),
        AIMessage(content="根据知识库结果，建议先检查排水和施肥。"),
    ])

    with patch(
        "agri_agent.tools.agent_tools.match_pest_knowledge",
        return_value=[{"causes": ["可能积水"], "recommendations": ["及时排水"]}],
    ):
        agent = _make_agent(router, llm)
        answer = agent.run("水稻叶片发黄怎么办", thread_id="react-thread")

    history = agent.get_history("react-thread")
    assert answer.startswith("根据知识库结果")
    assert [message["role"] for message in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["id"] == tool_call_id
    assert history[2]["tool_call_id"] == tool_call_id
    print("测试通过：react路线执行真实ToolNode，并保留前端工具轨迹所需消息")


def test_multiple_capabilities_enter_selective_planning():
    router = _ScriptedRouter([
        RouteDecision(
            route="react",
            capabilities=["diagnosis", "weather"],
            crop="水稻",
            city="长沙",
            symptom="叶片发黄",
            confidence=0.95,
        ),
    ])
    planning = _FakePlanningAgent()
    agent = _make_agent(router, planning_agent=planning)

    answer = agent.run("水稻叶片发黄，长沙明天适合打药吗", thread_id="multi-thread")
    snapshot = agent.get_snapshot("multi-thread")

    assert answer == "统一规划结果"
    assert snapshot["route"] == "planning"
    assert planning.calls[0]["capabilities"] == ["diagnosis", "weather"]
    assert planning.calls[0]["region"] == "湖南省"
    print("测试通过：两个能力需求自动升级为planning，并补全高置信城市所属省份")


def test_structured_form_forces_planning_and_bypasses_router():
    planning = _FakePlanningAgent()
    agent = _make_agent(_NeverRouter(), planning_agent=planning)

    answer = agent.run(
        "请生成完整农事计划",
        thread_id="form-thread",
        mode="planning",
        context={
            "crop": "水稻",
            "city": "长沙",
            "region": "湖南省",
            "symptom": "叶片发黄",
            "need": "种植补贴",
        },
    )
    snapshot = agent.get_snapshot("form-thread")

    assert answer == "统一规划结果"
    assert snapshot["route"] == "planning"
    assert snapshot["capabilities"] == ["diagnosis", "weather", "policy"]
    assert snapshot["context"]["region"] == "湖南省"
    assert planning.calls[0]["capabilities"] == ["diagnosis", "weather", "policy"]
    print("测试通过：结构化表单直接进入Planning节点，不浪费一次意图模型调用")


if __name__ == "__main__":
    test_direct_route_answers_without_tools_or_planning()
    test_empty_router_result_uses_keyword_fallback()
    test_followup_explanation_uses_recent_dialogue_instead_of_clarifying()
    test_action_advice_followup_prefers_conversation_context()
    test_preplanting_answer_fills_pending_stage_and_blocks_diagnosis()
    test_frontend_history_rehydrates_empty_checkpointer_after_restart()
    test_single_capability_is_not_forced_into_planning()
    test_province_name_is_not_accepted_as_weather_city()
    test_missing_slot_is_clarified_and_completed_on_next_turn()
    test_react_route_executes_tool_and_preserves_frontend_trace()
    test_multiple_capabilities_enter_selective_planning()
    test_structured_form_forces_planning_and_bypasses_router()
    print("\n全部测试通过")
