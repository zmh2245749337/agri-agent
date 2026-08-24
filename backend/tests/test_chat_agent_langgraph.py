# tests/test_chat_agent_langgraph.py
"""
chat_agent_langgraph.py的测试。核心是验证"图连线对不对"：agent节点收到没有
tool_calls的回复时循环正确终止、收到tool_calls时正确路由到tools节点执行并把
结果带回agent节点、MemorySaver按thread_id正确续接/隔离记忆、SystemMessage
固定id不会跨轮重复堆积。

用一个"按顺序吐预设回复"的假LLM（_ScriptedLLM）注入进ChatAgentLangGraph，
不需要真实调用大模型；工具执行这一步用unittest.mock.patch替换掉底层的
match_pest_knowledge等函数，不需要真实的RapidFuzz/网络/MCP——这样测试
验证的是"图逻辑本身对不对"，跟"RapidFuzz匹配准不准""网络请求成不成功"
是两件独立的事，后者已经在其它测试文件里覆盖过了。
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

# 允许在没有配置真实.env的机器上（比如CI）也能跑这份测试——ChatAgentLangGraph
# 内部无条件会构造一个MyLLM(用于看图)，MyLLM要求这三个环境变量非空，但测试里
# 用不到看图功能的真实网络调用，dummy值够用了
os.environ.setdefault("LLM_API_KEY", "test-dummy-key")
os.environ.setdefault("LLM_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LLM_MODEL_ID", "test-dummy-model")

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage

from agri_agent.agents.chat_agent_langgraph import ChatAgentLangGraph


class _ScriptedLLM:
    """按顺序返回预设的AIMessage，用来精确控制Function Calling循环走几轮、
    每轮模型"说"什么，不需要真实调用大模型。received_messages记录每次
    invoke()收到的完整消息列表，方便断言"某一轮模型看到的上下文里有没有
    包含预期的历史/工具结果" """

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.received_messages = []

    def bind_tools(self, tools):
        return self  # 测试场景不需要真的按工具生成schema，直接返回自己

    def invoke(self, messages):
        self.received_messages.append(messages)
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def test_run_returns_direct_answer_when_no_tool_call_needed():
    fake_llm = _ScriptedLLM([AIMessage(content="你好，我能帮你查天气/诊断/政策")])
    agent = ChatAgentLangGraph(llm=fake_llm)

    answer = agent.run("你好", thread_id="t-greeting")

    assert answer == "你好，我能帮你查天气/诊断/政策"
    assert fake_llm._call_count == 1  # 不需要工具调用时，只应该问一次模型
    print("测试通过：模型直接给最终答案时，循环只跑一轮就正确结束")


def test_run_executes_tool_call_then_returns_final_answer():
    tool_call_id = "call_abc123"
    responses = [
        AIMessage(content=None, tool_calls=[
            {"name": "diagnose_crop_disease", "args": {"crop": "水稻", "symptom_text": "叶子发黄"}, "id": tool_call_id}
        ]),
        AIMessage(content="根据检索结果，建议检查排水和施肥情况"),
    ]
    fake_llm = _ScriptedLLM(responses)

    with patch(
        "agri_agent.agents.chat_agent_langgraph.match_pest_knowledge",
        return_value=[{"causes": ["纹枯病"], "recommendations": ["检查排水", "适量追肥"]}],
    ) as mock_match:
        agent = ChatAgentLangGraph(llm=fake_llm)
        answer = agent.run("我家水稻叶子发黄", thread_id="t-diagnose")

    assert answer == "根据检索结果，建议检查排水和施肥情况"
    mock_match.assert_called_once_with("水稻", "叶子发黄")
    assert fake_llm._call_count == 2  # 第一轮要求调用工具，第二轮才给最终答案

    # 第二次问模型时，上下文里应该已经包含第一轮工具执行的结果
    second_round_messages = fake_llm.received_messages[1]
    tool_messages = [m for m in second_round_messages if getattr(m, "tool_call_id", None) == tool_call_id]
    assert len(tool_messages) == 1
    assert "纹枯病" in tool_messages[0].content
    print("测试通过：模型要求调用工具时，正确路由到tools节点执行、结果正确带回下一轮")


def test_memory_persists_across_calls_with_same_thread_id():
    fake_llm = _ScriptedLLM([
        AIMessage(content="我在长沙，最近适合打药"),
        AIMessage(content="湖南省有水稻种植补贴可以申请"),
    ])
    agent = ChatAgentLangGraph(llm=fake_llm)

    agent.run("我在长沙种水稻，最近适合打药吗？", thread_id="t-memory")
    agent.run("那我这边有没有种植补贴，我在湖南省", thread_id="t-memory")

    # 第二轮模型看到的消息列表，应该包含第一轮的用户提问和模型回复（证明
    # MemorySaver真的把上一轮的对话接上了，不是每次都从空白开始）
    second_call_messages = fake_llm.received_messages[1]
    contents = [m.content for m in second_call_messages if m.content]
    assert any("长沙" in c for c in contents)
    assert any("我在长沙，最近适合打药" in c for c in contents)
    print("测试通过：相同thread_id多次调用，MemorySaver正确接上了历史对话")


def test_different_thread_ids_do_not_share_memory():
    fake_llm = _ScriptedLLM([
        AIMessage(content="第一个对话的回答"),
        AIMessage(content="第二个对话的回答"),
    ])
    agent = ChatAgentLangGraph(llm=fake_llm)

    agent.run("这是thread A的问题", thread_id="thread-a")
    agent.run("这是thread B的问题", thread_id="thread-b")

    # thread-b这一轮看到的消息里，不应该出现thread-a的提问内容——不同thread_id
    # 应该是互相隔离的独立对话，不能串
    second_call_messages = fake_llm.received_messages[1]
    contents = [m.content for m in second_call_messages if m.content]
    assert not any("thread A" in c for c in contents)
    print("测试通过：不同thread_id的对话互相隔离，不会串上下文")


def test_system_message_not_duplicated_across_turns():
    fake_llm = _ScriptedLLM([
        AIMessage(content="第一轮回答"),
        AIMessage(content="第二轮回答"),
    ])
    agent = ChatAgentLangGraph(llm=fake_llm)

    agent.run("第一轮问题", thread_id="t-system-dedup")
    agent.run("第二轮问题", thread_id="t-system-dedup")

    config = {"configurable": {"thread_id": "t-system-dedup"}}
    final_messages = agent.compiled_graph.get_state(config).values["messages"]
    system_messages = [m for m in final_messages if getattr(m, "id", None) == "system-prompt"]

    assert len(system_messages) == 1, "SystemMessage固定id应该让它每轮被替换而不是重复堆积"
    print("测试通过：多轮对话后，SystemMessage在历史里始终只有一份，没有被重复堆积")


def test_image_data_url_appends_description_to_user_message():
    fake_llm = _ScriptedLLM([AIMessage(content="根据图片描述，建议进一步检查纹枯病")])

    with patch.object(ChatAgentLangGraph, "_describe_image", return_value="作物种类：水稻；症状描述：叶片发黄"):
        agent = ChatAgentLangGraph(llm=fake_llm)
        agent.run("你看看这个", thread_id="t-image", image_data_url="data:image/jpeg;base64,fakebase64")

    sent_messages = fake_llm.received_messages[0]
    human_contents = [m.content for m in sent_messages if type(m).__name__ == "HumanMessage"]
    assert any("作物种类：水稻" in c for c in human_contents)
    print("测试通过：图片识别成功时，识别结果被正确拼进了用户消息里再交给模型")


def test_image_description_failure_falls_back_gracefully():
    fake_llm = _ScriptedLLM([AIMessage(content="图片处理暂时失败，建议改用文字描述症状")])

    with patch.object(ChatAgentLangGraph, "_describe_image", side_effect=RuntimeError("视觉模型超时")):
        agent = ChatAgentLangGraph(llm=fake_llm)
        # 不应该抛异常，应该优雅降级
        answer = agent.run("你看看这个", thread_id="t-image-fail", image_data_url="data:image/jpeg;base64,fakebase64")

    assert answer == "图片处理暂时失败，建议改用文字描述症状"
    sent_messages = fake_llm.received_messages[0]
    human_contents = [m.content for m in sent_messages if type(m).__name__ == "HumanMessage"]
    assert any("视觉模型超时" in c for c in human_contents)
    print("测试通过：图片识别失败时不会让整轮对话崩掉，正确降级并如实告知失败原因")


if __name__ == "__main__":
    test_run_returns_direct_answer_when_no_tool_call_needed()
    test_run_executes_tool_call_then_returns_final_answer()
    test_memory_persists_across_calls_with_same_thread_id()
    test_different_thread_ids_do_not_share_memory()
    test_system_message_not_duplicated_across_turns()
    test_image_data_url_appends_description_to_user_message()
    test_image_description_failure_falls_back_gracefully()
    print("\n全部测试通过")
