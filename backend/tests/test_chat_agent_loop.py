# tests/test_chat_agent_loop.py
"""
验证ChatAgent的Function Calling循环逻辑本身对不对：
- 模型要求调用工具时，能正确执行、把结果按OpenAI协议格式塞回messages
- 模型不再要求调用工具时，循环正确结束、返回最终答案
- 连续调用两轮（带history）时，记忆确实在起作用
- 超过MAX_TOOL_ROUNDS时有兜底，不会死循环

用假的MyLLM（脚本化返回值）+ mock掉_execute_tool，只测"循环控制逻辑"本身，
不依赖真实大模型、真实工具、真实网络
"""
import sys
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# ChatAgent现在默认会构造一个真实的MyLLM(model=VISION_MODEL)作为vision_llm（除非显式传入），
# 就算测试里只关心chat_with_tools那部分逻辑、不会真的调用vision_llm.invoke()，
# MyLLM.__init__()本身也需要读到model/api_key/base_url这几个环境变量才能构造成功，
# 所以这里给一个假的环境变量，跟test_myllm_retry.py是同一个做法
os.environ.setdefault("LLM_MODEL_ID", "fake-model")
os.environ.setdefault("LLM_API_KEY", "fake-key")
os.environ.setdefault("LLM_BASE_URL", "http://fake")

from agri_agent.agents.chat_agent import ChatAgent


class FakeLLM:
    """按顺序返回脚本化的assistant消息，每调用一次chat_with_tools就吐出列表里的下一个"""
    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.call_count = 0

    def chat_with_tools(self, messages, tools, **kwargs):
        self.call_count += 1
        return self.scripted_responses.pop(0)


def _tool_call(tool_id, name, arguments_dict):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments_dict, ensure_ascii=False)),
    )


def _assistant_msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def test_single_tool_call_then_final_answer():
    """第一轮模型要求调用get_weather_forecast，第二轮不再要求工具、直接给答案"""
    fake_llm = FakeLLM([
        _assistant_msg(content=None, tool_calls=[_tool_call("call_1", "get_weather_forecast", {"city": "长沙"})]),
        _assistant_msg(content="长沙近期多雨，不适合打药，建议先做好排水。", tool_calls=None),
    ])

    with patch.object(ChatAgent, "_execute_tool", return_value={"weather_data": "多云转雨"}):
        agent = ChatAgent(llm=fake_llm)
        answer, history = agent.run("我在长沙种水稻，最近适合打药吗？")

    assert fake_llm.call_count == 2
    assert "排水" in answer
    # history里应该依次是：user、assistant(带tool_calls)、tool(工具结果)、assistant(最终答案)
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert history[2]["tool_call_id"] == "call_1"
    print("测试通过：单次工具调用后给出最终答案，history结构符合OpenAI协议格式")


def test_no_tool_call_needed():
    """模型判断不需要调用任何工具，第一轮就直接给答案"""
    fake_llm = FakeLLM([_assistant_msg(content="你好，我能帮你查天气、诊断作物问题、查补贴政策。", tool_calls=None)])

    agent = ChatAgent(llm=fake_llm)
    answer, history = agent.run("你好，你能做什么？")

    assert fake_llm.call_count == 1
    assert "补贴" in answer
    print("测试通过：不需要工具时，第一轮就直接返回答案，不会强行调用工具")


def test_memory_carries_across_two_runs():
    """验证第二轮调用时，传入的history确实被拼进了发给模型的messages里（记忆生效）"""
    fake_llm = FakeLLM([_assistant_msg(content="长沙近期多雨，暂缓打药。", tool_calls=None)])
    agent = ChatAgent(llm=fake_llm)
    _, history_after_round1 = agent.run("我在长沙种水稻，最近适合打药吗？")

    fake_llm2 = FakeLLM([_assistant_msg(content="湖南省有耕地地力保护补贴，约100元/亩。", tool_calls=None)])
    agent2 = ChatAgent(llm=fake_llm2)

    captured_messages = {}
    original_chat = fake_llm2.chat_with_tools
    def spy_chat_with_tools(messages, tools, **kwargs):
        captured_messages["messages"] = messages
        return original_chat(messages, tools, **kwargs)
    fake_llm2.chat_with_tools = spy_chat_with_tools

    agent2.run("那我这边有没有种植补贴可以申请，我在湖南省", history=history_after_round1)

    # 第二轮实际发给模型的messages里，应该包含第一轮的问答内容（记忆生效的证据）
    contents = [m.get("content", "") for m in captured_messages["messages"]]
    assert any("适合打药" in c for c in contents if c)
    assert any("暂缓打药" in c for c in contents if c)
    print("测试通过：第二轮调用时，第一轮的对话内容确实被带进了发给模型的messages里")


def test_llm_call_fails_returns_graceful_message_not_crash():
    """MyLLM.chat_with_tools持续失败（比如限流重试用完后抛出异常）时，
    ChatAgent不应该让异常往外传播崩溃，应该返回一句友好提示"""
    class AlwaysFailingLLM:
        def chat_with_tools(self, messages, tools, **kwargs):
            raise RuntimeError("429 rate limited (fake)")

    agent = ChatAgent(llm=AlwaysFailingLLM())
    answer, history = agent.run("我在长沙种水稻，最近适合打药吗？")

    assert "抱歉" in answer
    assert "429" in answer  # 把具体失败原因带出来，方便排查，不是笼统一句"出错了"
    assert history[-1]["role"] == "assistant"
    print("测试通过：大模型调用持续失败时，返回友好提示而不是让异常崩溃整个请求")


def test_image_description_merged_into_user_message():
    """有image_data_url时，应该先调用vision_llm.invoke()拿到症状描述，
    再把描述文字拼进用户消息，交给主对话模型（self.llm）"""
    class FakeVisionLLM:
        def __init__(self):
            self.received_messages = None

        def invoke(self, messages):
            self.received_messages = messages
            return "叶片边缘发黄，有褐色斑点"

    fake_vision = FakeVisionLLM()
    fake_llm = FakeLLM([_assistant_msg(content="看起来像是纹枯病，建议排查田间湿度。", tool_calls=None)])

    agent = ChatAgent(llm=fake_llm, vision_llm=fake_vision)
    answer, history = agent.run("帮我看看这个", image_data_url="data:image/jpeg;base64,FAKEDATA")

    # vision_llm确实被调用了，而且传的消息里带了图片这个content block
    assert fake_vision.received_messages is not None
    image_blocks = [b for b in fake_vision.received_messages[0]["content"] if b["type"] == "image_url"]
    assert image_blocks and image_blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,FAKEDATA"

    # 最终交给主对话模型的用户消息里，应该包含视觉模型给出的症状描述
    assert "叶片边缘发黄" in history[0]["content"]
    assert "看起来像是纹枯病" in answer
    print("测试通过：有图片时，视觉模型的描述被正确拼进了用户消息，再交给主对话模型")


def test_image_description_failure_degrades_gracefully():
    """vision_llm.invoke()失败时，不应该让整个run()崩溃，应该退化成
    "只处理文字部分+告知图片识别失败"，交给主对话模型自己决定怎么回应"""
    class FailingVisionLLM:
        def invoke(self, messages):
            raise RuntimeError("视觉模型超时（模拟）")

    fake_llm = FakeLLM([_assistant_msg(content="图片识别暂时失败了，能麻烦您用文字描述一下症状吗？", tool_calls=None)])
    agent = ChatAgent(llm=fake_llm, vision_llm=FailingVisionLLM())

    answer, history = agent.run("帮我看看这个", image_data_url="data:image/jpeg;base64,FAKEDATA")

    assert "图片识别失败" in history[0]["content"]  # 主对话模型收到的消息里带了失败提示
    assert "识别暂时失败" in answer  # 没有崩溃，走完了正常流程并给出了回应
    print("测试通过：视觉模型调用失败时，能优雅降级，不会让整个对话崩溃")


def test_no_image_skips_vision_call():
    """没有传image_data_url时，不应该调用vision_llm，避免不必要的模型调用"""
    class FakeVisionLLM:
        def __init__(self):
            self.called = False

        def invoke(self, messages):
            self.called = True
            return "不应该被调用"

    fake_vision = FakeVisionLLM()
    fake_llm = FakeLLM([_assistant_msg(content="你好！", tool_calls=None)])
    agent = ChatAgent(llm=fake_llm, vision_llm=fake_vision)

    agent.run("你好")

    assert not fake_vision.called
    print("测试通过：没有图片时，不会多此一举地调用视觉模型")


def test_trim_history_keeps_whole_turns_only():
    """构造15轮历史（每轮结构不同：有的只有user+assistant，有的还带一次工具调用），
    裁剪到MAX_HISTORY_TURNS=10轮后：轮数正确、且不会把某一轮的工具调用切掉一半"""
    history = []
    for i in range(15):
        history.append({"role": "user", "content": f"第{i}轮问题"})
        if i % 3 == 0:
            # 每隔几轮模拟一次带工具调用的轮次
            history.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            })
            history.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "{}"})
        history.append({"role": "assistant", "content": f"第{i}轮回答"})

    agent = ChatAgent(llm=FakeLLM([]))
    trimmed = agent._trim_history(history)

    user_count = sum(1 for m in trimmed if m["role"] == "user")
    assert user_count == ChatAgent.MAX_HISTORY_TURNS, f"应该剩10轮，实际剩{user_count}轮"
    # 裁剪后的第一条必须是user消息（轮次边界），不能是从某一轮中间切出来的assistant/tool残片
    assert trimmed[0]["role"] == "user"
    # 检查没有"孤儿tool消息"——每条tool消息前面必须紧跟着有对应tool_call_id的assistant消息
    tool_call_ids_seen = set()
    for m in trimmed:
        if m["role"] == "assistant" and m.get("tool_calls"):
            tool_call_ids_seen.update(tc["id"] for tc in m["tool_calls"])
        if m["role"] == "tool":
            assert m["tool_call_id"] in tool_call_ids_seen, "发现孤儿tool消息，说明裁剪切断了一次工具调用"
    print(f"测试通过：15轮历史裁剪到{ChatAgent.MAX_HISTORY_TURNS}轮，且没有切断任何一次工具调用")


def test_trim_history_noop_when_under_limit():
    """历史轮数没超过上限时，不应该做任何裁剪"""
    history = [{"role": "user", "content": "问题1"}, {"role": "assistant", "content": "回答1"}]
    agent = ChatAgent(llm=FakeLLM([]))
    trimmed = agent._trim_history(history)
    assert trimmed == history
    print("测试通过：历史轮数没超限时，不会做多余的裁剪")


def test_max_rounds_fallback_no_infinite_loop():
    """模型一直要求调用工具、从不给最终答案时，应该在MAX_TOOL_ROUNDS轮后兜底返回，不会死循环"""
    always_tool_call = _assistant_msg(
        content=None, tool_calls=[_tool_call("call_x", "get_weather_forecast", {"city": "长沙"})]
    )
    fake_llm = FakeLLM([always_tool_call] * ChatAgent.MAX_TOOL_ROUNDS)

    with patch.object(ChatAgent, "_execute_tool", return_value={"weather_data": "晴"}):
        agent = ChatAgent(llm=fake_llm)
        answer, history = agent.run("测试死循环兜底")

    assert fake_llm.call_count == ChatAgent.MAX_TOOL_ROUNDS
    assert "抱歉" in answer
    print(f"测试通过：连续{ChatAgent.MAX_TOOL_ROUNDS}轮都要求调用工具时，正确兜底返回，没有无限循环")


if __name__ == "__main__":
    test_single_tool_call_then_final_answer()
    test_no_tool_call_needed()
    test_memory_carries_across_two_runs()
    test_llm_call_fails_returns_graceful_message_not_crash()
    test_image_description_merged_into_user_message()
    test_image_description_failure_degrades_gracefully()
    test_no_image_skips_vision_call()
    test_trim_history_keeps_whole_turns_only()
    test_trim_history_noop_when_under_limit()
    test_max_rounds_fallback_no_infinite_loop()
    print("\n全部测试通过")
