# tests/test_a2a_lite.py
"""
core/a2a_lite.py本身的单元测试——只测这套借鉴A2A协议理念做的轻量调度层
（AgentCard/Message/Task/dispatch_task）自己的行为对不对，不涉及任何真实
子Agent、不需要网络/大模型。跟test_planning_agent.py分工不同：那边测的是
"PlanningAgent怎么用这套协议"，这里测的是"协议本身的行为"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.core.a2a_lite import AgentCard, Message, dispatch_task


def test_message_of_text_builds_single_text_part():
    msg = Message.of_text("user", "作物=水稻；症状=叶子发黄")
    assert msg.role == "user"
    assert msg.parts == [{"type": "text", "text": "作物=水稻；症状=叶子发黄"}]
    assert msg.text == "作物=水稻；症状=叶子发黄"
    print("测试通过：Message.of_text正确构造单个text part，text属性能正确取回")


def test_message_text_property_returns_empty_when_no_text_part():
    # 故意构造一个不含text类型part的消息（模拟以后扩展了别的part类型的场景），
    # text属性应该优雅地返回空字符串，不应该抛异常
    msg = Message(role="agent", parts=[{"type": "data", "data": {"foo": "bar"}}])
    assert msg.text == ""
    print("测试通过：消息里没有text类型的part时，text属性返回空字符串而不是报错")


def test_dispatch_task_success_produces_completed_task():
    card = AgentCard(name="测试Agent", description="用于单元测试的假Agent")

    task = dispatch_task(card, run_fn=lambda: "这是Agent的输出", input_text="测试输入")

    assert task.agent_name == "测试Agent"
    assert task.status == "completed"
    assert task.succeeded is True
    assert task.error is None
    assert task.input_message.text == "测试输入"
    assert task.output_message.text == "这是Agent的输出"
    assert task.result_text == "这是Agent的输出"
    assert task.started_at is not None and task.finished_at is not None
    assert task.finished_at >= task.started_at
    # 每次dispatch_task都应该生成不同的task_id，不能复用/写死
    task2 = dispatch_task(card, run_fn=lambda: "另一次输出", input_text="测试输入")
    assert task.task_id != task2.task_id
    print("测试通过：run_fn成功时，Task状态正确记录为completed，各字段都填充正确，task_id不重复")


def test_dispatch_task_failure_produces_failed_task_with_error():
    card = AgentCard(name="测试Agent", description="用于单元测试的假Agent")

    def _raise():
        raise RuntimeError("模拟子Agent内部报错")

    task = dispatch_task(card, run_fn=_raise, input_text="测试输入")

    assert task.status == "failed"
    assert task.succeeded is False
    assert task.output_message is None
    assert "模拟子Agent内部报错" in task.error
    # 失败时result_text应该优雅地返回空字符串，方便调用方不用先判断succeeded再取值
    assert task.result_text == ""
    print("测试通过：run_fn抛异常时，Task状态正确记录为failed，error里带原始异常信息，result_text安全返回空字符串")


if __name__ == "__main__":
    test_message_of_text_builds_single_text_part()
    test_message_text_property_returns_empty_when_no_text_part()
    test_dispatch_task_success_produces_completed_task()
    test_dispatch_task_failure_produces_failed_task_with_error()
    print("\n全部测试通过")
