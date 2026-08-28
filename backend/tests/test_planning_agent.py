# tests/test_planning_agent.py
"""
验证PlanningAgent的编排逻辑：三个子Agent各自可能成功或失败，PlanningAgent
应该能优雅地处理"部分失败"，而不是一个子Agent出错就导致整个流程崩溃。

用mock把三个子Agent和最终整合用的self.agent都换成假对象，不依赖真实网络、
真实MCP工具、真实大模型——只验证"编排逻辑"本身对不对，跟三个子Agent各自
内部的正确性是两件独立的事（那部分分别由diagnosis_agent/weather_agent/
policy_agent各自的测试和手动验证负责）。

政策子Agent支持通过官方A2A SDK独立服务化；这里验证PlanningAgent的选择逻辑：
配置远程客户端时走A2A，远程失败时回退本地，其余两个子Agent的并发和降级语义不变。
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.agents.planning_agent import PlanningAgent
from agri_agent.core.subagent_task import SubAgentTask


def _make_agent_with_fakes(diagnosis_run, weather_run, policy_run, final_run):
    """构造一个跳过真实__init__的PlanningAgent，四个用到大模型/子Agent的地方
    全部换成假实现，方便精确控制每一路的成功/失败"""
    with patch.object(PlanningAgent, "__init__", lambda self: None):
        agent = PlanningAgent()
    agent.diagnosis_agent = type("Fake", (), {"run": staticmethod(diagnosis_run)})()
    agent.weather_agent = type("Fake", (), {"run": staticmethod(weather_run)})()
    agent.policy_agent = type("Fake", (), {"run": staticmethod(policy_run)})()
    agent.agent = type("Fake", (), {"run": staticmethod(final_run)})()
    agent.policy_a2a_client = None
    agent.last_tasks = {}
    return agent


def test_all_succeed_combines_three_sections():
    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议内容",
        weather_run=lambda city: "天气建议内容",
        policy_run=lambda crop, region, need: "政策建议内容",
        final_run=lambda prompt: prompt,  # 直接把prompt原样返回，方便断言拼接是否正确
    )

    result = agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    assert "诊断建议内容" in result
    assert "天气建议内容" in result
    assert "政策建议内容" in result
    print("测试通过：三个子Agent都成功时，正确整合成一份prompt交给大模型")


def test_skips_diagnosis_when_no_symptom_text():
    calls = []
    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: calls.append("diagnosis") or "不应该被调用",
        weather_run=lambda city: "天气建议内容",
        policy_run=lambda crop, region, need: "政策建议内容",
        final_run=lambda prompt: prompt,
    )

    result = agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text=None)

    assert "diagnosis" not in calls  # 没传症状描述，诊断Agent压根不该被调用
    assert "天气建议内容" in result
    assert "政策建议内容" in result
    print("测试通过：不传symptom_text时，正确跳过诊断Agent，不会硬调用")


def test_one_agent_fails_others_still_combine():
    def failing_weather(city):
        raise RuntimeError("模拟网络超时")

    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议内容",
        weather_run=failing_weather,
        policy_run=lambda crop, region, need: "政策建议内容",
        final_run=lambda prompt: prompt,
    )

    result = agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    # 天气模块失败了，但诊断和政策这两块应该还在，不能因为一个模块失败就全盘皆输
    assert "诊断建议内容" in result
    assert "政策建议内容" in result
    print("测试通过：某一个子Agent失败时，其余部分仍能正常整合，不会整体崩溃")


def test_all_agents_fail_returns_apology_without_calling_llm():
    final_run_called = []

    def failing(*args, **kwargs):
        raise RuntimeError("模拟失败")

    agent = _make_agent_with_fakes(
        diagnosis_run=failing,
        weather_run=failing,
        policy_run=failing,
        final_run=lambda prompt: final_run_called.append(1) or prompt,
    )

    result = agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    assert "抱歉" in result
    assert not final_run_called  # 三个都失败时，不该再多此一举去调大模型整合空内容
    print("测试通过：三个子Agent全部失败时，直接返回提示，不会拿空内容硬调大模型")


def test_three_subagents_run_concurrently_not_sequentially():
    """三个子Agent改成线程池并发之后，最重要的一条回归测试：三段耗时应该约等于
    "最慢的那一段"，而不是"三段耗时相加"。用三个各自sleep 0.3秒的假实现模拟
    三次真实的网络I/O等待，如果还是顺序执行，总耗时会接近0.9秒；如果并发生效，
    总耗时应该接近0.3秒（留了不少余量给线程调度开销，避免测试环境波动导致误报）"""
    def _slow(seconds):
        def _inner(*args, **kwargs):
            time.sleep(seconds)
            return f"耗时{seconds}秒的结果"
        return _inner

    agent = _make_agent_with_fakes(
        diagnosis_run=_slow(0.3),
        weather_run=_slow(0.3),
        policy_run=_slow(0.3),
        final_run=lambda prompt: prompt,
    )

    start = time.time()
    agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")
    elapsed = time.time() - start

    # 顺序执行至少要0.9秒，这里用0.6秒做判定线——留了充足余量，
    # 只要真的走了并发，不可能卡在这条线以上
    assert elapsed < 0.6, f"耗时{elapsed:.2f}秒，看起来还是顺序执行的，并发没有生效"
    print(f"测试通过：三个子Agent确实是并发执行的，总耗时{elapsed:.2f}秒（而不是顺序执行的~0.9秒）")


def test_section_order_stays_fixed_regardless_of_completion_order():
    """哪个子Agent先跑完是不确定的（取决于线程调度），但用户看到的报告结构应该
    始终是"诊断→天气→政策"这个固定顺序，不能因为并发就变得随机——特意让天气
    最先完成、诊断最后完成，验证最终结果里的顺序没有被打乱"""
    def _slow(seconds, text):
        def _inner(*args, **kwargs):
            time.sleep(seconds)
            return text
        return _inner

    agent = _make_agent_with_fakes(
        diagnosis_run=_slow(0.15, "诊断内容"),  # 最慢，最后完成
        weather_run=_slow(0.01, "天气内容"),     # 最快，最先完成
        policy_run=_slow(0.08, "政策内容"),      # 中间完成
        final_run=lambda prompt: prompt,
    )

    result = agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    # 不管完成顺序如何，拼进最终prompt里的顺序必须是诊断在前、天气居中、政策在后
    assert result.index("诊断内容") < result.index("天气内容") < result.index("政策内容")
    print("测试通过：不管三个子Agent实际完成顺序如何，最终报告结构始终保持诊断→天气→政策的固定顺序")


def test_last_tasks_records_task_status_for_each_subagent():
    """run()结束应该把三个子Agent各自的统一Task对象记录在
    self.last_tasks里，状态机(completed/failed)要如实反映每个子Agent的
    真实执行结果——这是协议化改造之后新增的可观测性，专门验证它真的生效了，
    不是只加了个没人读的字段"""
    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议内容",
        weather_run=lambda city: "天气建议内容",
        policy_run=lambda crop, region, need: "政策建议内容",
        final_run=lambda prompt: prompt,
    )
    agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    assert set(agent.last_tasks.keys()) == {"diagnosis", "weather", "policy"}
    for task in agent.last_tasks.values():
        assert task.status == "completed"
        assert task.succeeded
        assert task.error is None
    assert agent.last_tasks["diagnosis"].result_text == "诊断建议内容"
    print("测试通过：run()结束后，三个子Agent的Task状态都正确记录为completed")


def test_last_tasks_marks_failed_subagent_with_error_message():
    def failing_weather(city):
        raise RuntimeError("模拟网络超时")

    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议内容",
        weather_run=failing_weather,
        policy_run=lambda crop, region, need: "政策建议内容",
        final_run=lambda prompt: prompt,
    )
    agent.run(crop="水稻", city="长沙", region="湖南省", symptom_text="叶子发黄")

    weather_task = agent.last_tasks["weather"]
    assert weather_task.status == "failed"
    assert not weather_task.succeeded
    assert "模拟网络超时" in weather_task.error
    assert weather_task.result_text == ""
    # 失败的子Agent不该影响其它子Agent对应Task的状态
    assert agent.last_tasks["diagnosis"].status == "completed"
    print("测试通过：某个子Agent失败时，对应Task状态记录为failed且带错误信息，不影响其它Task")


def test_explicit_capabilities_only_run_requested_subagents():
    calls = []
    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: calls.append("diagnosis") or "诊断建议内容",
        weather_run=lambda city: calls.append("weather") or "天气建议内容",
        policy_run=lambda crop, region, need: calls.append("policy") or "不应该被调用",
        final_run=lambda prompt: prompt,
    )

    result = agent.run(
        crop="水稻",
        city="长沙",
        symptom_text="叶片发黄",
        capabilities=["diagnosis", "weather"],
    )

    assert set(calls) == {"diagnosis", "weather"}
    assert "policy" not in calls
    assert set(agent.last_tasks) == {"diagnosis", "weather"}
    assert "诊断建议内容" in result and "天气建议内容" in result
    assert "严禁创建空章节" in result
    assert '分成"近期要做的事"和"可以了解的政策/资源"两部分' not in result
    print("测试通过：显式capabilities只执行本轮需要的专业模块，不会无条件查询政策")


def test_policy_uses_remote_a2a_when_client_is_configured():
    class FakeA2AClient:
        def run(self, crop, region, need):
            return SubAgentTask(
                agent_name="远程政策Agent",
                input_text=f"{crop}/{region}/{need}",
                protocol="A2A/JSONRPC",
                status="completed",
                result_text="远程A2A政策结果",
            )

    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议",
        weather_run=lambda city: "天气建议",
        policy_run=lambda crop, region, need: "不应调用本地政策Agent",
        final_run=lambda prompt: prompt,
    )
    agent.policy_a2a_client = FakeA2AClient()

    result = agent.run("水稻", "长沙", "湖南省", "叶片发黄")

    assert "远程A2A政策结果" in result
    assert agent.last_tasks["policy"].protocol == "A2A/JSONRPC"


def test_policy_a2a_failure_falls_back_to_local_agent():
    class FailingA2AClient:
        def run(self, crop, region, need):
            raise RuntimeError("远程服务不可达")

    agent = _make_agent_with_fakes(
        diagnosis_run=lambda crop, symptom: "诊断建议",
        weather_run=lambda city: "天气建议",
        policy_run=lambda crop, region, need: "本地回退政策结果",
        final_run=lambda prompt: prompt,
    )
    agent.policy_a2a_client = FailingA2AClient()

    result = agent.run("水稻", "长沙", "湖南省", "叶片发黄")

    policy_task = agent.last_tasks["policy"]
    assert "本地回退政策结果" in result
    assert policy_task.protocol == "local-fallback"
    assert "远程服务不可达" in policy_task.fallback_reason


if __name__ == "__main__":
    test_all_succeed_combines_three_sections()
    test_skips_diagnosis_when_no_symptom_text()
    test_one_agent_fails_others_still_combine()
    test_all_agents_fail_returns_apology_without_calling_llm()
    test_three_subagents_run_concurrently_not_sequentially()
    test_section_order_stays_fixed_regardless_of_completion_order()
    test_last_tasks_records_task_status_for_each_subagent()
    test_last_tasks_marks_failed_subagent_with_error_message()
    test_explicit_capabilities_only_run_requested_subagents()
    print("\n全部测试通过")
