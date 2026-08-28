import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.core.subagent_task import run_local_subagent


def test_local_subagent_success():
    task = run_local_subagent("测试Agent", lambda: "执行成功", "测试输入")

    assert task.agent_name == "测试Agent"
    assert task.status == "completed"
    assert task.succeeded
    assert task.result_text == "执行成功"
    assert task.protocol == "local"


def test_local_subagent_failure_is_recorded():
    def fail():
        raise RuntimeError("模拟失败")

    task = run_local_subagent("测试Agent", fail, "测试输入")

    assert task.status == "failed"
    assert not task.succeeded
    assert "模拟失败" in task.error
    assert task.result_text == ""
