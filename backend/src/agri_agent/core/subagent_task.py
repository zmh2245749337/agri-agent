"""PlanningAgent 内部统一使用的子任务结果。

这个类型不冒充任何外部协议：进程内调用和 A2A 远程调用最终都会转换成
同一份结果，方便 PlanningAgent 做失败隔离和结果拼装。
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4


@dataclass
class SubAgentTask:
    agent_name: str
    input_text: str
    protocol: str = "local"
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "submitted"
    result_text: str = ""
    # 子Agent生成自然语言时可能省略城市名、政策链接等原始字段。保留结构化
    # 证据，既用于调试/评测，也让上层能够追溯回答依据。
    evidence: Any = None
    error: str | None = None
    fallback_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


def run_local_subagent(
    agent_name: str,
    run_fn: Callable[[], str],
    input_text: str,
) -> SubAgentTask:
    """执行一个进程内子 Agent，并把成功/失败转换为统一任务结果。"""
    task = SubAgentTask(agent_name=agent_name, input_text=input_text)
    task.status = "working"
    try:
        task.result_text = run_fn()
        task.status = "completed"
    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        print(f"[local] Task[{task.task_id}] {agent_name} 执行失败：{exc}")
    return task
