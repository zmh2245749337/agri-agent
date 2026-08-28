import sys
from pathlib import Path

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.a2a.policy_client import PolicyA2AClient, PolicyA2AError
from agri_agent.a2a.policy_server import create_policy_a2a_app


class FakePolicyAgent:
    def run(self, crop, region, need):
        return f"{region}{crop}{need}：可查询当地农业主管部门。"


class FailingPolicyAgent:
    def run(self, crop, region, need):
        raise RuntimeError("模拟政策服务内部失败")


def test_official_a2a_card_message_task_and_artifact_flow():
    app = create_policy_a2a_app(
        public_url="http://policy.test",
        policy_agent=FakePolicyAgent(),
    )
    client = PolicyA2AClient(
        "http://policy.test",
        transport=httpx.ASGITransport(app=app),
    )

    task = client.run("水稻", "湖南省", "种植补贴")

    assert task.status == "completed"
    assert task.protocol == "A2A/JSONRPC"
    assert task.task_id
    assert "湖南省水稻种植补贴" in task.result_text


def test_remote_failed_task_becomes_client_error():
    app = create_policy_a2a_app(
        public_url="http://policy.test",
        policy_agent=FailingPolicyAgent(),
    )
    client = PolicyA2AClient(
        "http://policy.test",
        transport=httpx.ASGITransport(app=app),
    )

    with pytest.raises(PolicyA2AError, match="失败"):
        client.run("水稻", "湖南省", "种植补贴")
