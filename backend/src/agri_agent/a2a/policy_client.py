"""PlanningAgent 使用的官方 A2A 政策 Agent 客户端。"""

import asyncio

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_data_message
from a2a.types import Role, SendMessageRequest, TaskState

from agri_agent.core.subagent_task import SubAgentTask


class PolicyA2AError(RuntimeError):
    """远程政策 Agent 无法完成任务。"""


class PolicyA2AClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def arun(self, crop: str, region: str, need: str) -> SubAgentTask:
        """发现 Agent Card，通过 JSON-RPC 发送消息并读取最终 Task。"""
        http_client = httpx.AsyncClient(
            transport=self.transport,
            timeout=self.timeout_seconds,
        )
        config = ClientConfig(
            streaming=False,
            httpx_client=http_client,
            supported_protocol_bindings=["JSONRPC"],
            accepted_output_modes=["text/plain"],
        )
        client = await create_client(self.base_url, client_config=config)
        try:
            message = new_data_message(
                {"crop": crop, "region": region, "need": need},
                media_type="application/json",
                role=Role.ROLE_USER,
            )
            request = SendMessageRequest(message=message)
            final_task = None
            direct_message = ""
            async for response in client.send_message(request):
                if response.HasField("task"):
                    final_task = response.task
                elif response.HasField("message"):
                    direct_message = get_message_text(response.message)

            if final_task is None:
                if direct_message:
                    return SubAgentTask(
                        agent_name="AgriAgent Policy Service",
                        input_text=f"作物={crop}；地区={region}；需求={need}",
                        protocol="A2A/JSONRPC",
                        status="completed",
                        result_text=direct_message,
                    )
                raise PolicyA2AError("A2A 响应中没有 Message 或 Task")

            status = final_task.status.state
            if status != TaskState.TASK_STATE_COMPLETED:
                error = "远程任务未完成"
                if final_task.status.HasField("message"):
                    error = get_message_text(final_task.status.message) or error
                raise PolicyA2AError(
                    f"A2A Task[{final_task.id}] 状态={TaskState.Name(status)}：{error}"
                )

            result_text = "\n".join(
                text
                for text in (get_artifact_text(item) for item in final_task.artifacts)
                if text
            )
            if not result_text:
                raise PolicyA2AError(f"A2A Task[{final_task.id}] 已完成但没有 Artifact")

            return SubAgentTask(
                agent_name="AgriAgent Policy Service",
                input_text=f"作物={crop}；地区={region}；需求={need}",
                protocol="A2A/JSONRPC",
                task_id=final_task.id,
                status="completed",
                result_text=result_text,
            )
        finally:
            await client.close()

    def run(self, crop: str, region: str, need: str) -> SubAgentTask:
        """供 PlanningAgent 的线程池同步调用。"""
        return asyncio.run(self.arun(crop, region, need))
