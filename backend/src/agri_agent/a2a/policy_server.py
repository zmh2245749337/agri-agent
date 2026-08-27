"""把政策补贴子 Agent 暴露为标准 A2A JSON-RPC 服务。

实现基于官方 ``a2a-sdk`` 1.x：发布 Agent Card，接收结构化 Message，
并通过 Task / Artifact / TaskStatusUpdateEvent 返回可追踪的任务生命周期。
"""

import asyncio
import os
from typing import Protocol

import uvicorn
from a2a.helpers import (
    get_data_parts,
    new_task_from_user_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette


class PolicyAgentLike(Protocol):
    def run(self, crop: str, region: str, need: str) -> str: ...


def build_policy_agent_card(public_url: str) -> AgentCard:
    """构建供其他 Agent 自动发现的标准 Agent Card。"""
    return AgentCard(
        name="AgriAgent Policy Service",
        description="检索农业补贴政策并生成带来源说明的申请建议。",
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
        ),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=public_url.rstrip("/"),
                protocol_version="1.0",
            )
        ],
        skills=[
            AgentSkill(
                id="agricultural-policy-advice",
                name="农业补贴政策查询",
                description="根据作物、地区和用户需求检索政策并生成申请建议。",
                tags=["agriculture", "policy", "rag"],
                examples=["查询湖南省水稻种植补贴"],
                input_modes=["application/json"],
                output_modes=["text/plain"],
            )
        ],
    )


def _parse_policy_payload(context: RequestContext) -> tuple[str, str, str]:
    if context.message is None:
        raise ValueError("A2A 请求缺少 Message")
    data_parts = get_data_parts(context.message.parts)
    if not data_parts or not isinstance(data_parts[0], dict):
        raise ValueError("A2A Message 必须包含 application/json Data Part")
    payload = data_parts[0]
    missing = [key for key in ("crop", "region", "need") if not payload.get(key)]
    if missing:
        raise ValueError(f"缺少必填字段：{', '.join(missing)}")
    return str(payload["crop"]), str(payload["region"]), str(payload["need"])


class PolicyAgentExecutor(AgentExecutor):
    """将同步 PolicySubsidyAgent 适配成官方 A2A Task 生命周期。"""

    def __init__(self, policy_agent: PolicyAgentLike):
        self._policy_agent = policy_agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ValueError("A2A 请求缺少 Message")

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        await updater.update_status(TaskState.TASK_STATE_WORKING)

        try:
            crop, region, need = _parse_policy_payload(context)
            result = await asyncio.to_thread(
                self._policy_agent.run,
                crop,
                region,
                need,
            )
            await updater.add_artifact(
                parts=[new_text_part(result, media_type="text/plain")],
                name="policy-advice",
            )
            await updater.update_status(TaskState.TASK_STATE_COMPLETED)
        except Exception as exc:
            error_message = updater.new_agent_message(
                [new_text_part(f"政策 Agent 执行失败：{exc}")]
            )
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=error_message,
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("当前政策查询任务不支持取消")


def create_policy_a2a_app(
    policy_agent: PolicyAgentLike | None = None,
    public_url: str = "http://127.0.0.1:8011",
) -> Starlette:
    """创建同时提供 Agent Card 与 A2A JSON-RPC 端点的应用。"""
    if policy_agent is None:
        # 延迟导入，测试 Agent Card/协议层时无需加载向量模型和真实 LLM。
        from agri_agent.agents.policy_agent import PolicySubsidyAgent

        policy_agent = PolicySubsidyAgent()

    agent_card = build_policy_agent_card(public_url)
    request_handler = DefaultRequestHandler(
        agent_executor=PolicyAgentExecutor(policy_agent),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, rpc_url="/"))
    return Starlette(routes=routes)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    host = os.getenv("POLICY_A2A_HOST", "127.0.0.1")
    port = int(os.getenv("POLICY_A2A_PORT", "8011"))
    public_url = os.getenv("POLICY_A2A_PUBLIC_URL", f"http://{host}:{port}")
    app = create_policy_a2a_app(public_url=public_url)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
