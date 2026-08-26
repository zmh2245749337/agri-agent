"""LangChain消息与项目现有HTTP聊天历史格式之间的适配。"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def serialize_messages(messages: list) -> list[dict]:
    """转换为前端认识的role/content/tool_calls结构，并省略system消息。"""
    history = []

    for message in messages:
        if isinstance(message, SystemMessage):
            continue

        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
            continue

        if isinstance(message, AIMessage):
            item = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tool_call.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": (
                                tool_call["args"]
                                if isinstance(tool_call.get("args"), str)
                                else json.dumps(tool_call.get("args", {}), ensure_ascii=False)
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            history.append(item)
            continue

        if isinstance(message, ToolMessage):
            history.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            )

    return history


def deserialize_messages(history: list[dict] | None) -> list:
    """把前端历史恢复为LangChain消息，用于内存checkpointer重启后的会话续接。"""
    messages = []

    for item in history or []:
        role = item.get("role")
        content = item.get("content") or ""

        if role == "user":
            messages.append(HumanMessage(content=content))
            continue

        if role == "assistant":
            tool_calls = []
            for tool_call in item.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "args": arguments,
                    }
                )
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
            continue

        if role == "tool" and item.get("tool_call_id"):
            messages.append(
                ToolMessage(content=content, tool_call_id=item["tool_call_id"])
            )

    return messages
