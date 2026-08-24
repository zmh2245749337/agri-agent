# src/agri_agent/agents/chat_agent_langgraph.py
"""
ChatAgent（chat_agent.py，手写Function Calling循环）的LangGraph版本——解决的是
完全同一个问题（reason-act-observe循环：模型自己判断要不要调用工具、调哪个，
直到给出最终答案为止），用业界最主流的Agent编排框架重新实现一遍，目的是能
拿两份实现对比着讲清楚"框架到底帮你做了什么、你自己还剩下什么要做"。

SYSTEM_PROMPT/IMAGE_DESCRIBE_PROMPT/VISION_MODEL这些业务话术直接从chat_agent.py
导入复用，不重新写一遍——这两个实现之间的差异应该只发生在"循环怎么跑起来"这一层，
不应该在业务逻辑上有任何出入，不然没法说清楚两边行为上的差异到底是"框架带来的"
还是"我自己两次实现写得不一样"导致的，对比就失去意义了。

跟ChatAgent逐项对应关系：
- TOOLS的JSON Schema手写 → @tool装饰器 + 类型注解 + docstring，LangChain自动
  推导出等价的Schema，少写了一份手工维护的parameters字典，代价是schema怎么生成
  的细节被框架隐藏了，不如手写的直观、可控。
- MyLLM.chat_with_tools()手动传tools=TOOLS → llm.bind_tools(TOOLS)，效果一样，
  只是"工具怎么序列化进请求"这层细节被框架包装掉了。
- `for _ in range(MAX_TOOL_ROUNDS): ...` 手写循环 + `if not assistant_message.tool_calls`
  判断 → StateGraph的agent/tools两个节点 + tools_condition条件边，框架把这个循环
  和判断标准化成了一个可复用的图结构（本质上就是LangGraph自带的create_react_agent
  预置模板做的事，这里手动搭一遍graph是为了把每一步拆开看清楚，不用一行代码糊过去）。
- ChatAgent._trim_history() + 调用方手动存/传history列表 → MemorySaver checkpointer +
  thread_id，记忆由框架接管，调用方不用自己维护消息列表了——但下面会看到，这不代表
  "历史裁剪"这件事框架也帮你做了，这是两回事（见run()方法里的说明）。

这不是要替换掉ChatAgent——两份实现都保留，服务的目的不一样：ChatAgent是这个项目
真正在跑的服务，用自建框架证明理解Function Calling的底层机制；这份LangGraph版本
是一份对比学习/面试谈资用的验证性实现，不接入api/main.py，不影响现有服务。

依赖单独放在requirements-langgraph.txt里，不merge进主requirements.txt——这是
刻意的取舍：LangGraph生态的依赖链比较重，只为了一份对比学习用的旁支实现，
不应该让主服务多背上这些依赖，想跑这份代码的人单独装一下就行。
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agri_agent.agents.chat_agent import IMAGE_DESCRIBE_PROMPT, SYSTEM_PROMPT, VISION_MODEL
from agri_agent.agents.weather_agent import query_weather
from agri_agent.core.my_llm import MyLLM
from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge
from agri_agent.tools.policy_match_tool import match_policy
from agri_agent.tools.web_search_tool import web_search


# 图的状态定义。LangGraph要求State是一个TypedDict，每个字段可以指定一个"reducer"
# 函数，决定"节点返回的更新要怎么合并进现有state"——messages字段用官方提供的
# add_messages这个reducer：默认是"追加"，但如果新消息的id跟已有消息的id相同，
# 会变成"替换"而不是"追加"（下面SystemMessage固定id就是利用了这个机制）。
# 这个reducer机制本身就是ChatAgent里没有的东西——手写版本里"新消息怎么拼进
# messages列表"是每处手动append，这里是声明式的，state该怎么合并是graph自己管的
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 固定id的SystemMessage：每次run()都重新传这一条，因为id相同，add_messages
# reducer会原地替换而不是不断往消息列表里堆重复的system消息——这是LangGraph
# 惯用的"每轮都声明一次系统提示词，但不担心它被重复叠加"写法，替代了ChatAgent
# 里"只在拼messages的时候加一次system"这种手动控制
_SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT, id="system-prompt")


# ---------- 四个工具：用@tool装饰器包一层，函数体直接复用原始工具函数 ----------
# 类型注解（crop: str）加docstring，LangChain会自动推导出等价于ChatAgent里TOOLS
# 那份手写JSON Schema的效果——参数名、类型、以及docstring本身（当成description）
# 都会被序列化进发给大模型的请求里，不需要再手写一份parameters字典
@tool
def diagnose_crop_disease(crop: str, symptom_text: str) -> dict:
    """查询本地农作物病虫害知识库，根据作物名称和症状描述返回可能病因和处理建议。
    适用于用户描述了具体症状（比如叶子发黄、有虫斑、卷叶等）的情况。"""
    return {"matched_knowledge": match_pest_knowledge(crop, symptom_text)}


@tool
def get_weather_forecast(city: str) -> dict:
    """查询指定城市的实时天气预报，用于判断近期是否适合打药、灌溉、收割等农事操作。"""
    return {"weather_data": asyncio.run(query_weather(city))}


@tool
def search_subsidy_policy(crop: str, region: str, need: str) -> dict:
    """在本地政策知识库中检索农业补贴政策（关键词粗筛+向量语义精排两阶段检索）。
    region必须是省级行政区（比如"湖南省"而不是"长沙"）。
    如果返回的matched_policies是空列表，说明本地库没有覆盖，应该接着调用
    web_search_policy联网搜索，不要就此放弃。"""
    return {"matched_policies": match_policy(crop, region, need)}


@tool
def web_search_policy(query: str) -> dict:
    """联网搜索最新的农业补贴政策信息。只应该在search_subsidy_policy本地检索不到
    结果时调用，不要跳过本地检索直接联网搜。"""
    return {"search_results": web_search(query)}


TOOLS = [diagnose_crop_disease, get_weather_forecast, search_subsidy_policy, web_search_policy]


class ChatAgentLangGraph:
    """跟ChatAgent对应的LangGraph实现，详细的逐项对应关系见文件顶部注释。"""

    def __init__(self, llm=None, vision_model=None):
        # llm参数允许调用方（主要是测试）注入一个假的、只要支持bind_tools()/invoke()
        # 接口的对象，不传就用配置好的真实ChatOpenAI——跟ChatAgent/MyLLM的llm=None
        # 依赖注入写法是同一个习惯
        llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL_ID"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0.7,
            timeout=60,
        )
        # bind_tools()对应ChatAgent里chat_with_tools(messages, tools=TOOLS)手动传
        # tools参数的那一步——绑完之后返回的对象，每次invoke()都会自动带上工具信息
        self.llm_with_tools = llm.bind_tools(TOOLS)

        # 看图这一步（视觉模型描述症状）跟"用什么框架编排Function Calling循环"
        # 完全无关，是一次确定性的预处理调用，继续用MyLLM就行，没必要为了这一步
        # 也套一层LangChain——这也说明手写组件和框架组件在同一个服务里是可以
        # 共存的，不是非此即彼
        self.vision_llm = MyLLM(model=vision_model or VISION_MODEL)

        graph = self._build_graph()
        # MemorySaver：LangGraph内置的检查点存储（这里用的是纯内存版，进程重启就丢，
        # 生产场景可以换成SqliteSaver/PostgresSaver这些持久化实现，接口不变）。
        # 每个thread_id对应一条独立的对话线程，compile时把checkpointer接进去，
        # 之后invoke()只要传相同的thread_id，历史消息会自动被接上——这一步是
        # ChatAgent._trim_history()+调用方手动传history列表这套机制的替代品
        self.checkpointer = MemorySaver()
        self.compiled_graph = graph.compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        # ToolNode是LangGraph自带的预置节点：读最后一条AIMessage里的tool_calls，
        # 依次找到对应工具、执行、把结果包装成ToolMessage加回state——对应
        # ChatAgent.run()里那段"for tool_call in assistant_message.tool_calls: ..."
        # 手写循环，效果完全等价，只是不用自己写了
        graph.add_node("tools", ToolNode(TOOLS))

        graph.add_edge(START, "agent")
        # tools_condition是LangGraph自带的判断函数：检查最后一条消息有没有
        # tool_calls，有就路由到"tools"节点，没有就路由到END——对应ChatAgent.run()
        # 里`if not assistant_message.tool_calls: ... return`这一步判断，
        # 框架把这个if判断标准化成了一个可复用的条件边
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        # tools节点执行完，边指回agent节点，形成"问模型→（可能）调工具→再问模型"
        # 的循环，直到tools_condition判断不再需要调工具为止——对应ChatAgent.run()
        # 里for循环"执行完这一轮工具，回到循环开头再问一次模型"这个动作。
        # 注意这里没有像ChatAgent.MAX_TOOL_ROUNDS那样设一个显式的轮数上限——
        # 这是个已知的简化，真要用在生产场景，应该在_agent_node里自己数一下
        # 轮数、超过上限就强制返回，LangGraph不会替你兜这个底
        graph.add_edge("tools", "agent")
        return graph

    def _agent_node(self, state: AgentState) -> dict:
        """对应ChatAgent.run()循环体里`self.llm.chat_with_tools(messages, tools=TOOLS)`
        这一步调用——每次进这个节点，都拿当前完整的state["messages"]问一次模型，
        模型的回复（可能带tool_calls，也可能是最终答案）作为这个节点的输出，
        通过add_messages reducer追加进state"""
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _describe_image(self, image_data_url: str) -> str:
        """跟ChatAgent._describe_image()完全相同的逻辑，因为看图这一步不涉及
        LangGraph/Function Calling循环，没有必要在这个版本里重新设计"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }]
        return self.vision_llm.invoke(messages)

    def run(self, user_message: str, thread_id: str = "default", image_data_url: str = None) -> str:
        """
        跟ChatAgent.run()接口上最大的差异在这里：ChatAgent靠调用方显式传入/存储
        history列表维持记忆，这里靠thread_id——同一个thread_id多次调用，
        checkpointer会自动把上一次的state接上，调用方不需要自己攒消息列表了。

        这不是"更好"，只是记忆的管理责任从调用方转移到了框架内部：ChatAgent那种
        显式history更适合"需要把对话记录存进自己的数据库做审计/展示"这类场景
        （前端/API直接能拿到消息列表，可以随意处理）；这里的隐式记忆更省代码，
        但对话历史被框架的checkpointer管起来了，想拿出来做别的事得调用
        `self.compiled_graph.get_state(config).values["messages"]`这类API，
        多了一层间接。

        另外要老实说清楚一个容易被误解的点：MemorySaver只是负责"存和续接"，
        不负责"裁剪"——ChatAgent.MAX_HISTORY_TURNS那种"只保留最近10轮"的裁剪
        逻辑，这里没有实现，长对话下state会一直增长下去。这是刻意先不做的
        简化（真要做，需要在_agent_node或者一个专门的pre-model-hook节点里，
        参照_trim_history()的思路自己写裁剪，LangGraph不会替你自动做这件事），
        不是"框架帮我们把这个问题也解决了"，这个边界必须讲清楚，不然会被面试官
        问住。
        """
        if image_data_url:
            try:
                image_description = self._describe_image(image_data_url)
                user_message = f"{user_message}\n\n[图片识别到的症状描述：{image_description}]"
            except Exception as e:
                user_message = (
                    f"{user_message}\n\n"
                    f"[用户上传了一张图片，但图片识别失败（{e}），请基于文字部分回答，"
                    f"并告知用户图片处理暂时失败，建议改用文字描述症状]"
                )

        config = {"configurable": {"thread_id": thread_id}}
        result = self.compiled_graph.invoke(
            {"messages": [_SYSTEM_MESSAGE, HumanMessage(content=user_message)]},
            config=config,
        )
        return result["messages"][-1].content


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = ChatAgentLangGraph()

    # 用同一个thread_id跑两轮对话，第二轮故意不再提"水稻"和"长沙"，
    # 用来验证MemorySaver确实记住了上一轮的上下文——跟chat_agent.py __main__
    # 里那段"两轮对话测记忆"的demo是同一个验证目的，方便直接对比两边输出
    print("=== 第一轮：问天气 ===")
    answer1 = agent.run("我在长沙种水稻，最近适合打药吗？", thread_id="demo")
    print(answer1)

    print("\n=== 第二轮：带着记忆接着问（注意这句没有再提“水稻”和“长沙”） ===")
    answer2 = agent.run("那我这边有没有种植补贴可以申请，我在湖南省", thread_id="demo")
    print(answer2)
