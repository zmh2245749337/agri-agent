"""AgriAgent统一主图：意图路由、ReAct工具调用、Planning节点和多轮槽位状态。"""
import os
import time
from datetime import date
from typing import Annotated, Literal, Optional, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator

from agri_agent.agents.planning_agent import PlanningAgent
from agri_agent.agents.prompts import IMAGE_DESCRIBE_PROMPT, SYSTEM_PROMPT, VISION_MODEL
from agri_agent.core.message_history import deserialize_messages, serialize_messages
from agri_agent.core.my_llm import MyLLM
from agri_agent.tools.agent_tools import TOOLS


Capability = Literal["diagnosis", "weather", "policy"]
Route = Literal["direct", "react", "planning", "clarify"]


class RouteDecision(BaseModel):
    """路由模型只能输出这些字段，最终路由仍由程序校验。"""

    route: Route
    capabilities: list[Capability] = Field(default_factory=list)
    crop: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    symptom: Optional[str] = None
    need: Optional[str] = None
    growth_stage: Optional[str] = None
    has_symptom: Optional[bool] = None
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("route", mode="before")
    @classmethod
    def normalize_model_route(cls, value):
        """兼容部分 OpenAI-Compatible 模型把能力名填入 route 的情况。"""
        aliases = {
            "diagnosis": "react",
            "weather": "react",
            "policy": "react",
            "tool": "react",
            "function_calling": "react",
            "plan": "planning",
            "multi_agent": "planning",
            "answer": "direct",
            "chat": "direct",
            "ask_user": "clarify",
            "clarification": "clarify",
        }
        normalized = str(value).strip().lower() if value is not None else value
        return aliases.get(normalized, normalized)


class AgriState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    mode: str
    route: str
    capabilities: list[str]
    confidence: float
    missing_fields: list[str]
    crop: Optional[str]
    city: Optional[str]
    region: Optional[str]
    symptom: Optional[str]
    need: Optional[str]
    growth_stage: Optional[str]
    has_symptom: Optional[bool]
    active_goal: Optional[str]
    pending_slot: Optional[str]
    tool_rounds: int
    task_status: dict


ROUTER_PROMPT = """你是AgriAgent的请求路由器，只负责分类和提取字段，不回答用户问题。
请返回RouteDecision结构化结果，不要输出额外说明。

route判断规则：
- direct：寒暄、项目能力介绍、一般农艺知识，或对上一条回答的追问、解释和质疑。
- react：单一农业能力查询，由工具调用循环动态完成。
- planning：用户明确要求计划、方案、安排，或者同时需要两个及以上农业能力。
- clarify：用户意图本身无法判断；字段是否齐全最终会由程序再次校验。

capabilities只能选择：
- diagnosis：作物症状、病虫害、叶片异常、长势异常。
- weather：实时天气、预报、是否适合打药/灌溉/收割。
- policy：补贴、政策、申请条件、扶持项目。

“现在是否适合播种/种植”通常是农时和品种问题，不等于天气查询；除非用户明确提到天气、
降雨或气温，否则按一般农艺问题处理。city只能提取城市级地名，不能把江苏、湖南等省级地区
填进city；省级地名只放region。遇到“为什么没有建议”“刚才是什么意思”“那具体怎么做”
等承接上文的问题，必须结合最近对话判断，不要因为单看这一句话不完整就选择clarify。

同时提取crop、city、region、symptom、need、growth_stage和has_symptom。growth_stage使用简洁中文，
例如“未开始种植”“育秧”“移栽”“分蘖”“拔节孕穗”“抽穗”“灌浆”“成熟收获”。用户明确说
尚未种植或没有症状时，has_symptom必须为false，不能选择diagnosis，也不能要求补充症状。
助手上一轮提到“病虫害”不代表用户当前存在症状，能力必须以用户当前需求为准。
可以在高度确定时把常见城市补全到省级地区，
但不要猜测病害名称、政策名称或用户没有表达的症状。多轮对话要结合下面给出的已知字段。
confidence表示对路由和字段提取的整体把握程度。"""

DIRECT_PROMPT = """你是“智慧农事助手”。当前节点负责不调用外部工具的连续对话。
- 结合最近对话和已确认的作物、地区等信息，直接回应用户最后一句，不要让用户重复描述。
- 对一般农艺问题给出有用但谨慎的判断；缺少会显著影响结论的信息时，先给可执行的通用建议，再用一句话询问关键细节。
- 涉及“现在是否适合种植”或“最近必须做什么”时，先区分尚未播种、育秧、移栽和田间生长期；不得把这些阶段互斥的操作同时说成必做事项。阶段不明时，明确标出哪些建议是通用的、哪些必须确认阶段后才能决定。
- 判断农时时，必须把今天的日期和自己引用的常规播种、移栽窗口进行比较；如果当前日期已经超出该窗口，不能同时声称“可能正进入适播期”，例外情况必须说明还需要确认品种、茬口或栽培方式。
- 对“为什么没有建议”“还有什么建议”“那具体怎么做”等追问，先核对上一条回答再解释，并立即结合已确认的目标和阶段补充3到5条具体、可执行的建议；不得只解释原因后让用户重新选择想了解哪一部分。若上一条其实已有建议，可简短说明，但仍要补充更具体的行动项，不能盲目认错或大段重复原答案。
- 已确认种植阶段后不得重复询问。如果阶段会显著影响建议且尚未确认，回答末尾只询问这一个最高价值信息，不要顺带推销用户没有问过的天气或补贴功能。
- 当前节点没有工具。不得输出工具名、函数名、JSON参数或假装已经查询实时数据。
- 不确定实时天气、最新政策或具体地块情况时要明确说明，不得编造查询结果和日期。"""


_SLOT_LABELS = {
    "crop": "种植的作物",
    "city": "所在城市",
    "region": "省级行政区",
    "symptom": "具体症状",
    "need": "政策需求",
    "growth_stage": "种植阶段",
    "request": "想解决的具体问题",
}

_REQUIRED_FIELDS = {
    "diagnosis": ("crop", "symptom"),
    "weather": ("city",),
    "policy": ("crop", "region"),
}

_CAPABILITY_TOOL_NAMES = {
    "diagnosis": "diagnose_crop_disease",
    "weather": "get_weather_forecast",
    "policy": "search_subsidy_policy",
}

_EXPLICIT_CAPABILITY_KEYWORDS = {
    "diagnosis": ("症状", "病害", "虫害", "发黄", "斑点", "卷叶", "枯萎"),
    "weather": ("天气", "气温", "下雨", "降雨", "打药", "灌溉", "收割"),
    "policy": ("政策", "补贴", "扶持", "申请条件"),
}

# 只覆盖项目演示中反复使用、归属没有歧义的城市；其他地点仍由结构化路由提取。
_HIGH_CONFIDENCE_CITY_REGIONS = {
    "南京": "江苏省",
    "扬州": "江苏省",
    "长沙": "湖南省",
}

_PROVINCE_ONLY_NAMES = {
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建",
    "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州",
    "云南", "陕西", "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
}

_GROWTH_STAGE_PATTERNS = (
    ("未开始种植", ("还没有开始种植", "还没开始种植", "尚未开始种植", "还没有种植", "还没种植", "还没种", "准备种植")),
    ("育秧", ("育秧", "秧苗期")),
    ("移栽", ("移栽", "插秧")),
    ("分蘖", ("分蘖",)),
    ("拔节孕穗", ("拔节", "孕穗")),
    ("抽穗", ("抽穗", "扬花")),
    ("灌浆", ("灌浆",)),
    ("成熟收获", ("成熟", "准备收获", "收获期")),
)

_NO_SYMPTOM_PHRASES = (
    "没有症状", "没症状", "还没有症状", "未出现症状", "没发现异常", "没有病虫害",
)

_VAGUE_REQUEST_PATTERNS = (
    "帮我看看", "这个怎么办", "有点问题", "这情况正常吗", "帮我分析",
    "问题想请教", "感觉不太对", "帮我处理", "拿不准", "有点麻烦",
    "需要怎么弄", "出了点状况", "不知道下一步", "帮忙判断",
    "说不清楚", "给点建议",
)


class AgriGraphAgent:
    """面向API的统一Agent，所有用户入口最终进入这张图。"""

    MAX_TOOL_ROUNDS = 5
    MIN_ROUTE_CONFIDENCE = 0.65
    ROUTER_RETRIES = 2
    ROUTER_RETRY_BACKOFF_S = 0.5

    def __init__(
        self,
        llm=None,
        router=None,
        planning_agent=None,
        vision_llm=None,
        vision_model=None,
    ):
        self.llm = llm or ChatOpenAI(
            model=os.getenv("LLM_MODEL_ID"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0.3,
            timeout=60,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.router = router or self.llm.with_structured_output(
            RouteDecision,
            method="function_calling",
        )
        self.planning_agent = planning_agent or PlanningAgent()
        self.vision_llm = vision_llm or MyLLM(model=vision_model or VISION_MODEL)

        self.checkpointer = MemorySaver()
        self.compiled_graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgriState)
        graph.add_node("route", self._route_node)
        graph.add_node("direct", self._direct_node)
        graph.add_node("react", self._react_node)
        graph.add_node("tools", ToolNode(TOOLS))
        graph.add_node("planning", self._planning_node)
        graph.add_node("clarify", self._clarify_node)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {
                "direct": "direct",
                "react": "react",
                "planning": "planning",
                "clarify": "clarify",
            },
        )
        graph.add_edge("direct", END)
        graph.add_conditional_edges(
            "react",
            self._after_react,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "react")
        graph.add_edge("planning", END)
        graph.add_edge("clarify", END)
        return graph

    @staticmethod
    def _known_context(state: AgriState) -> dict:
        return {
            key: state.get(key)
            for key in (
                "crop", "city", "region", "symptom", "need", "growth_stage",
                "has_symptom", "active_goal", "pending_slot",
            )
            if state.get(key) is not None and state.get(key) != ""
        }

    @staticmethod
    def _latest_user_text(state: AgriState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _recent_dialogue(state: AgriState, max_messages: int = 6) -> str:
        """把最近对话作为路由数据嵌入最后一条分类指令，兼顾上下文与工具调用兼容性。"""
        lines = []
        for message in state.get("messages", []):
            if isinstance(message, HumanMessage):
                role = "用户"
            elif isinstance(message, AIMessage) and message.content:
                role = "助手"
            else:
                continue

            content = str(message.content).strip()
            if len(content) > 800:
                content = f"{content[:400]}\n...[中间内容省略]...\n{content[-400:]}"
            lines.append(f"{role}：{content}")
        return "\n".join(lines[-max_messages:])

    @staticmethod
    def _is_explicit_plan_request(text: str) -> bool:
        return any(word in text for word in ("完整计划", "制定计划", "行动计划", "行动方案", "规划一下", "安排一下"))

    @staticmethod
    def _explicit_capabilities(text: str) -> list[Capability]:
        """明确业务词不完全交给概率路由，避免政策等请求落入无工具Direct。"""
        return [
            capability
            for capability, keywords in _EXPLICIT_CAPABILITY_KEYWORDS.items()
            if any(keyword in text for keyword in keywords)
        ]

    @staticmethod
    def _infer_high_confidence_location(text: str, city: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        for known_city, region in _HIGH_CONFIDENCE_CITY_REGIONS.items():
            if known_city in text:
                return known_city, region

        compact_city = (city or "").strip().removesuffix("市")
        region = _HIGH_CONFIDENCE_CITY_REGIONS.get(compact_city)
        return (compact_city, region) if region else (None, None)

    @staticmethod
    def _should_answer_from_context(state: AgriState) -> bool:
        messages = state.get("messages", [])
        has_previous_answer = any(
            isinstance(message, AIMessage) and message.content
            for message in messages[:-1]
        )
        text = AgriGraphAgent._latest_user_text(state)
        return has_previous_answer and any(
            phrase in text
            for phrase in (
                "为什么", "怎么没有", "没有建议", "什么意思", "解释一下", "刚才", "上面", "前面",
                "必须采取的措施", "必须要采取的措施", "最近要做", "接下来怎么做", "具体怎么做", "还有什么建议",
            )
        )

    @staticmethod
    def _is_province_only(value: str) -> bool:
        compact = value.strip()
        for suffix in ("壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省"):
            if compact.endswith(suffix):
                compact = compact[: -len(suffix)]
                break
        return compact in _PROVINCE_ONLY_NAMES

    @staticmethod
    def _infer_growth_stage(text: str) -> Optional[str]:
        for stage, phrases in _GROWTH_STAGE_PATTERNS:
            if any(phrase in text for phrase in phrases):
                return stage
        return None

    @staticmethod
    def _mentions_no_symptom(text: str) -> bool:
        return any(phrase in text for phrase in _NO_SYMPTOM_PHRASES)

    @staticmethod
    def _is_vague_request(text: str) -> bool:
        compact = text.strip().rstrip("。！？?!")
        return len(compact) <= 24 and any(
            phrase in compact for phrase in _VAGUE_REQUEST_PATTERNS
        )

    @staticmethod
    def _infer_pending_slot(answer: str) -> Optional[str]:
        tail = answer[-500:]
        asks_question = any(marker in tail for marker in ("请告诉", "请问", "您目前", "能否告知", "？", "?"))
        if asks_question and any(phrase in tail for phrase in ("哪个阶段", "什么阶段", "生长阶段", "尚未开始种植")):
            return "growth_stage"
        return None

    def _fallback_decision(self, state: AgriState) -> RouteDecision:
        """路由模型异常时的保守兜底，只做高置信关键词判断。"""
        text = self._latest_user_text(state)
        capabilities = []
        if any(word in text for word in ("症状", "病害", "虫害", "发黄", "斑", "卷叶", "枯萎")):
            capabilities.append("diagnosis")
        if any(word in text for word in ("天气", "气温", "下雨", "降雨", "打药", "灌溉", "收割")):
            capabilities.append("weather")
        if any(word in text for word in ("政策", "补贴", "扶持", "申请")):
            capabilities.append("policy")

        explicit_plan = self._is_explicit_plan_request(text)
        if explicit_plan or len(capabilities) >= 2:
            route = "planning"
        elif capabilities:
            route = "react"
        else:
            route = "direct"
        return RouteDecision(route=route, capabilities=capabilities, confidence=0.7)

    def _invoke_router(self, messages: list) -> RouteDecision:
        """对短暂空响应或协议解析失败做有限重试，最终仍交给保守兜底。"""
        last_error: Exception | None = None
        for attempt in range(self.ROUTER_RETRIES + 1):
            try:
                decision = self.router.invoke(messages)
                if decision is None:
                    raise ValueError("路由模型返回空结果")
                if not isinstance(decision, RouteDecision):
                    decision = RouteDecision.model_validate(decision)
                return decision
            except Exception as exc:
                last_error = exc
                if attempt < self.ROUTER_RETRIES:
                    time.sleep(self.ROUTER_RETRY_BACKOFF_S * (attempt + 1))
        raise last_error or RuntimeError("路由模型调用失败")

    def _route_node(self, state: AgriState) -> dict:
        mode = state.get("mode", "auto")
        known = self._known_context(state)

        if mode == "planning":
            capabilities = []
            if known.get("crop") and known.get("symptom"):
                capabilities.append("diagnosis")
            if known.get("city"):
                capabilities.append("weather")
            if known.get("crop") and known.get("region"):
                capabilities.append("policy")
            decision = RouteDecision(
                route="planning",
                capabilities=capabilities,
                confidence=1.0,
            )
        else:
            router_messages = [
                SystemMessage(
                    content=(
                        f"{ROUTER_PROMPT}\n\n"
                        f"当前会话已确认字段：{known or '暂无'}\n"
                        f"上一轮能力需求：{state.get('capabilities', [])}\n"
                        f"上一轮缺失字段：{state.get('missing_fields', [])}\n"
                        f"助手正在等待用户回答的槽位：{state.get('pending_slot') or '无'}\n"
                        f"最近对话中最后一条用户消息是本轮需要分类的对象。"
                    )
                ),
                HumanMessage(
                    content=(
                        "最近对话如下：\n"
                        f"{self._recent_dialogue(state)}\n\n"
                        "请勿回答农业问题本身，必须调用RouteDecision工具返回分类和字段。"
                    )
                ),
            ]
            try:
                decision = self._invoke_router(router_messages)
            except Exception as exc:
                print(f"[AgriGraph] 路由模型失败，使用保守规则兜底：{exc}")
                decision = self._fallback_decision(state)

        latest_user_text = self._latest_user_text(state)
        updates = {}
        for field_name in ("crop", "city", "region", "symptom", "need", "growth_stage"):
            value = getattr(decision, field_name, None)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if field_name == "city" and self._is_province_only(value):
                    continue
                if field_name == "need" and "policy" not in decision.capabilities:
                    continue
                updates[field_name] = value

        location_city, location_region = self._infer_high_confidence_location(
            latest_user_text,
            updates.get("city") or known.get("city"),
        )
        if location_city:
            updates.setdefault("city", location_city)
            updates.setdefault("region", location_region)

        inferred_stage = self._infer_growth_stage(latest_user_text)
        if inferred_stage:
            updates["growth_stage"] = inferred_stage

        no_symptom = self._mentions_no_symptom(latest_user_text)
        if decision.has_symptom is not None:
            updates["has_symptom"] = decision.has_symptom
        if no_symptom or (inferred_stage == "未开始种植" and not updates.get("symptom")):
            updates["has_symptom"] = False
            updates["symptom"] = None
        elif updates.get("symptom"):
            updates["has_symptom"] = True

        effective = {**known, **updates}
        explicit_capabilities = self._explicit_capabilities(latest_user_text)
        vague_request = self._is_vague_request(latest_user_text)
        # 确定性关键词用于补充模型结果，而不是覆盖模型识别出的隐含能力。
        # 例如“玉米果穗变成黑粉包，再看看天气”只有天气命中显式关键词，
        # 但模型识别出的 diagnosis 仍必须保留，才能进入多 Agent 编排。
        capabilities = list(dict.fromkeys([
            *decision.capabilities,
            *explicit_capabilities,
        ]))
        route = decision.route
        confidence = decision.confidence
        next_pending_slot = state.get("pending_slot")
        answered_pending_slot = bool(
            next_pending_slot
            and next_pending_slot in updates
            and updates.get(next_pending_slot) is not None
        )
        if answered_pending_slot:
            next_pending_slot = None
            if not explicit_capabilities:
                route = "direct"
                capabilities = []
            confidence = max(confidence, 0.95)

        # 追问后的短回答（例如只回复“水稻”）可能被模型判成direct。只要它确实
        # 填补了上一轮缺失槽位，就恢复上一轮能力需求，而不是丢掉待办任务。
        pending_fields = set(state.get("missing_fields", []))
        resumed_pending_request = (
            state.get("route") == "clarify"
            and bool(state.get("capabilities"))
            and bool(pending_fields & updates.keys())
            and not capabilities
        )
        if resumed_pending_request:
            capabilities = list(state["capabilities"])
            route = "planning" if len(capabilities) >= 2 else "react"

        if self._should_answer_from_context(state) and not explicit_capabilities:
            route = "direct"
            capabilities = []
            confidence = max(confidence, 0.8)

        if vague_request and not explicit_capabilities and not self._should_answer_from_context(state):
            route = "clarify"
            capabilities = []
            confidence = max(confidence, 0.9)

        if explicit_capabilities:
            route = (
                "planning"
                if self._is_explicit_plan_request(latest_user_text) or len(capabilities) >= 2
                else "react"
            )
            confidence = max(confidence, 0.95)

        if effective.get("growth_stage") == "未开始种植" and effective.get("has_symptom") is False:
            capabilities = [capability for capability in capabilities if capability != "diagnosis"]
            if route in ("react", "planning", "clarify") and not capabilities:
                route = "direct"
                confidence = max(confidence, 0.9)

        if route == "direct":
            capabilities = []
        elif route == "react" and len(capabilities) >= 2:
            route = "planning"
        elif route == "planning" and len(capabilities) == 1 and not self._is_explicit_plan_request(latest_user_text):
            route = "react"
        elif route == "planning" and not capabilities:
            route = "clarify" if self._is_explicit_plan_request(latest_user_text) else "direct"

        if "policy" not in capabilities and "need" in updates:
            updates.pop("need")
            if "need" not in known:
                effective.pop("need", None)

        if "policy" in capabilities and not effective.get("need"):
            updates["need"] = "种植补贴"
            effective["need"] = "种植补贴"

        previous_goal = state.get("active_goal")
        if "diagnosis" in capabilities:
            updates["active_goal"] = "作物诊断"
        elif "weather" in capabilities:
            updates["active_goal"] = "天气农事"
        elif "policy" in capabilities:
            updates["active_goal"] = "政策查询"
        elif effective.get("growth_stage") == "未开始种植":
            updates["active_goal"] = "种植准备"
        elif any(word in latest_user_text for word in ("种植", "种水稻", "怎么种")):
            updates["active_goal"] = "种植咨询"
        elif previous_goal:
            updates["active_goal"] = previous_goal

        missing_fields = []
        for capability in capabilities:
            for field_name in _REQUIRED_FIELDS[capability]:
                if not effective.get(field_name) and field_name not in missing_fields:
                    missing_fields.append(field_name)

        if vague_request and route == "clarify":
            missing_fields = ["request"]
        elif route == "clarify" and not missing_fields:
            allowed = set(_SLOT_LABELS)
            missing_fields = [name for name in decision.missing_fields if name in allowed] or ["request"]
        elif missing_fields:
            route = "clarify"
        elif confidence < self.MIN_ROUTE_CONFIDENCE:
            route = "clarify"
            missing_fields = ["request"]

        return {
            **updates,
            "route": route,
            "capabilities": capabilities,
            "confidence": confidence,
            "missing_fields": missing_fields,
            "tool_rounds": 0,
            "task_status": {},
            "pending_slot": next_pending_slot,
        }

    def _model_messages(self, state: AgriState) -> list:
        known = self._known_context(state)
        context_note = f"\n\n当前会话已确认的农业信息：{known}" if known else ""
        return [
            SystemMessage(content=f"{SYSTEM_PROMPT}{context_note}"),
            *state.get("messages", []),
        ]

    def _direct_messages(self, state: AgriState) -> list:
        known = self._known_context(state)
        return [
            SystemMessage(
                content=(
                    f"{DIRECT_PROMPT}\n\n"
                    f"今天是{date.today().isoformat()}。\n"
                    f"当前会话已确认的农业信息：{known or '暂无'}"
                )
            ),
            *state.get("messages", []),
        ]

    def _direct_node(self, state: AgriState) -> dict:
        response = self.llm.invoke(self._direct_messages(state))
        return {
            "messages": [response],
            "pending_slot": self._infer_pending_slot(str(response.content)),
        }

    def _react_node(self, state: AgriState) -> dict:
        rounds = state.get("tool_rounds", 0)
        if state.get("messages") and isinstance(state["messages"][-1], ToolMessage):
            rounds += 1
        if rounds >= self.MAX_TOOL_ROUNDS:
            return {
                "messages": [AIMessage(content="工具调用次数已达到上限，请缩小问题范围后再试。")],
                "tool_rounds": rounds,
            }

        model = self.llm_with_tools
        capabilities = state.get("capabilities") or []
        # 路由器已经确认单一能力且必填槽位齐全时，第一轮必须落到对应工具，
        # 避免模型跳过检索、直接凭参数知识回答。工具返回后恢复自动选择，
        # 政策检索仍可根据本地结果继续调用联网回退。
        if rounds == 0 and len(capabilities) == 1:
            tool_name = _CAPABILITY_TOOL_NAMES.get(capabilities[0])
            if tool_name:
                model = self.llm.bind_tools(TOOLS, tool_choice=tool_name)

        response = model.invoke(self._model_messages(state))
        if rounds == 0 and len(capabilities) == 1 and not getattr(response, "tool_calls", None):
            capability = capabilities[0]
            tool_name = _CAPABILITY_TOOL_NAMES.get(capability)
            tool_args = {
                "diagnosis": {
                    "crop": state.get("crop"),
                    "symptom_text": state.get("symptom"),
                },
                "weather": {"city": state.get("city")},
                "policy": {
                    "crop": state.get("crop"),
                    "region": state.get("region"),
                    "need": state.get("need") or "种植补贴",
                },
            }.get(capability)
            if tool_name and tool_args and all(value for value in tool_args.values()):
                # 部分 OpenAI-Compatible 服务会忽略 tool_choice。此处把已由路由器
                # 确认的能力和已校验槽位转换为标准 ToolCall，确保先检索再回答。
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"guard-{uuid4().hex}",
                    }],
                )
        return {"messages": [response], "tool_rounds": rounds}

    @staticmethod
    def _after_react(state: AgriState) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "end"

    def _planning_node(self, state: AgriState) -> dict:
        try:
            answer = self.planning_agent.run(
                crop=state.get("crop"),
                city=state.get("city"),
                region=state.get("region"),
                symptom_text=state.get("symptom"),
                need=state.get("need") or "种植补贴",
                capabilities=state.get("capabilities"),
            )
            task_status = {
                key: {
                    "status": task.status,
                    "error": task.error,
                    "protocol": getattr(task, "protocol", "local"),
                    "task_id": getattr(task, "task_id", None),
                    "fallback_reason": getattr(task, "fallback_reason", None),
                }
                for key, task in self.planning_agent.last_tasks.items()
            }
        except Exception as exc:
            answer = f"生成综合计划时发生错误：{exc}。请稍后重试，或先单独咨询具体问题。"
            task_status = {"planning": {"status": "failed", "error": str(exc)}}
        return {"messages": [AIMessage(content=answer)], "task_status": task_status}

    @staticmethod
    def _clarify_node(state: AgriState) -> dict:
        labels = [_SLOT_LABELS.get(name, name) for name in state.get("missing_fields", [])]
        if labels == [_SLOT_LABELS["request"]]:
            content = "请再具体描述一下你想解决的问题，例如作物症状、天气、补贴，或需要制定完整计划。"
        else:
            content = f"为了继续处理，请补充：{'、'.join(labels)}。"
        pending_slot = state.get("missing_fields", [None])[0] if state.get("missing_fields") else None
        return {"messages": [AIMessage(content=content)], "pending_slot": pending_slot}

    def _describe_image(self, image_data_url: str) -> str:
        return self.vision_llm.invoke(
            [{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }]
        )

    def run(
        self,
        user_message: str,
        thread_id: str,
        image_data_url: str = None,
        mode: str = "auto",
        context: dict = None,
        history: list[dict] = None,
    ) -> str:
        if image_data_url:
            try:
                description = self._describe_image(image_data_url)
                user_message = f"{user_message}\n\n[图片识别到的症状描述：{description}]"
            except Exception as exc:
                user_message = (
                    f"{user_message}\n\n[用户上传了一张图片，但图片识别失败（{exc}），"
                    "请基于文字部分回答，并建议用户补充作物和症状描述]"
                )

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.MAX_TOOL_ROUNDS * 2 + 6,
        }
        existing_state = self.compiled_graph.get_state(config)
        restored_messages = []
        if not (existing_state.values or {}).get("messages") and history:
            restored_messages = deserialize_messages(history)

        payload = {
            "messages": [*restored_messages, HumanMessage(content=user_message)],
            "mode": mode,
        }
        allowed_context = {
            "crop", "city", "region", "symptom", "need", "growth_stage",
            "has_symptom", "active_goal", "pending_slot",
        }
        for key, value in (context or {}).items():
            if key in allowed_context and value is not None and value != "":
                payload[key] = value

        result = self.compiled_graph.invoke(payload, config=config)
        return result["messages"][-1].content

    def get_history(self, thread_id: str) -> list[dict]:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.compiled_graph.get_state(config)
        messages = state.values.get("messages", []) if state.values else []
        return serialize_messages(messages)

    def get_snapshot(self, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.compiled_graph.get_state(config)
        values = state.values or {}
        return {
            "route": values.get("route"),
            "capabilities": values.get("capabilities", []),
            "confidence": values.get("confidence"),
            "missing_fields": values.get("missing_fields", []),
            "context": self._known_context(values),
            "task_status": values.get("task_status", {}),
            "pending_slot": values.get("pending_slot"),
        }
