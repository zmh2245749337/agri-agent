# src/agri_agent/core/a2a_lite.py
"""
借鉴A2A（Agent2Agent）协议核心设计思想的一个轻量实现，用来规范PlanningAgent
和它的三个子Agent（CropDiagnosisAgent/WeatherAgent/PolicySubsidyAgent）之间
的调用方式。

先说清楚这个模块"不是什么"：这不是A2A官方规范的完整实现。真正的A2A协议是
JSON-RPC 2.0 + HTTP传输，要求每个Agent各自暴露一个独立的HTTP服务、提供
/.well-known/agent.json做能力发现、任务生命周期支持异步轮询/推送通知等。
对本项目里"一个进程内3个子Agent"这种规模去实现完整协议栈，属于过度设计——
真要给面试官解释"为什么不上完整协议"，答案就是这个：协议本身要解决的是
"跨进程/跨团队/跨组织的Agent互操作"问题，本项目里3个子Agent同属一个代码库、
同一个进程，没有这个互操作需求，硬套完整协议只会增加复杂度，不会带来实际收益。

这个模块借的是A2A"设计理念"上的三个核心概念，而不是协议细节：
1. AgentCard（能力名片）——每个Agent不再是一个被裸调用的函数，而是先声明
   "我是谁、我能干什么"，调用方按这张名片来分发任务，而不是直接耦合到
   具体的方法签名上。
2. Message/Part（消息用统一的信封包装）——输入输出不再是裸的字符串参数，
   而是包一层{role, parts}的结构，parts里每个part标明类型（这里只用到
   "text"类型）。这样以后要给某个子Agent的输入/输出加别的模态（比如一张图片
   part），协议结构不需要改。
3. Task状态机（submitted → working → completed/failed）——每次调用都产出
   一个有明确生命周期状态的Task对象，而不是"要么拿到字符串，要么吃一个裸
   异常"。这跟之前PlanningAgent里_safe_call做的事情（catch异常、返回None）
   本质上是同一件事，但状态显式化之后，天然就有了可观测性——比如以后想把
   PlanningAgent自己的执行过程也做成一个类似ChatAgent工具调用轨迹面板那样
   的可视化界面，数据结构已经现成了，不需要另外再设计一套。

传输层维持进程内函数调用（没有走HTTP/JSON-RPC），这是刻意的取舍：先把
"Agent间通信该长什么样子"这个协议形状定下来、跑通，如果以后真的需要把某个
子Agent拆成独立服务（比如WeatherAgent单独部署、被别的项目复用），只需要把
dispatch_task内部"直接调用run_fn()"换成"发一个HTTP请求"，上层PlanningAgent
和调用方完全不用改，因为它们只依赖AgentCard/Message/Task这几个数据结构，
不依赖"调用是进程内还是跨进程"这个实现细节——这也是协议化设计相比"直接
写死函数调用"的核心价值所在。
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class AgentCard:
    """Agent的"能力名片"，对应A2A协议里的Agent Card概念（这里做了简化，
    只保留跟本项目场景相关的字段，略去了url/provider/version等面向
    跨组织服务发现的字段——本项目里Agent发现是代码里写死的，不需要那些）"""

    name: str
    description: str
    skills: list = field(default_factory=list)  # 能力标签，比如["病虫害诊断", "知识库检索"]
    input_modes: list = field(default_factory=lambda: ["text"])
    output_modes: list = field(default_factory=lambda: ["text"])


@dataclass
class Message:
    """统一的消息信封，对应A2A协议里的Message概念。role是"user"（发给Agent的
    输入）或"agent"（Agent给出的输出）；parts是一个列表，每个元素是
    {"type": "text", "text": "..."}这样的字典——本项目目前只用到text类型，
    但结构上留了口子，以后要传别的类型（比如data/file）不需要改这个类"""

    role: str
    parts: list

    @property
    def text(self) -> str:
        """本项目里几乎所有Part都是text类型，这个便捷属性省得每次都手动
        从parts[0]里取text字段"""
        for part in self.parts:
            if part.get("type") == "text":
                return part.get("text", "")
        return ""

    @staticmethod
    def of_text(role: str, text: str) -> "Message":
        return Message(role=role, parts=[{"type": "text", "text": text}])


@dataclass
class Task:
    """一次Agent调用的完整生命周期记录，对应A2A协议里的Task概念（同样做了
    简化，去掉了sessionId/artifacts/history等本项目用不上的字段）。
    status只有四种取值：submitted（已创建，还没开始跑）→ working（正在跑）→
    completed（成功，output_message有值）或failed（失败，error有值）"""

    task_id: str
    agent_name: str
    status: str
    input_message: Message
    output_message: Optional[Message] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    @property
    def result_text(self) -> str:
        """成功时返回output_message里的文本，失败/还没跑完时返回空字符串——
        调用方不用先判断succeeded再取值，直接拿result_text就行，跟之前
        _safe_call失败时返回None、调用方用`if results.get(...)`判断是
        类似的使用习惯，改造前后对下游代码的影响降到最低"""
        return self.output_message.text if self.output_message else ""


def dispatch_task(card: AgentCard, run_fn: Callable[[], str], input_text: str) -> Task:
    """把"调用一个子Agent"这件事包装成一次标准的Task生命周期，替代之前
    PlanningAgent._safe_call()里"try/except+返回None"的裸调用方式。

    run_fn是一个无参可调用对象（调用方用lambda或functools.partial把子Agent
    真正需要的参数提前绑定好，比如`lambda: self.diagnosis_agent.run(crop, symptom_text)`），
    这样dispatch_task本身不需要关心每个子Agent的方法签名长什么样，只关心
    "调用它、记录结果"这一件事——这也是A2A"Agent Card+统一Task接口"设计的
    好处：调用方（PlanningAgent）只需要认识AgentCard和Task这两个通用概念，
    不需要为每个子Agent的方法签名单独写一份适配代码。
    """
    task = Task(
        task_id=uuid.uuid4().hex[:8],
        agent_name=card.name,
        status="submitted",
        input_message=Message.of_text("user", input_text),
        started_at=time.time(),
    )
    task.status = "working"
    try:
        output_text = run_fn()
        task.output_message = Message.of_text("agent", output_text)
        task.status = "completed"
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        print(f"[A2A-lite] Task[{task.task_id}] {card.name} 执行失败：{e}，本次将跳过这部分")
    finally:
        task.finished_at = time.time()
    return task
