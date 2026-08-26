# backend/src/agri_agent/models/schemas.py
"""
FastAPI接口用的请求/响应数据结构（Pydantic模型）。放在单独的models模块里，
不和api/tools/agents混在一起——接口的"数据形状"和接口的"业务逻辑"是两件
独立的事，分开之后FastAPI路由函数只管调用Agent、不用关心字段校验的细节，
Pydantic会自动做类型校验、生成OpenAPI文档（访问 /docs 能看到交互式接口文档）。
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DiagnosisRequest(BaseModel):
    crop: str = Field(..., description="作物名称，比如'水稻'")
    symptom_text: str = Field(..., description="症状描述，比如'叶子发黄'")


class DiagnosisResponse(BaseModel):
    result: str


class WeatherRequest(BaseModel):
    city: str = Field(..., description="城市名，比如'长沙'（高德天气查询用，城市级别）")


class WeatherResponse(BaseModel):
    result: str


class PolicyRequest(BaseModel):
    crop: str = Field(..., description="作物名称")
    region: str = Field(..., description="省级行政区，比如'湖南省'（政策库匹配用，省级粒度）")
    need: str = Field(default="种植补贴", description="想咨询的政策需求，不填则用默认值")


class PolicyResponse(BaseModel):
    result: str


class PlanningRequest(BaseModel):
    crop: str = Field(..., description="作物名称")
    city: str = Field(..., description="城市名（给天气用）")
    region: str = Field(..., description="省级行政区（给政策用）")
    symptom_text: Optional[str] = Field(default=None, description="症状描述，不填就跳过诊断这一步")
    need: str = Field(default="种植补贴", description="想咨询的政策需求")


class PlanningResponse(BaseModel):
    result: str


class AgriContext(BaseModel):
    crop: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    symptom: Optional[str] = None
    need: Optional[str] = None
    growth_stage: Optional[str] = None
    has_symptom: Optional[bool] = None
    active_goal: Optional[str] = None
    pending_slot: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户本轮输入的自然语言消息")
    # history是AgriGraphAgent返回的messages列表格式（role/content/tool_calls这些字段），
    # 不是自定义结构，所以用比较宽松的Dict[str, Any]，不单独为它定义严格的Pydantic模型——
    # 这份数据是后端自己生成、前端只负责原样存和原样传回来，不需要校验内部字段
    history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="之前的对话历史（不含system prompt），第一次调用可以不传"
    )
    image_data_url: Optional[str] = Field(
        default=None,
        description="用户上传的图片，data URL格式（如'data:image/jpeg;base64,xxx'），不上传图片就不传这个字段",
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="对话线程ID；AgriGraph用它隔离并续接会话",
    )
    mode: Literal["auto", "planning"] = Field(
        default="auto",
        description="auto由统一路由器判断；planning强制进入Planning节点",
    )
    context: Optional[AgriContext] = Field(
        default=None,
        description="前端已经确认的农业字段；用于修正模型提取结果和后端重启后的上下文恢复",
    )


class ChatResponse(BaseModel):
    result: str
    history: List[Dict[str, Any]] = Field(..., description="更新后的完整对话历史，前端需要存起来，下一轮调用时传回来")
    thread_id: Optional[str] = Field(default=None, description="本轮使用的对话线程ID")
    route: Optional[str] = Field(default=None, description="统一图本轮实际采用的执行路线")
    capabilities: List[str] = Field(default_factory=list, description="本轮需要的农业能力")
    context: Dict[str, Any] = Field(default_factory=dict, description="会话中已经确认的农业字段")
    missing_fields: List[str] = Field(default_factory=list, description="继续执行前仍需要用户补充的字段")
    confidence: Optional[float] = Field(default=None, description="结构化路由判断置信度")
    task_status: Dict[str, Any] = Field(default_factory=dict, description="Planning节点中各专业任务的执行状态")
    pending_slot: Optional[str] = Field(default=None, description="助手上一轮明确询问、等待用户回答的槽位")
