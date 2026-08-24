# backend/src/agri_agent/models/schemas.py
"""
FastAPI接口用的请求/响应数据结构（Pydantic模型）。放在单独的models模块里，
不和api/tools/agents混在一起——接口的"数据形状"和接口的"业务逻辑"是两件
独立的事，分开之后FastAPI路由函数只管调用Agent、不用关心字段校验的细节，
Pydantic会自动做类型校验、生成OpenAPI文档（访问 /docs 能看到交互式接口文档）。
"""
from typing import Any, Dict, List, Optional

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


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户本轮输入的自然语言消息")
    # history是ChatAgent.run()返回的messages列表格式（role/content/tool_calls这些字段），
    # 不是自定义结构，所以用比较宽松的Dict[str, Any]，不单独为它定义严格的Pydantic模型——
    # 这份数据是后端自己生成、前端只负责原样存和原样传回来，不需要校验内部字段
    history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="之前的对话历史（不含system prompt），第一次调用可以不传"
    )
    image_data_url: Optional[str] = Field(
        default=None,
        description="用户上传的图片，data URL格式（如'data:image/jpeg;base64,xxx'），不上传图片就不传这个字段",
    )


class ChatResponse(BaseModel):
    result: str
    history: List[Dict[str, Any]] = Field(..., description="更新后的完整对话历史，前端需要存起来，下一轮调用时传回来")
