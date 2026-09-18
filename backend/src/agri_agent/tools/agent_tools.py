"""AgriGraph ReAct 分支暴露给模型的原始工具注册表。"""
import asyncio

from langchain_core.tools import tool

from agri_agent.agents.weather_agent import query_weather
from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge
from agri_agent.tools.semantic_pest_knowledge_tool import semantic_match_pest_knowledge
from agri_agent.tools.long_context_diagnosis_tool import long_context_diagnose
from agri_agent.tools.policy_match_tool import match_policy
from agri_agent.tools.web_search_tool import web_search


@tool
def diagnose_crop_disease(crop: str, symptom_text: str) -> dict:
    """查询本地农作物病虫害知识库，根据作物名称和症状描述返回可能病因和处理建议。
    适用于用户描述了具体症状（比如叶子发黄、有虫斑、卷叶等）的情况。"""
    matched = match_pest_knowledge(crop, symptom_text)
    if not matched:
        matched = semantic_match_pest_knowledge(crop, symptom_text)
    if not matched:
        matched = long_context_diagnose(crop, symptom_text)
    return {"matched_knowledge": matched}


@tool
def get_weather_forecast(city: str) -> dict:
    """查询指定城市的实时天气预报，用于判断近期是否适合打药、灌溉、收割等农事操作。"""
    return {"weather_data": asyncio.run(query_weather(city))}


@tool
def search_subsidy_policy(crop: str, region: str, need: str) -> dict:
    """在本地政策知识库中检索农业补贴政策（关键词粗筛+向量语义精排两阶段检索）。
    region必须是省级行政区（比如"湖南省"而不是"长沙"）。本地库没有覆盖时，
    应继续调用web_search_policy联网搜索，不要直接放弃。"""
    return {"matched_policies": match_policy(crop, region, need)}


@tool
def web_search_policy(query: str) -> dict:
    """联网搜索最新农业补贴政策。只在本地政策检索无结果时调用。"""
    return {"search_results": web_search(query)}


TOOLS = [
    diagnose_crop_disease,
    get_weather_forecast,
    search_subsidy_policy,
    web_search_policy,
]
