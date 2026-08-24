# backend/src/agri_agent/api/main.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agri_agent.core.my_llm import MyLLM
from agri_agent.agents.planning_agent import PlanningAgent
from agri_agent.agents.chat_agent import ChatAgent
from agri_agent.models.schemas import (
    DiagnosisRequest,
    DiagnosisResponse,
    WeatherRequest,
    WeatherResponse,
    PolicyRequest,
    PolicyResponse,
    PlanningRequest,
    PlanningResponse,
    ChatRequest,
    ChatResponse,
)

app = FastAPI(title="AgriAgent 智慧农事助手 API", version="0.1.0")

# 前端（Streamlit）跑在另一个端口/另一台机器上，浏览器直接从Streamlit页面发请求到这里
# 会被同源策略拦截，需要开CORS。开发阶段先允许所有来源，部署时应该收紧成前端的具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局只初始化一次：MyLLM（一个OpenAI client连接）+ PlanningAgent。
# PlanningAgent内部已经持有诊断/天气/政策三个子Agent（共享同一个llm实例），
# 这里直接复用它们做单独查询的接口，不用重复创建、重复初始化连接。
# BGE向量模型、BM25索引这些重量级加载在Tool模块导入时只做一次（模块级代码），
# 不管上面创建了几个Agent实例，都不会重复加载模型——这是FastAPI应用启动时
# 只应该付一次的"冷启动成本"，不能出现在每个请求里
_llm = MyLLM()
_planning_agent = PlanningAgent(llm=_llm)
_diagnosis_agent = _planning_agent.diagnosis_agent
_weather_agent = _planning_agent.weather_agent
_policy_agent = _planning_agent.policy_agent
_chat_agent = ChatAgent(llm=_llm)  # 固定流水线（PlanningAgent）和Function Calling（ChatAgent）两套并存，共享同一个llm连接


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnosis", response_model=DiagnosisResponse)
def diagnosis(req: DiagnosisRequest):
    try:
        result = _diagnosis_agent.run(req.crop, req.symptom_text)
        return DiagnosisResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/weather", response_model=WeatherResponse)
def weather(req: WeatherRequest):
    try:
        result = _weather_agent.run(req.city)
        return WeatherResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/policy", response_model=PolicyResponse)
def policy(req: PolicyRequest):
    # policy_agent.run()内部已经有try/except兜底（网络/大模型失败会降级返回原始检索
    # 数据，不会抛异常），这里的try/except是防止未预料到的异常（比如match_policy本身
    # 报错）导致整个接口返回500而没有任何错误信息
    try:
        result = _policy_agent.run(req.crop, req.region, req.need)
        return PolicyResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/planning", response_model=PlanningResponse)
def planning(req: PlanningRequest):
    # PlanningAgent.run()内部走a2a_lite.dispatch_task的Task状态机已经处理了
    # "某个子Agent失败"的情况，这里的try/except兜底的是"三个子Agent全部失败/
    # 传参有问题"这类更外层的异常
    try:
        result = _planning_agent.run(
            crop=req.crop,
            city=req.city,
            region=req.region,
            symptom_text=req.symptom_text,
            need=req.need,
        )
        return PlanningResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # ChatAgent.run()内部已经有try/except兜底（大模型调用持续失败时返回友好提示，
    # 不会抛异常），这里的try/except兜底的是更外层的意外情况（比如前端传来的
    # history格式有问题导致的异常）
    try:
        result, history = _chat_agent.run(req.message, history=req.history, image_data_url=req.image_data_url)
        return ChatResponse(result=result, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
