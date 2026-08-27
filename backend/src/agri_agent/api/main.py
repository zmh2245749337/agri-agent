# backend/src/agri_agent/api/main.py
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agri_agent.agents.agri_graph import AgriGraphAgent
from agri_agent.core.my_llm import MyLLM
from agri_agent.agents.planning_agent import PlanningAgent
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
_chat_agent = AgriGraphAgent(planning_agent=_planning_agent)


@app.get("/health")
def health():
    return {"status": "ok", "agent": "agri_graph"}


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
    # PlanningAgent.run()内部已经把进程内任务与A2A远程任务统一成SubAgentTask，
    # 并处理了单个子Agent失败和A2A不可用时的本地回退；这里兜底的是三个子Agent全部失败/
    # 传参有问题"这类更外层的异常
    try:
        result = _chat_agent.run(
            "请根据已填写的信息生成一份完整农事行动计划。",
            thread_id=uuid.uuid4().hex,
            mode="planning",
            context={
                "crop": req.crop,
                "city": req.city,
                "region": req.region,
                "symptom": req.symptom_text,
                "need": req.need,
            },
        )
        return PlanningResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # AgriGraphAgent.run()负责图内降级，这里兜底前端历史格式等更外层异常。
    try:
        thread_id = req.thread_id or uuid.uuid4().hex
        context = req.context.model_dump(exclude_none=True) if req.context else None
        result = _chat_agent.run(
            req.message,
            thread_id=thread_id,
            image_data_url=req.image_data_url,
            mode=req.mode,
            context=context,
            history=req.history,
        )
        history = _chat_agent.get_history(thread_id)
        snapshot = _chat_agent.get_snapshot(thread_id)

        return ChatResponse(
            result=result,
            history=history,
            thread_id=thread_id,
            route=snapshot.get("route"),
            capabilities=snapshot.get("capabilities", []),
            context=snapshot.get("context", {}),
            missing_fields=snapshot.get("missing_fields", []),
            confidence=snapshot.get("confidence"),
            task_status=snapshot.get("task_status", {}),
            pending_slot=snapshot.get("pending_slot"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
