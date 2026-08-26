# src/agri_agent/mcp_server.py
"""
把AgriAgent自己的三个核心工具（诊断/天气/政策）反过来暴露成一个MCP Server，
让别的支持MCP协议的客户端（比如Claude Desktop、别的Agent项目）可以直接调用。

这跟weather_agent.py里"调用高德天气MCP Server"是同一个协议的两个方向：
之前AgriAgent只是MCP的消费方（client，去调别人暴露的工具），这个文件让AgriAgent
也能当MCP的服务方（server，把自己的能力暴露给别人）——同一个项目里同时实践了
MCP协议的两端，比只做过其中一端的故事更完整。

暴露的是三个原始工具函数（match_pest_knowledge/query_weather/match_policy），
不是包了一层大模型总结的CropDiagnosisAgent.run()这些方法——原因和AgriGraph的
工具注册表设计是同一个道理（见tools/agent_tools.py）：调用这个MCP Server的是另一个
Agent/大模型，它自己会对拿到的结构化数据做推理和总结，如果这里已经先用大模型
总结成一段自然语言，等于让别人的Agent再去理解、再加工一段"别的大模型已经生成好的
文字"，多余且信息有损失（结构化数据里的原始字段，总结成文字后就丢掉了）。工具只吐
数据、生成自然语言的事交给调用方自己的Agent去做，这个边界在整个项目里是一贯的。

联网搜索（web_search_tool）没有暴露成MCP工具——它是个通用能力，不是AgriAgent的
差异化能力，调用方自己的Agent大概率已经有自己的联网搜索工具，没必要重复暴露一个。

运行方式：
    python backend/src/agri_agent/mcp_server.py
默认走stdio传输，配合MCP客户端的配置文件使用（比如Claude Desktop的
claude_desktop_config.json），配置里的command/args指向这个文件的绝对路径即可，
不需要额外起一个新进程/新端口——跟FastAPI的HTTP服务是两套完全独立的暴露方式，
互不冲突，可以同时存在。

手动测试（不需要真的接一个MCP客户端）：
    pip install "mcp[cli]"
    mcp dev backend/src/agri_agent/mcp_server.py
会在浏览器打开一个交互式的Inspector页面，可以直接点击调用每个工具、看输入输出，
是官方SDK提供的标准调试方式。
"""
import sys
from pathlib import Path

# 这个文件直接在 agri_agent/ 目录下（不像agents/、tools/里的文件深一层），
# 所以只需要parents[1]就能到backend/src——MCP客户端通常是拿这个文件的绝对路径
# 直接执行（不是用python -m这种包调用方式），所以跟项目里其他__main__脚本一样，
# 需要手动把src目录加进sys.path，才能找到agri_agent这个包
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge
from agri_agent.tools.policy_match_tool import match_policy
from agri_agent.agents.weather_agent import query_weather

mcp = FastMCP("AgriAgent")


@mcp.tool()
def diagnose_crop_disease(crop: str, symptom_text: str) -> list:
    """查询本地农作物病虫害知识库，根据作物名称和症状描述返回可能的病因和处理建议。
    适用于用户描述了具体症状（比如叶子发黄、有虫斑、卷叶等）的情况。
    返回结构化的匹配记录列表，可能是空列表（表示本地知识库没有覆盖到这个症状）。
    """
    return match_pest_knowledge(crop, symptom_text)


@mcp.tool()
async def get_weather_forecast(city: str) -> str:
    """查询指定城市的实时天气预报原始数据，用于判断近期是否适合打药、灌溉、收割等农事操作。
    city需要是城市级别的地名（比如"长沙"），不是省级行政区。
    """
    return await query_weather(city)


@mcp.tool()
def search_subsidy_policy(crop: str, region: str, need: str) -> list:
    """在本地农业补贴政策知识库中检索（关键词粗筛+向量语义精排两阶段检索，
    地区不匹配的政策会被硬过滤掉）。region需要是省级行政区（比如"湖南省"），
    不是城市。返回按相关度排序的匹配记录列表，每条带match_score和match_reason，
    可能是空列表（表示本地库没有覆盖到、或者相关度都不够）。
    """
    return match_policy(crop, region, need)


if __name__ == "__main__":
    mcp.run()
