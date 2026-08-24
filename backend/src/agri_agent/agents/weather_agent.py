import asyncio
import os
import sys
from pathlib import Path

# 把 src 目录加入 Python 的模块搜索路径，这样才能找到 agri_agent.core 下面咱们自己写的模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agri_agent.core.my_llm import MyLLM
from agri_agent.core.my_agent import MyAgent


async def query_weather(city: str) -> str:
    """调用高德MCP查询天气，返回原始JSON字符串"""
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@amap/amap-maps-mcp-server"],
        env={"AMAP_MAPS_API_KEY": os.environ["AMAP_MAPS_API_KEY"]},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("maps_weather", arguments={"city": city})
            return result.content[0].text


class WeatherAgent:
    def __init__(self, llm=None):
        self.llm = llm or MyLLM()
        self.agent = MyAgent("WeatherAgent", self.llm)

    def run(self, city: str) -> str:
        weather_data = asyncio.run(query_weather(city))
        prompt = f"""你是一个农事天气顾问。根据下面的天气预报数据，判断未来几天是否适合打农药、灌溉、收割等农事操作，给出简洁的建议（3句话以内）：

{weather_data}
"""
        return self.agent.run(prompt)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = WeatherAgent()
    print(agent.run("南京"))