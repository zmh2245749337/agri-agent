# src/agri_agent/tools/web_search_tool.py
import os

import requests

# 官方REST接口，走标准Bearer鉴权，跟chat接口是同一个智谱账号/密钥，
# 不需要额外申请，也不需要装官方的zai SDK——用requests直接调HTTP接口更轻量，
# 跟项目里其他地方的依赖风格（openai库直接打Zhipu的OpenAI兼容端点）也是一个思路：
# 优先用标准/通用的调用方式，不为了用SDK而多引入一个专用依赖
WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"


def web_search(query: str, count: int = 5) -> list:
    """
    调用智谱Web Search API做实时联网搜索。

    只负责"拿数据"，不在这一层调用大模型——跟policy_match_tool.py一样的
    Tool职责边界（Tool只吐结构化数据，生成自然语言回答的事交给Agent层做）。

    返回：[{"title":..., "content":..., "link":..., "publish_date":...}, ...]
    查不到或请求失败时返回空列表，不抛异常中断上层调用。
    """
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("未找到LLM_API_KEY环境变量，请检查.env文件")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "search_query": query[:70],  # 官方建议不超过70字符，超出部分直接截断
        "search_engine": "search_std",  # 基础档位引擎，按量计费里最便宜的一档，够用
        "search_intent": False,  # 不需要额外做"要不要搜索"的意图识别，直接搜
        "count": count,
        "content_size": "medium",  # 摘要长度适中，太长的话拼进prompt会占用不少token
    }

    try:
        response = requests.post(WEB_SEARCH_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # 网络问题或接口报错时不让整个Agent崩掉，返回空列表交给上层做兜底提示
        print(f"[web_search_tool] 请求失败：{e}")
        return []

    results = []
    for item in data.get("search_result", []):
        results.append({
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "link": item.get("link", ""),
            "publish_date": item.get("publish_date", ""),
        })
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    for r in web_search("2026年云南省咖啡种植补贴政策"):
        print(f"[{r['publish_date']}] {r['title']} —— {r['link']}")
