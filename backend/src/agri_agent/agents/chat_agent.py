# src/agri_agent/agents/chat_agent.py
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from agri_agent.core.my_llm import MyLLM
from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge
from agri_agent.tools.policy_match_tool import match_policy
from agri_agent.tools.web_search_tool import web_search
from agri_agent.agents.weather_agent import query_weather


# 四个原始工具的JSON Schema——注意暴露的是原始工具（match_pest_knowledge/query_weather/
# match_policy/web_search），不是WeatherAgent.run()这种"已经调用过一次大模型"的方法。
# 这样自然语言生成（RAG里的"G"）全部由ChatAgent这一个大模型在最后统一完成，
# 不会出现"一个大模型总结另一个大模型已经生成好的文字"这种多余、低效的调用
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "diagnose_crop_disease",
            "description": (
                "查询本地农作物病虫害知识库，根据作物名称和症状描述返回可能病因和处理建议。"
                "适用于用户描述了具体症状（比如叶子发黄、有虫斑、卷叶等）的情况。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "作物名称，比如水稻、小麦"},
                    "symptom_text": {"type": "string", "description": "用户描述的症状原文"},
                },
                "required": ["crop", "symptom_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "查询指定城市的实时天气预报，用于判断近期是否适合打药、灌溉、收割等农事操作。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，比如长沙、南京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_subsidy_policy",
            "description": (
                "在本地政策知识库中检索农业补贴政策（关键词粗筛+向量语义精排两阶段检索）。"
                "region必须是省级行政区（比如'湖南省'而不是'长沙'）。"
                "如果返回的matched_policies是空列表，说明本地库没有覆盖，应该接着调用web_search_policy联网搜索，不要就此放弃。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "作物名称"},
                    "region": {"type": "string", "description": "省级行政区，比如'湖南省'"},
                    "need": {"type": "string", "description": "用户的政策需求描述，比如'种植补贴'"},
                },
                "required": ["crop", "region", "need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_policy",
            "description": "联网搜索最新的农业补贴政策信息。只应该在search_subsidy_policy本地检索不到结果时调用，不要跳过本地检索直接联网搜。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词，建议包含地区、作物、需求"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是"智慧农事助手"，一个帮助农民解决种植问题的AI助手，能力包括：作物病虫害诊断、天气与农事建议、农业补贴政策查询。

调用工具的规则：
1. 用户提到具体症状（比如"叶子发黄""有虫眼"，包括用户上传图片后系统自动识别出的症状描述）时，调用diagnose_crop_disease。如果图片识别结果里已经给出了作物种类判断（比如"作物种类：水稻"），直接拿这个结果当crop参数用，不用再反问用户是什么作物；只有图片没能判断出作物种类（识别结果写的是"不确定"之类），并且用户文字里也没提作物名称时，才需要反问用户这是什么作物。
2. 用户想知道天气、或者要不要打药/灌溉/收割时，调用get_weather_forecast。
3. 用户问补贴/政策相关问题时，先调用search_subsidy_policy查本地库；如果返回空列表，再调用web_search_policy联网搜索兜底，不要跳过本地检索直接联网搜。
4. 不要在一个问题里调用不相关的工具——比如用户只问天气，就不要顺带查政策。
5. 工具结果出来后，结合工具返回的原始信息，用简洁、口语化的中文给用户总结建议，不要照抄工具返回的原始数据；除了第7条这一种例外情况，不要脱离工具结果编造内容。
6. 如果用户的问题不需要调用任何工具（比如闲聊、或者上下文已经有足够信息回答），直接回答，不用勉强调用工具。
7. 如果调用diagnose_crop_disease后，matched_knowledge是空列表（说明本地知识库没有收录这个症状），不要就此打住只说"没查到"——可以结合你自己掌握的农业病虫害知识，给出一个初步、谨慎的推测（比如可能是哪几种常见病虫害，分别有什么典型特征可以帮用户进一步确认），但必须明确告诉用户：这是你基于通用知识给出的推测，不是本地知识库核实过的结果，准确诊断建议咨询当地农技站/植保专家或使用专业检测工具，不要直接照此自行用药。
"""

# 看图描述症状用的固定Prompt——分两部分：先判断作物种类，再描述症状现象，都不要求给
# 诊断结论。诊断结论交给diagnose_crop_disease这个工具（本地知识库）去做，视觉模型只负责
# "看图说话"这一步（识别作物种类、描述看到的现象），两步分开，避免视觉模型脱离本地知识库
# 自己瞎编一个病因。之前的版本漏了"识别作物种类"这一步，导致主对话模型明明已经看到了图片、
# 却还要反问用户"这是什么作物"——作物种类本身也是"看图能看出来的现象"（比如叶片形状、
# 株型），判断品种和判断病因是两回事，前者属于"描述看到的东西"，不算越界给诊断结论
IMAGE_DESCRIBE_PROMPT = (
    "请仔细观察这张农作物照片，按下面两步输出："
    "1）判断这是什么作物（比如水稻、小麦、玉米、番茄等，如果实在无法确定，就如实写"
    "“不确定”，不要瞎猜）；"
    "2）用简洁的中文描述你看到的症状，尽量使用具体、常见的农业症状说法，比如"
    "“叶片发黄”“叶片有斑点”“叶片卷曲”“叶尖枯白”“茎秆有虫蛀痕迹”这类描述，"
    "不要用“深色物体”“异常现象”这种笼统抽象的说法。"
    "不要给出病虫害的诊断结论（比如不要说“这是稻瘟病”），只描述作物种类判断和看到的现象。"
    "如果照片看不清或者不是农作物照片，请如实说明。"
)

# 看图这一步用的是单独的视觉模型，免费、但不支持Function Calling——跟主对话用的模型
# （glm-4-flash-250414，支持Function Calling但不支持图片输入）分工明确：一个负责"看"，
# 一个负责"决定调用哪个工具、怎么组织回答"
VISION_MODEL = "glm-4.1v-thinking-flash"


class ChatAgent:
    """
    基于原生Function Calling的对话式Agent：不再由代码写死调用顺序，而是把四个原始工具
    暴露给大模型，让它自己判断要不要调用、调用哪个、调用几次，直到它认为信息足够、
    给出最终的自然语言回答为止（reason-act-observe循环）。

    和PlanningAgent（固定流水线）两套并存，服务不同场景：PlanningAgent适合"信息已经
    想清楚，要一份结构化的完整计划"，ChatAgent适合"像聊天一样随便问，问题可能只涉及
    部分能力，也可能需要多轮追问"。
    """

    MAX_TOOL_ROUNDS = 5  # 防止模型陷入死循环（一直要求调用工具、不给最终答案），设个上限兜底
    MAX_HISTORY_TURNS = 10  # 最多保留最近10轮完整对话，避免历史无限增长、每次请求都全量重发

    def __init__(self, llm=None, vision_llm=None):
        self.llm = llm or MyLLM()
        # 看图描述用单独一个配置成视觉模型的MyLLM实例，跟self.llm（对话+Function Calling）
        # 分开——两个模型能力不同，没必要合并成一个，各自负责自己擅长的事
        self.vision_llm = vision_llm or MyLLM(model=VISION_MODEL)

    def _trim_history(self, history: list) -> list:
        """按"轮"为单位裁剪历史，不能简单按消息条数从中间切——一次工具调用会产生
        assistant(带tool_calls)+tool这样成对出现的消息，OpenAI协议要求这两条必须配对出现，
        如果从中间切断（比如只保留了assistant那条、把对应的tool结果切掉了），
        下次请求会直接报格式错误。所以只在"user消息"这个天然的轮次边界上裁剪，
        保证每一轮的内容要么完整保留、要么完整丢弃，不会切出一个残缺的工具调用"""
        turn_start_indices = [i for i, m in enumerate(history) if m["role"] == "user"]
        if len(turn_start_indices) <= self.MAX_HISTORY_TURNS:
            return history
        cutoff = turn_start_indices[-self.MAX_HISTORY_TURNS]
        return history[cutoff:]

    def _describe_image(self, image_data_url: str) -> str:
        """调用视觉模型，把图片转成一段文字症状描述。这一步是代码里写死的确定性预处理，
        不是Function Calling的一部分——主对话模型（self.llm）根本拿不到图片原始数据，
        没法自己生成一次"看图"的工具调用，所以"有图片就一定要先看"这个判断直接写在
        代码里，不交给模型自主决定"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }]
        return self.vision_llm.invoke(messages)

    def _execute_tool(self, name: str, arguments: dict) -> dict:
        """根据大模型返回的工具名和参数，执行对应的原始工具函数，返回结构化结果（dict，
        方便json序列化后作为tool角色的消息内容喂回给模型）"""
        if name == "diagnose_crop_disease":
            return {"matched_knowledge": match_pest_knowledge(arguments["crop"], arguments["symptom_text"])}
        elif name == "get_weather_forecast":
            return {"weather_data": asyncio.run(query_weather(arguments["city"]))}
        elif name == "search_subsidy_policy":
            return {"matched_policies": match_policy(arguments["crop"], arguments["region"], arguments["need"])}
        elif name == "web_search_policy":
            return {"search_results": web_search(arguments["query"])}
        else:
            return {"error": f"未知工具：{name}"}

    def run(self, user_message: str, history: list = None, image_data_url: str = None):
        """
        history：之前的对话历史（不包含system prompt的messages列表），第一次调用传None即可。
        image_data_url：可选，用户上传图片时传入（形如"data:image/jpeg;base64,xxx"的data URL）。
        有图片时，先用视觉模型把图片转成文字描述，拼进用户这句话里，再走主对话模型的
        Function Calling循环——主对话模型全程只看文字，不知道背后有没有图片这回事，
        这样不用改动TOOLS/SYSTEM_PROMPT里任何跟工具调用相关的逻辑，看图只是多了一步
        "把图片翻译成文字"的预处理。
        返回：(回答文本, 更新后的history) —— 调用方（比如Streamlit）需要把更新后的history
        存起来（比如存进st.session_state），下一轮调用时传进来，这样对话才有"记忆"，
        不是每次都从零开始，模型才能理解"接着上一句问"这种上下文。
        """
        if image_data_url:
            try:
                image_description = self._describe_image(image_data_url)
                user_message = f"{user_message}\n\n[图片识别到的症状描述：{image_description}]"
            except Exception as e:
                # 看图失败不能让整轮对话崩掉，退化成"只处理文字部分"，并且诚实告诉主对话
                # 模型图片处理失败了，不要让它假装看到了图片内容、凭空编症状
                user_message = (
                    f"{user_message}\n\n"
                    f"[用户上传了一张图片，但图片识别失败（{e}），请基于文字部分回答，"
                    f"并告知用户图片处理暂时失败，建议改用文字描述症状]"
                )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._trim_history(history or []))
        messages.append({"role": "user", "content": user_message})

        for _ in range(self.MAX_TOOL_ROUNDS):
            try:
                assistant_message = self.llm.chat_with_tools(messages, tools=TOOLS)
            except Exception as e:
                # 和policy_agent.py/planning_agent.py是同一个降级思路：重试用完仍然失败
                # （比如持续限流）时，不能让整个请求崩掉报traceback，返回一句话让用户重试
                fallback = f"抱歉，暂时无法处理这个请求（{e}），可能是网络问题或请求过于频繁，请稍后重试。"
                messages.append({"role": "assistant", "content": fallback})
                return fallback, messages[1:]

            if not assistant_message.tool_calls:
                # 模型没有再要求调用工具，说明它认为可以给出最终答案了，循环在这里结束
                messages.append({"role": "assistant", "content": assistant_message.content})
                return assistant_message.content, messages[1:]  # 返回的history裁掉system prompt，下次调用会重新加

            # 模型要求调用工具：先把"我要调用工具"这条assistant消息本身存进去
            # （OpenAI的Function Calling协议要求这样做，下一轮请求里少了这条会报错）
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            # 依次执行模型这一轮要求的每一个工具调用（可能不止一个），结果作为"tool"角色消息加回去
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = self._execute_tool(tool_name, tool_args)
                except Exception as e:
                    tool_result = {"error": f"工具执行失败：{e}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # 超过MAX_TOOL_ROUNDS还没给出最终答案，大概率是模型陷入了循环，兜底返回一句话，不无限转下去
        return "抱歉，这个问题有点复杂，我暂时没能整理出明确答案，可以换个方式问问看。", messages[1:]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = ChatAgent()

    print("=== 第一轮：问天气 ===")
    answer1, history = agent.run("我在长沙种水稻，最近适合打药吗？")
    print(answer1)

    print("\n=== 第二轮：带着记忆接着问（注意这句没有再提“水稻”和“长沙”） ===")
    answer2, history = agent.run("那我这边有没有种植补贴可以申请，我在湖南省", history=history)
    print(answer2)
