# frontend/app.py
import base64
import os
import uuid

import requests
import streamlit as st

# 后端地址通过环境变量配置，本地开发默认打localhost，部署到云端时改成后端服务的
# 公网地址即可，前端代码本身不用改——这也是前后端分离的一个直接好处
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="智慧农事助手 AgriAgent", page_icon="🌾", layout="centered")
st.title("🌾 智慧农事助手 AgriAgent")
st.caption("作物诊断 · 天气建议 · 政策补贴 · 一站式农事行动计划")

# 图片描述文字拼进用户消息时用的两个标记前缀（要跟agri_graph.py里的文案完全一致），
# 用来在重新渲染历史记录时把这段"内部标记文字"切掉，只给用户看他自己原本打的那句话——
# 这段标记是喂给大模型看的上下文，不是说给用户听的
_IMAGE_DESC_MARKER = "\n\n[图片识别到的症状描述："
_IMAGE_FAIL_MARKER = "\n\n[用户上传了一张图片，但图片识别失败"

# 工具调用轨迹面板用的中文友好标签——AgriGraph返回的history里完整记录了
# 每一轮的tool_calls和tool结果（工具名、参数、返回的原始数据），之前只是渲染时
# 把这些过滤掉了、只显示最终的文字回答。这里把它们重新展示出来（放进一个默认收起的
# 折叠面板里，不打扰正常对话阅读），能让人直接看到"模型这一轮到底调用了哪些工具、
# 传了什么参数、拿到了什么原始结果"，对讲清楚Function Calling的实际工作过程很有用
_TOOL_LABELS = {
    "diagnose_crop_disease": "🩺 查询病虫害知识库",
    "get_weather_forecast": "☀️ 查询天气预报",
    "search_subsidy_policy": "📋 查询本地政策库",
    "web_search_policy": "🌐 联网搜索政策",
}

_CONTEXT_LABELS = {
    "crop": "作物",
    "city": "城市",
    "region": "地区",
    "symptom": "症状",
    "need": "需求",
    "growth_stage": "种植阶段",
    "has_symptom": "是否有症状",
    "active_goal": "当前目标",
    "pending_slot": "待确认",
}


def _display_text(raw_content: str) -> str:
    """把user消息里拼接的图片相关内部标记切掉，只保留用户自己输入的部分用于展示"""
    text = raw_content.split(_IMAGE_DESC_MARKER)[0]
    text = text.split(_IMAGE_FAIL_MARKER)[0]
    return text


def _render_tool_trace(tool_steps: list) -> None:
    """在一个默认收起的折叠面板里，展示这一轮对话背后实际发生的工具调用过程"""
    with st.expander(f"🔧 查看工具调用过程（共{len(tool_steps)}次）"):
        for i, step in enumerate(tool_steps, 1):
            label = _TOOL_LABELS.get(step["name"], step["name"])
            st.markdown(f"**{i}. {label}** `{step['name']}`")
            st.caption("调用参数：")
            st.code(step["arguments"], language="json")
            if step["result"] is not None:
                st.caption("返回结果：")
                st.code(step["result"], language="json")
            if i < len(tool_steps):
                st.divider()


def call_backend(path: str, payload: dict, timeout: int = 120) -> dict:
    """统一封装对后端的POST请求，失败时给出清晰的错误提示，返回完整的响应dict——
    不同接口返回的字段不完全一样（/chat比其他接口多一个history字段），
    统一返回dict由调用方自己取需要的字段，比只返回result字符串更灵活"""
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"result": f"⚠️ 连不上后端服务（{BACKEND_URL}），请确认后端已启动：\nuvicorn agri_agent.api.main:app --reload"}
    except requests.exceptions.Timeout:
        return {"result": "⚠️ 请求超时，大模型响应较慢或网络不稳定，可以稍后重试"}
    except requests.exceptions.HTTPError as e:
        return {"result": f"⚠️ 后端返回错误：{e}"}


# ========== 聊天是唯一用户入口 ==========
# 作物、地区、症状等结构化字段都由AgriGraph从自然语言中提取；信息不足时，
# 助手会在对话里追问并把答案保存到当前会话，不再要求用户填写独立表单。
st.caption("直接说出你的问题，比如“我在长沙种水稻，最近适合打药吗”。信息不够时助手会接着询问，也可以附一张作物照片。")

if "chat_history" not in st.session_state:
    # 存的是AgriGraphAgent返回的messages列表格式（不含system prompt），
    # 每轮对话结束后原样存回session_state，下一轮调用时原样传给后端——
    # 这份"记忆"只存在于这一次浏览器会话里，刷新页面/关掉标签页就没了
    st.session_state.chat_history = []
if "chat_thread_id" not in st.session_state:
    # AgriGraph用thread_id在MemorySaver中隔离并续接会话。
    st.session_state.chat_thread_id = uuid.uuid4().hex
if "chat_context" not in st.session_state:
    st.session_state.chat_context = {}
if "chat_image_map" not in st.session_state:
    # 图片是前端本地的展示需求，后端history里只有文字（图片已经被视觉模型翻译成文字了，
    # 原始图片数据没必要也不应该跟着对话历史来回传）。所以单独在前端维护一个
    # "user消息文本 -> 图片字节"的映射，只用来在重新渲染历史气泡时把缩略图带出来，
    # 用消息内容本身（而不是列表下标）当key，是因为后端history可能会因为超过10轮被裁剪，
    # 下标会跟着变，但每条消息的文本内容不会变
    st.session_state.chat_image_map = {}

if st.session_state.chat_context:
    confirmed = []
    for key, value in st.session_state.chat_context.items():
        if value is None or value == "":
            continue
        if key == "has_symptom":
            value = "有" if value else "无"
        elif key == "pending_slot":
            value = _CONTEXT_LABELS.get(value, value)
        confirmed.append(f"{_CONTEXT_LABELS.get(key, key)}：{value}")
    if confirmed:
        st.caption("已确认信息 · " + " | ".join(confirmed))

if st.button("🗑️ 清空对话", key="btn_clear_chat"):
    st.session_state.chat_history = []
    st.session_state.chat_image_map = {}
    st.session_state.chat_thread_id = uuid.uuid4().hex
    st.session_state.chat_context = {}
    st.rerun()
# role=user和role=assistant(有content)的消息渲染成聊天气泡；role=assistant但
# content为空（那是"我要调用工具"的中间消息）、role=tool（工具执行结果）不再是
# 直接跳过不展示——而是先攒起来，等遇到这一轮真正的最终回答时，在它上方放一个默认
# 收起的折叠面板展示出来。用pending_tool_steps这个缓冲区攒（不是复杂的下标遍历）：
# 每遇到一条带tool_calls的assistant消息就记一笔"调用了什么、传了什么参数"，
# 每遇到一条tool消息就把对应call_id的结果补上，攒到下一条有内容的assistant消息
# 出现时统一展示、然后清空——这样天然支持一轮里连续调用好几次工具的情况
pending_tool_steps = []
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(_display_text(msg["content"]))
            image_bytes = st.session_state.chat_image_map.get(msg["content"])
            if image_bytes:
                st.image(image_bytes, width=200)
    elif msg["role"] == "assistant" and msg.get("tool_calls"):
        # 极少数模型在决定调用工具的这一轮，content和tool_calls可能同时非空
        # （项目里实际用的glm-4-flash-250414在这种情况下content一直是None，
        # 但这里不依赖这个假设，防止真遇到这种情况时文字内容被悄悄丢掉）
        if msg.get("content"):
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        for tc in msg["tool_calls"]:
            pending_tool_steps.append({
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
                "result": None,
            })
    elif msg["role"] == "tool":
        for step in pending_tool_steps:
            if step["call_id"] == msg.get("tool_call_id"):
                step["result"] = msg["content"]
    elif msg["role"] == "assistant" and msg.get("content"):
        if pending_tool_steps:
            _render_tool_trace(pending_tool_steps)
            pending_tool_steps = []
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# 图片上传用st.chat_input自带的accept_file，不再单独摆一个st.file_uploader大框在
# 输入框上方——之前那种写法会常驻显示一个"Upload / 200MB per file • JPG, PNG, WEBP"
# 的拖拽框，字面是英文（Streamlit内置组件的文案不支持改成中文），常驻占地方也显别扭。
# accept_file这个参数会直接把"添加附件"做成输入框内的一个小图标（类似ChatGPT那种回形针
# 按钮），不点开的时候完全不占版面，点开才会弹出小的文件选择区——同样是Streamlit自带
# 组件、里面的文案同样是英文改不了，但只在主动点开时才会看到，不再是一直杵在页面上
prompt = st.chat_input(
    "问点什么吧，也可以点左边的回形针附一张照片...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

if prompt and (prompt.text or prompt.files):
    # 只传了图片没打字的情况也允许发送，给个默认文案，避免消息气泡空白
    user_input = prompt.text.strip() if prompt.text else "帮我看看这张照片"

    image_data_url = None
    image_bytes = None
    if prompt.files:
        uploaded_file = prompt.files[0]  # 目前只处理第一张，多图场景不是这次要解决的需求
        image_bytes = uploaded_file.getvalue()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = uploaded_file.type or "image/jpeg"
        image_data_url = f"data:{mime};base64,{b64}"

    with st.chat_message("user"):
        st.markdown(user_input)
        if image_bytes:
            st.image(image_bytes, width=200)

    old_history = st.session_state.chat_history
    spinner_text = (
        "看图识别症状 + 思考中（可能需要调用一到多个工具，稍等一下）..."
        if image_data_url
        else "思考中（可能需要调用一到多个工具，稍等一下）..."
    )
    with st.spinner(spinner_text):
        resp = call_backend(
            "/chat",
            {
                "message": user_input,
                "history": old_history,
                "image_data_url": image_data_url,
                "thread_id": st.session_state.chat_thread_id,
                "mode": "auto",
                "context": st.session_state.chat_context,
            },
            timeout=120,
        )

    new_history = resp.get("history", old_history)
    st.session_state.chat_thread_id = resp.get("thread_id") or st.session_state.chat_thread_id
    st.session_state.chat_context = resp.get("context") or st.session_state.chat_context
    if image_bytes and len(new_history) > len(old_history):
        # 新增的第一条消息就是这一轮后端实际存进history的user消息（内容 = 用户原话 +
        # 视觉模型拼接的症状描述标记），用它的完整文本当key存图，保证后面重新渲染这条
        # 历史记录时能精确匹配到对应的图片
        injected_user_msg = new_history[len(old_history)]
        if injected_user_msg.get("role") == "user":
            st.session_state.chat_image_map[injected_user_msg["content"]] = image_bytes
    st.session_state.chat_history = new_history

    with st.chat_message("assistant"):
        st.markdown(resp["result"])

    st.rerun()
