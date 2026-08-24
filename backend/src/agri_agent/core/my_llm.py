# src/agri_agent/core/my_llm.py
import os
import time

from openai import OpenAI, RateLimitError


class MyLLM:
    """自己实现的LLM封装，参考HelloAgentsLLM的设计思路，
    但只支持OpenAI兼容接口（智谱GLM就是这种），去掉了多厂商适配层"""

    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        temperature=0.7,
        timeout=60,
        max_retries=2,
        retry_delay=3,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.model or not self.api_key or not self.base_url:
            raise ValueError("必须提供 model / api_key / base_url（或对应的环境变量）")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _create_with_retry(self, **kwargs):
        """封装限流重试逻辑的底层调用，invoke()和chat_with_tools()共用一份重试逻辑，
        不用在两个方法里各写一遍。只对RateLimitError重试，其他异常直接抛出去"""
        timeout = kwargs.pop("timeout", self.timeout)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(timeout=timeout, **kwargs)
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"[MyLLM] 遇到限流（第{attempt + 1}次尝试失败），{self.retry_delay}秒后自动重试...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[MyLLM] 重试{self.max_retries}次后仍然限流，放弃重试")

        raise last_error

    def invoke(self, messages: list, **kwargs) -> str:
        """非流式调用，传入messages列表，返回模型回复的文本内容。
        给现有四个Agent（Weather/Diagnosis/Policy/Planning）用，接口和之前完全一样，没有破坏性改动"""
        temperature = kwargs.pop("temperature", self.temperature)
        response = self._create_with_retry(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    def chat_with_tools(self, messages: list, tools: list, tool_choice="auto", **kwargs):
        """支持Function Calling的调用。和invoke()的关键区别：
        返回的是完整的message对象，不是只返回文本——因为调用方（ChatAgent）需要检查
        message.tool_calls来判断模型这一轮是"要求调用工具"还是"已经给出最终答案"，
        只返回纯文本的话这个信息就丢了。给新的ChatAgent（Function Calling循环）用"""
        temperature = kwargs.pop("temperature", self.temperature)
        response = self._create_with_retry(
            model=self.model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        return response.choices[0].message
