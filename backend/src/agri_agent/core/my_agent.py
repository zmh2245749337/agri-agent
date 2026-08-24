class MyAgent:
    """自己实现的最小Agent类，参考SimpleAgent的设计思路：
    绑定一个LLM + 一段系统提示词(system_prompt)，
    run()方法接收用户输入，拼成对话格式发给LLM，返回回复文本"""

    def __init__(self, name: str, llm, system_prompt: str = ""):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})
        return self.llm.invoke(messages)