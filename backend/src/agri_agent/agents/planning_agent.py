# src/agri_agent/agents/planning_agent.py
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from agri_agent.core.my_llm import MyLLM
from agri_agent.core.my_agent import MyAgent
from agri_agent.core.subagent_task import SubAgentTask, run_local_subagent
from agri_agent.a2a.policy_client import PolicyA2AClient
from agri_agent.agents.diagnosis_agent import CropDiagnosisAgent
from agri_agent.agents.weather_agent import WeatherAgent


class PlanningAgent:
    """
    整个项目的"总调度"：并发调用 CropDiagnosisAgent、WeatherAgent、PolicySubsidyAgent
    三个互相没有数据依赖的子Agent，拿到各自独立生成的自然语言建议后，再用一次
    大模型调用把三段内容整合成一份有优先级、连贯的"农事行动计划"。

    这里说的"固定流水线"（跟ChatAgent的Function Calling动态路由相对）指的是
    "调用哪几个子Agent、调用条件"是代码写死的，不是模型自己判断的——不代表
    "三个子Agent必须排队顺序执行"。三者之间没有先后依赖，顺序执行只是白白
    浪费时间，所以用线程池并发跑，跟"固定流水线"这个设计定位并不矛盾。

    这里"调用"的对象是三个Agent，不是Tool——这是它和前面几个Agent的关键区别：
    前面几个Agent是"Tool的消费者"（拿Tool吐出的原始数据去生成一段回答），
    PlanningAgent是"Agent的消费者"（拿其他Agent已经生成好的自然语言建议做二次整合），
    在整个项目的分层里，它站在比其他三个Agent更高一层。

    政策子Agent可以通过官方a2a-sdk作为独立服务运行：PlanningAgent先读取远程
    Agent Card，再使用A2A JSON-RPC Message/Task协议委派任务并读取Artifact。
    远程服务不可用时自动回退到进程内PolicySubsidyAgent，诊断和天气则继续使用
    进程内并发，避免为展示协议而把所有模块过度服务化。
    """

    def __init__(self, llm=None, policy_a2a_url: str | None = None, policy_a2a_client=None):
        # PolicySubsidyAgent导入时会初始化BGE检索依赖；延迟到真正创建编排器时
        # 再加载，协议层和纯编排单测无需为一个不会执行的本地回退加载向量模型。
        from agri_agent.agents.policy_agent import PolicySubsidyAgent

        self.llm = llm or MyLLM()
        self.agent = MyAgent("PlanningAgent", self.llm)
        # 三个子Agent共享同一个MyLLM实例，不用各自重新初始化一遍client连接
        self.diagnosis_agent = CropDiagnosisAgent(llm=self.llm)
        self.weather_agent = WeatherAgent(llm=self.llm)
        self.policy_agent = PolicySubsidyAgent(llm=self.llm)
        if policy_a2a_client is not None:
            self.policy_a2a_client = policy_a2a_client
        else:
            remote_url = policy_a2a_url or os.getenv("POLICY_A2A_URL")
            self.policy_a2a_client = PolicyA2AClient(remote_url) if remote_url else None
        # 记录最近一次run()里三个子Agent的统一任务结果，其中policy.protocol
        # 会明确标注local、A2A/JSONRPC或local-fallback，便于定位远程调用情况。
        self.last_tasks = {}

    def _dispatch_policy_task(self, crop: str, region: str, need: str) -> SubAgentTask:
        input_text = f"作物={crop}；地区={region}；需求={need}"
        if self.policy_a2a_client is None:
            return run_local_subagent(
                "政策补贴建议",
                lambda: self.policy_agent.run(crop, region, need),
                input_text,
            )

        try:
            return self.policy_a2a_client.run(crop, region, need)
        except Exception as exc:
            print(f"[PlanningAgent] A2A政策服务不可用，回退本地执行：{exc}")
            task = run_local_subagent(
                "政策补贴建议",
                lambda: self.policy_agent.run(crop, region, need),
                input_text,
            )
            task.protocol = "local-fallback"
            task.fallback_reason = str(exc)
            return task

    def run(
        self,
        crop: str = None,
        city: str = None,
        region: str = None,
        symptom_text: str = None,
        need: str = "种植补贴",
        capabilities: list[str] = None,
    ) -> str:
        # 老的结构化/planning入口不传capabilities时，保持原行为：有症状就诊断，
        # 同时查询天气和政策。统一AgriGraph会显式传入本轮需要的能力，避免为了
        # "规划"这个名字无条件调用用户没有询问的模块。
        if capabilities is None:
            selected = {"weather", "policy"}
            if symptom_text:
                selected.add("diagnosis")
        else:
            selected = set(capabilities) & {"diagnosis", "weather", "policy"}
        # ---------- 第一步～第三步：诊断/天气/政策 三个子Agent并发执行 ----------
        # 这三步互相之间没有数据依赖（谁都不需要等另一个的结果才能开始），之前是顺序
        # 调用，总耗时=三段耗时相加；三个子Agent内部主要都是网络I/O等待（调大模型、
        # 查天气MCP、查政策库+向量检索+可能的联网搜索兜底），不是CPU密集型计算，
        # 用线程池并发跑更合适——I/O等待期间GIL会被释放，线程之间不会互相卡住，
        # 也不需要为了并发把整条调用链改成async（self.llm.invoke()目前是同步阻塞
        # 调用，weather_agent.run()内部还用asyncio.run()桥接了一次MCP的异步调用，
        # 真要全链路改成async会牵一发动全身，线程池是改动最小、风险最低的方案）。
        # 并发之后总耗时约等于三段里最慢的那一段，而不是三段相加。
        start_time = time.time()

        futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 没传症状描述就直接不提交这个任务，跟之前"跳过诊断步骤"是同一个效果，
            # 不会因为并发就多此一举地调用诊断Agent。每个future跑的都是
            # run_local_subagent把进程内成功/失败转换成统一任务结果；政策分支则由
            # _dispatch_policy_task决定走官方A2A远程服务还是本地回退。
            if "diagnosis" in selected and symptom_text:
                futures["diagnosis"] = executor.submit(
                    run_local_subagent,
                    "作物诊断建议",
                    lambda: self.diagnosis_agent.run(crop, symptom_text),
                    f"作物={crop}；症状={symptom_text}",
                )
            if "weather" in selected and city:
                futures["weather"] = executor.submit(
                    run_local_subagent,
                    "天气建议",
                    lambda: self.weather_agent.run(city),
                    f"城市={city}",
                )
            if "policy" in selected and crop and region:
                futures["policy"] = executor.submit(
                    self._dispatch_policy_task,
                    crop,
                    region,
                    need,
                )

            # 三个入口都会把异常转换成SubAgentTask，不会有单个子Agent异常从
            # future.result()继续向外扩散。
            tasks = {key: future.result() for key, future in futures.items()}

        self.last_tasks = tasks
        elapsed = time.time() - start_time
        print(f"[PlanningAgent] {len(tasks)}个子模块并发执行完成，耗时{elapsed:.1f}秒")

        # ---------- 按固定顺序拼装结果（诊断→天气→政策），不受并发完成顺序影响 ----------
        # 三个线程谁先跑完是不确定的，但用户看到的报告结构应该是稳定、可预期的，
        # 所以这里不是按"谁先完成就先放谁"拼，而是始终按诊断/天气/政策这个固定顺序
        # 从tasks字典里取值——并发只改变了"怎么算出来的"，不改变"最终呈现成什么样"。
        # task.result_text在失败/不存在时返回空字符串，效果等价于之前results.get(...)
        # 拿到None，下面的if判断不用跟着改
        sections = []
        if tasks.get("diagnosis") and tasks["diagnosis"].succeeded:
            sections.append(f"【作物诊断建议】\n{tasks['diagnosis'].result_text}")
        if tasks.get("weather") and tasks["weather"].succeeded:
            sections.append(f"【天气与农事建议】\n{tasks['weather'].result_text}")
        if tasks.get("policy") and tasks["policy"].succeeded:
            sections.append(f"【政策补贴建议】\n{tasks['policy'].result_text}")

        if not sections:
            return "很抱歉，本次没有子模块生成有效建议，请补充必要信息、检查网络连接或稍后重试。"

        # ---------- 第四步：整合成一份连贯的行动计划 ----------
        combined = "\n\n".join(sections)
        prompt = f"""你是一个农事顾问。下面是针对作物"{crop}"，由本轮实际需要的专业模块分别生成的建议，
这些建议是各自独立生成的，可能有一些重复或者角度分散。请把它们整合成一份连贯、有优先级的行动建议，
根据实际内容选择贴切的小标题，用简洁清晰的语言呈现。不要逐字重复原文，也不要脱离这些内容编造新信息。
只展示本轮确实有结果的部分；严禁创建空章节，严禁输出“暂无内容”“本模块无相关建议”之类的占位文字：

{combined}
"""
        print("正在整合成最终行动计划...")
        try:
            return self.agent.run(prompt)
        except Exception as e:
            return (
                f"（整合失败：{e}，可能是网络问题或请求超时，请稍后重试）\n\n"
                f"以下是各模块分别生成的建议，供参考：\n\n{combined}"
            )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = PlanningAgent()

    result = agent.run(
        crop="水稻",
        city="长沙",
        region="湖南省",
        symptom_text="最近发现叶子发黄",
        need="水稻种植补贴",
    )
    print("\n=== 最终行动计划 ===")
    print(result)
