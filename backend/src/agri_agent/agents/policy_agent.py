# src/agri_agent/agents/policy_agent.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from agri_agent.core.my_llm import MyLLM
from agri_agent.core.my_agent import MyAgent
from agri_agent.tools.policy_match_tool import match_policy
from agri_agent.tools.web_search_tool import web_search


class PolicySubsidyAgent:
    def __init__(self, llm=None):
        self.llm = llm or MyLLM()
        self.agent = MyAgent("PolicySubsidyAgent", self.llm)

    def run(self, crop: str, region: str, need: str) -> str:
        matched = match_policy(crop, region, need)

        if matched:
            # 本地库有相关度足够高的结果，走本地知识库这条路——优点是可解释
            # （能说清楚为什么匹配上）、零延迟、零调用成本，优先用这条路
            lines = []
            for p in matched:
                lines.append(
                    f"《{p['title']}》（{p['region']}，适用作物：{p['crop']}）\n"
                    f"内容：{p['content']}\n"
                    f"来源：{p['source']}\n"
                    f"匹配依据：{p['match_reason']}（相关度{p['match_score']}）"
                )
            policy_text = "\n\n".join(lines)
            source_note = "以下内容来自本地政策知识库（人工整理核实过的真实政策条目）："
        else:
            # 本地库没查到够格的结果——可能是本地库压根没收录这个地区/作物，
            # 也可能是MIN_SCORE把所有候选都过滤掉了。这种情况不再是简单地告诉
            # 用户"查不到"，而是兜底转向联网实时搜索，尽量还是给出有效信息
            web_results = web_search(f"{region} {crop} {need} 2026年 政策")
            if not web_results:
                policy_text = "本地政策库和联网搜索均未查到相关信息，请如实告知用户暂时查不到，不要编造内容。"
                source_note = ""
            else:
                lines = []
                for r in web_results:
                    date_part = f"\n发布日期：{r['publish_date']}" if r["publish_date"] else ""
                    lines.append(f"《{r['title']}》\n内容：{r['content']}\n链接：{r['link']}{date_part}")
                policy_text = "\n\n".join(lines)
                source_note = (
                    "本地政策库没有查到相关度足够高的记录，以下内容来自实时联网搜索"
                    "（未经人工核实，请提醒用户以官方渠道信息为准）："
                )

        prompt = f"""你是一个农业补贴政策顾问。用户种植的作物是"{crop}"，所在地区是"{region}"，想咨询的问题是"{need}"。

{source_note}

{policy_text}

请结合用户的具体情况，用简洁清晰的语言给出建议（说明能申请哪些补贴、大致标准、以及信息来源方便核实），
不要照抄原文，也不要脱离上面的材料编造内容。如果材料来自联网搜索，请提醒用户以官方渠道信息为准。
"""
        print("正在请求大模型生成回答...")
        try:
            return self.agent.run(prompt)
        except Exception as e:
            # invoke()现在设了60秒超时，超时或者其他网络问题会在这里被捕获——
            # 与其让调用方看到一整段原始traceback，不如至少把检索/搜索到的原始
            # 信息返回回去，好歹比"程序直接崩掉、什么都拿不到"要有用
            return (
                f"（大模型生成回答失败：{e}，可能是网络问题或请求超时，请稍后重试）\n\n"
                f"以下是检索到的原始信息，供参考：\n\n{policy_text}"
            )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = PolicySubsidyAgent()

    print("=== 测试1：本地库精确命中（甘蔗/广西壮族自治区） ===")
    print(agent.run("甘蔗", "广西壮族自治区", "甘蔗机械化补贴"))

    # 加了地区硬过滤之后，这组预期只会命中"全国/不限"的农机购置补贴这一条
    # （广西甘蔗那条不该再出现了），不会真正触发联网兜底——因为这条通用政策
    # 本身确实适用于任何作物/地区，本地给出答案是对的，不算兜底没生效。
    # 联网兜底分支的正确性由 tests/test_policy_agent_fallback.py 用mock验证，
    # 不依赖凑一个"本地库刚好完全空手"的真实查询（这在当前语料库下很难凑出来）
    print("\n=== 测试2：本地库只命中通用政策，不该再有广西甘蔗误伤（咖啡/云南省） ===")
    print(agent.run("咖啡", "云南省", "咖啡种植补贴"))
