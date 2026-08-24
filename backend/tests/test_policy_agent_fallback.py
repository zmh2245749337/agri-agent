# tests/test_policy_agent_fallback.py
"""
验证PolicySubsidyAgent的联网兜底分支。

这段代码不依赖真实网络请求，也不调用真实大模型——用unittest.mock强制让
match_policy()返回空列表（模拟"本地库彻底没查到"的情况），检查代码是否
正确地转向调用web_search()，并把联网搜索结果正确地拼进了喂给大模型的
prompt里。

为什么要mock：加了地区硬过滤之后，本地政策库里那条"全国/不限"的农机购置
补贴几乎对任何真实查询都成立（它本身就适用所有地区所有作物），导致联网
兜底这条分支很难被一个真实的、合理的查询触发。与其硬凑一个不自然的测试
数据，不如直接mock掉match_policy的返回值，只验证"分支逻辑本身对不对"——
这是工程上验证低频/难触发代码路径的标准做法。
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.agents.policy_agent import PolicySubsidyAgent


def test_falls_back_to_web_search_when_local_empty():
    fake_web_results = [
        {
            "title": "测试用政策标题",
            "content": "测试用政策内容摘要",
            "link": "https://example.com/fake-policy",
            "publish_date": "2026-08-01",
        }
    ]

    with patch("agri_agent.agents.policy_agent.match_policy", return_value=[]) as mock_match, \
         patch("agri_agent.agents.policy_agent.web_search", return_value=fake_web_results) as mock_search, \
         patch.object(PolicySubsidyAgent, "__init__", lambda self: None):

        agent = PolicySubsidyAgent()
        # 把self.agent替换成一个假对象，run()方法直接原样返回收到的prompt，
        # 这样就不需要真的连大模型，还能直接断言prompt里有没有拼进联网搜索结果
        agent.agent = type("FakeAgent", (), {"run": lambda self, prompt: prompt})()

        result = agent.run("火龙果", "西藏自治区", "种植补贴")

        # 断言1：match_policy确实被调用过（说明本地检索这一步没被跳过）
        mock_match.assert_called_once()
        # 断言2：match_policy返回空列表之后，web_search确实被调用了（说明真的转去联网了）
        mock_search.assert_called_once()
        # 断言3：联网搜索的结果确实被拼进了最终喂给大模型的prompt里
        assert "测试用政策标题" in result
        assert "https://example.com/fake-policy" in result
        # 断言4：prompt里带上了"这是联网搜索来的、没人工核实"这层提示，
        # 不能让大模型误以为这是本地权威知识库的内容
        assert "联网搜索" in result

    print("测试通过：本地检索为空时，PolicySubsidyAgent正确转向联网搜索兜底分支")


def test_uses_local_when_matched():
    fake_local_results = [
        {
            "title": "测试用本地政策",
            "region": "广西壮族自治区",
            "crop": "甘蔗",
            "content": "本地政策内容",
            "source": "https://example.com/local-source",
            "match_score": 0.8,
            "match_reason": "地区匹配；作物匹配",
        }
    ]

    with patch("agri_agent.agents.policy_agent.match_policy", return_value=fake_local_results) as mock_match, \
         patch("agri_agent.agents.policy_agent.web_search") as mock_search, \
         patch.object(PolicySubsidyAgent, "__init__", lambda self: None):

        agent = PolicySubsidyAgent()
        agent.agent = type("FakeAgent", (), {"run": lambda self, prompt: prompt})()

        result = agent.run("甘蔗", "广西壮族自治区", "甘蔗补贴")

        mock_match.assert_called_once()
        # 关键断言：本地库有结果的时候，联网搜索完全不应该被调用——
        # 避免"本地库明明查到了，还多此一举去联网、多花一次调用成本"
        mock_search.assert_not_called()
        assert "测试用本地政策" in result
        assert "本地政策知识库" in result

    print("测试通过：本地检索有结果时，不会误触发联网搜索")


if __name__ == "__main__":
    test_falls_back_to_web_search_when_local_empty()
    test_uses_local_when_matched()
    print("\n全部测试通过")
