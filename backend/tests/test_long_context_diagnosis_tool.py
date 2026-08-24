# tests/test_long_context_diagnosis_tool.py
"""
long_context_diagnosis_tool.py的单元测试。这个模块的核心逻辑（JSON数组容错
解析、index越界过滤、prompt拼装）都是纯Python逻辑，不依赖rapidfuzz这类
第三方库，也不需要真实调用大模型——用一个假的llm对象（只要有个invoke()方法、
返回预设文本）就能完整测试，这跟test_myllm_retry.py用真实httpx构造异常对象、
但不用真实网络请求是同一个测试思路。

真正"长上下文推理准不准"这件事，不是这份文件要回答的问题——那是
eval/diagnosis_retrieval_eval.py要跑真实LLM才能量化出来的结论，这份文件
只保证"不管大模型回复的格式多不规范，这段解析代码都不会崩、不会返回错误结果"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.tools.long_context_diagnosis_tool import (
    long_context_diagnose,
    _parse_indices,
    PEST_KNOWLEDGE_BASE,
)


class _FakeLLM:
    """假的LLM，只有一个invoke()方法，返回预先设好的文本，
    不会真的发网络请求"""

    def __init__(self, canned_response: str):
        self.canned_response = canned_response
        self.last_messages = None

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        return self.canned_response


def test_parse_indices_handles_clean_json_array():
    assert _parse_indices("[2, 5]") == [2, 5]
    print("测试通过：干净的JSON数组能正确解析")


def test_parse_indices_handles_empty_array():
    assert _parse_indices("[]") == []
    print("测试通过：空数组正确解析为空列表")


def test_parse_indices_extracts_array_from_surrounding_text():
    # 模型偶尔不听话，会在数组前后加解释文字，正则应该能从中间抠出数组来
    raw = "根据症状描述，匹配的记录编号是[3, 7]，希望对您有帮助"
    assert _parse_indices(raw) == [3, 7]
    print("测试通过：数组外面包着解释文字时，仍能正确抠出JSON数组")


def test_parse_indices_extracts_array_from_code_fence():
    raw = "```json\n[1, 4]\n```"
    assert _parse_indices(raw) == [1, 4]
    print("测试通过：数组被markdown代码块围栏包裹时，仍能正确抠出")


def test_parse_indices_returns_empty_when_no_array_present():
    assert _parse_indices("完全不相关的文字，没有任何数组") == []
    assert _parse_indices("") == []
    print("测试通过：回复里压根没有数组时，安全返回空列表而不是报错")


def test_parse_indices_tolerates_negative_numbers_without_crashing():
    # 负数不是prompt里要求的合法index，但正则本身不该因为看到"-"就直接
    # 解析失败——数组能不能被抠出来，和数组里的值合不合法，是两件独立的事
    assert _parse_indices("[0, 9999, -1]") == [0, 9999, -1]
    print("测试通过：数组里带负数时，正则解析本身不会失败（合法性过滤是下一步的事）")


def test_long_context_diagnose_maps_indices_back_to_knowledge_entries():
    fake_llm = _FakeLLM("[0]")
    result = long_context_diagnose("水稻", "我家水稻叶子有点黄", llm=fake_llm)

    assert result == [PEST_KNOWLEDGE_BASE[0]]
    # prompt里应该带上作物和症状描述，方便核实拼装逻辑没写错
    prompt_text = fake_llm.last_messages[0]["content"]
    assert "水稻" in prompt_text and "我家水稻叶子有点黄" in prompt_text
    print("测试通过：模型返回[0]时，正确映射回知识库第0条记录，prompt里正确带上了作物和症状")


def test_long_context_diagnose_filters_out_of_bounds_and_negative_indices():
    # 越界（超出知识库长度）和非法（负数）的index，不该让程序崩溃，
    # 也不该被当成"某条真实记录"混进结果里
    fake_llm = _FakeLLM("[0, 9999, -1]")
    result = long_context_diagnose("水稻", "随便测试", llm=fake_llm)

    assert result == [PEST_KNOWLEDGE_BASE[0]]
    print("测试通过：越界/负数index被安全过滤，只保留真实存在的记录")


def test_long_context_diagnose_returns_empty_list_when_model_finds_nothing():
    fake_llm = _FakeLLM("[]")
    result = long_context_diagnose("水稻", "随便测试", llm=fake_llm)

    assert result == []
    print("测试通过：模型判断没有匹配记录时，正确返回空列表")


def test_long_context_diagnose_returns_empty_list_when_model_reply_unparseable():
    # 模型完全不按格式回复的极端情况，不该抛异常导致整个诊断流程中断——
    # 这是长上下文这条技术路线相比RapidFuzz多出来的一类新风险（依赖模型
    # 输出格式的稳定性），必须有兜底
    fake_llm = _FakeLLM("抱歉，我无法理解这个请求")
    result = long_context_diagnose("水稻", "随便测试", llm=fake_llm)

    assert result == []
    print("测试通过：模型回复完全不符合格式要求时，不会抛异常，安全返回空列表")


if __name__ == "__main__":
    test_parse_indices_handles_clean_json_array()
    test_parse_indices_handles_empty_array()
    test_parse_indices_extracts_array_from_surrounding_text()
    test_parse_indices_extracts_array_from_code_fence()
    test_parse_indices_returns_empty_when_no_array_present()
    test_parse_indices_tolerates_negative_numbers_without_crashing()
    test_long_context_diagnose_maps_indices_back_to_knowledge_entries()
    test_long_context_diagnose_filters_out_of_bounds_and_negative_indices()
    test_long_context_diagnose_returns_empty_list_when_model_finds_nothing()
    test_long_context_diagnose_returns_empty_list_when_model_reply_unparseable()
    print("\n全部测试通过")
