# tests/test_myllm_retry.py
"""
验证MyLLM._create_with_retry()的限流自动重试逻辑：只对openai.RateLimitError重试，
固定延迟、最多重试max_retries次，重试次数用完仍失败就把异常抛出去交给上层降级逻辑。

用一个假的OpenAI客户端（可以指定"前N次调用失败，之后成功"）测试，不依赖真实网络/
真实智谱账号——这段逻辑本来就是为了应对第九节记录过的真实429限流问题加的，
但测试不应该依赖"真的触发一次限流"这种不可控的外部条件。

为了让重试测试跑得快，这里把retry_delay设成0（不用真的等3秒），
只验证"重试了几次、最后有没有抛异常"这些逻辑，不测真实的等待时长。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from openai import RateLimitError
from agri_agent.core.my_llm import MyLLM

# openai.RateLimitError继承自APIStatusError，构造函数要求传一个真实的httpx.Response
# （不能传None——内部会读response.headers/status_code这些属性），这里造一个最简单的
# 假HTTP请求/响应对象，只是为了满足构造函数签名，不会真的发起网络请求
_FAKE_REQUEST = httpx.Request("POST", "https://fake.example.com/chat/completions")


def _make_rate_limit_error():
    fake_response = httpx.Response(status_code=429, request=_FAKE_REQUEST)
    return RateLimitError("429 rate limited (fake)", response=fake_response, body=None)


class _FakeCompletions:
    """前fail_times次调用抛RateLimitError，之后返回一个假的成功响应"""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _make_rate_limit_error()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"成功回复（第{self.calls}次调用）"))]
        )


def _make_llm_with_fake_client(fail_times, max_retries=2):
    llm = MyLLM(model="fake-model", api_key="fake-key", base_url="http://fake", max_retries=max_retries, retry_delay=0)
    fake_completions = _FakeCompletions(fail_times)
    llm.client.chat.completions = fake_completions
    return llm, fake_completions


def test_retries_then_succeeds():
    """前1次失败，第2次成功——max_retries=2应该够用，不会把异常抛出去"""
    llm, fake_completions = _make_llm_with_fake_client(fail_times=1)
    result = llm.invoke([{"role": "user", "content": "你好"}])

    assert "成功回复" in result
    assert fake_completions.calls == 2  # 第1次失败+第2次成功，一共调用了2次
    print("测试通过：遇到限流重试后成功，不会误报失败")


def test_retries_exhausted_then_raises():
    """一直失败（次数超过max_retries能扛住的范围）：重试用完之后应该把异常抛出去，
    交给上层（Agent的try/except）做降级，而不是自己吞掉或者无限重试"""
    llm, fake_completions = _make_llm_with_fake_client(fail_times=99, max_retries=2)

    try:
        llm.invoke([{"role": "user", "content": "你好"}])
        assert False, "应该抛出RateLimitError，不应该正常返回"
    except RateLimitError:
        pass

    # max_retries=2意味着总共尝试3次（第1次 + 2次重试），第3次失败后不再重试
    assert fake_completions.calls == 3
    print("测试通过：重试次数用完后正确抛出异常，不会无限重试")


def test_first_try_succeeds_no_retry():
    """第一次就成功的情况，不应该触发任何多余的重试/等待"""
    llm, fake_completions = _make_llm_with_fake_client(fail_times=0)
    result = llm.invoke([{"role": "user", "content": "你好"}])

    assert "成功回复" in result
    assert fake_completions.calls == 1  # 只应该调用一次，没有多余重试
    print("测试通过：第一次就成功时不会触发多余的重试")


def test_non_rate_limit_error_is_not_retried():
    """非RateLimitError（比如鉴权错误、参数错误）不应该被重试——重试对这类错误
    没有意义，只会拖慢失败反馈的速度，这是'错误分类处理'设计的核心验证点"""
    llm = MyLLM(model="fake-model", api_key="fake-key", base_url="http://fake", max_retries=2, retry_delay=0)

    call_count = {"n": 0}

    def _always_raise_value_error(**kwargs):
        call_count["n"] += 1
        raise ValueError("模拟鉴权/参数错误（不是限流）")

    llm.client.chat.completions.create = _always_raise_value_error

    try:
        llm.invoke([{"role": "user", "content": "你好"}])
        assert False, "应该抛出ValueError"
    except ValueError:
        pass

    assert call_count["n"] == 1  # 非RateLimitError只应该尝试一次，不重试
    print("测试通过：非限流类异常不会被重试，第一次失败就直接抛出")


if __name__ == "__main__":
    test_retries_then_succeeds()
    test_retries_exhausted_then_raises()
    test_first_try_succeeds_no_retry()
    test_non_rate_limit_error_is_not_retried()
    print("\n全部测试通过")
