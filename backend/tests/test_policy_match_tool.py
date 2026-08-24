# tests/test_policy_match_tool.py
"""
验证policy_match_tool.match_policy()的两阶段检索逻辑——这是整个项目里技术含量
最高的一块（BM25粗筛+地区硬过滤+BGE向量精排+规则加权），但之前只有policy_match_tool.py
自己的__main__手动打印、以及policy_agent.py测试里被mock掉的间接验证，缺一份能直接
断言"检索结果对不对"的自动化测试，这份文件补上这个缺口。

跟test_pest_knowledge_tool.py一样，这里不mock任何东西，直接用真实的jieba分词+
BM25Okapi+BGE-small-zh-v1.5模型跑，因为要测的就是"真实检索算法+真实政策数据"
组合起来的排序/过滤行为对不对。

注意：这份测试需要sentence-transformers真实可用（第一次运行会从HuggingFace下载
BGE模型，几百MB，需要联网），运行会比其他测试慢一些（模型加载+embedding计算），
这是正常现象，不是卡住了。

下面的断言刻意没有去断言具体的相似度分数（比如"一定是0.774"）——具体分数依赖
BGE模型版本、数据预处理细节，没必要测到这么精确，容易变成"模型换了小版本测试就
莫名其妙挂了"这种脆弱测试。真正该固定下来的是"结构性、可解释的行为"：排序对不对、
地区硬过滤有没有生效（这是笔记第九节真实踩坑修复过的地方）、分数下限有没有生效、
返回条数有没有超限——这些是代码逻辑本身的正确性，不随模型版本细节而改变。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.tools.policy_match_tool import match_policy, MIN_SCORE, TOP_K_FINAL


def test_region_and_crop_exact_match_ranks_first():
    """甘蔗/广西壮族自治区查询：地区+作物都精确匹配的"广西糖料蔗"补助，
    应该排在结果第一位——这是最基本的检索正确性验证"""
    results = match_policy("甘蔗", "广西壮族自治区", "甘蔗机械化补贴")
    assert results, "应该至少检索到一条结果"
    assert "糖料蔗" in results[0]["title"] or "广西" in results[0]["title"]
    assert results[0]["match_score"] >= MIN_SCORE
    print("测试通过：地区+作物精确匹配的政策排在结果第一位")


def test_region_hard_filter_excludes_unrelated_region():
    """笔记第九节记录过的真实踩坑回归测试：'咖啡/云南省'这组查询，加地区硬过滤
    之前会被"广西糖料蔗"这条无关政策误伤（字面/语义背景噪声凑巧压线通过），
    加了硬过滤之后不应该再出现——这是这份测试里最重要的一条断言"""
    results = match_policy("咖啡", "云南省", "咖啡种植补贴")
    titles = [r["title"] for r in results]
    regions = [r["region"] for r in results]
    assert not any("广西" in t for t in titles)
    assert not any("广西" in r for r in regions)
    print("测试通过：地区不匹配的政策（广西糖料蔗）不会误伤到云南的查询结果里")


def test_universal_policy_reachable_from_any_region():
    """'全国/不限'类政策（比如农机购置补贴）应该对任何地区的查询都有资格进入候选，
    不应该被地区硬过滤误伤——地区硬过滤挡的是"地区对不上且不是全国通用"的候选，
    不应该连'全国'本身也挡在外面。用一个语料库里没有专门政策的冷门地区测试"""
    results = match_policy("苹果", "西藏自治区", "农机购置补贴")
    titles = [r["title"] for r in results]
    assert any("农机购置" in t for t in titles), "全国通用的农机购置补贴应该能被任何地区查到"
    print("测试通过：全国通用政策不受地区硬过滤影响，任何地区都能查到")


def test_all_returned_results_meet_min_score():
    """不管查询是什么，返回结果里不应该出现低于MIN_SCORE的弱相关结果——
    宁可少给结果，也不能为了凑够TOP_K_FINAL硬塞不相关的政策（笔记第九节第1个坑）"""
    for crop, region, need in [
        ("水稻", "湖南省", "种植补贴"),
        ("大豆", "黑龙江省", "种植补贴"),
        ("茶叶", "福建省", "种植补贴"),
    ]:
        results = match_policy(crop, region, need)
        for r in results:
            assert r["match_score"] >= MIN_SCORE, f"发现低于MIN_SCORE的结果混进了返回列表：{r}"
    print("测试通过：所有查询的返回结果都不低于MIN_SCORE，没有强行凑数")


def test_results_never_exceed_top_k_final():
    """不管候选池里有多少条通过了筛选，最终返回条数都不应该超过TOP_K_FINAL"""
    results = match_policy("水稻", "全国", "种植补贴")
    assert len(results) <= TOP_K_FINAL
    print(f"测试通过：返回结果数量没有超过TOP_K_FINAL（{TOP_K_FINAL}）")


def test_match_reason_is_explainable():
    """每条返回结果都应该带有match_reason字段，且不是空字符串——
    政策模块的一个设计重点是"可解释性"（能说清楚为什么匹配上了），这条测试
    确保这个字段确实被填充了，不是被漏掉的空壳字段"""
    results = match_policy("甘蔗", "广西壮族自治区", "甘蔗机械化补贴")
    assert results
    for r in results:
        assert r.get("match_reason"), f"结果缺少match_reason，可解释性字段没填：{r}"
    print("测试通过：每条结果都带有非空的match_reason，可解释性字段生效")


if __name__ == "__main__":
    test_region_and_crop_exact_match_ranks_first()
    test_region_hard_filter_excludes_unrelated_region()
    test_universal_policy_reachable_from_any_region()
    test_all_returned_results_meet_min_score()
    test_results_never_exceed_top_k_final()
    test_match_reason_is_explainable()
    print("\n全部测试通过")
