# tests/test_pest_knowledge_tool.py
"""
验证pest_knowledge_tool.match_pest_knowledge()的模糊匹配逻辑本身对不对。

跟test_planning_agent.py/test_policy_agent_fallback.py不一样，这份测试没有mock掉
match_pest_knowledge——RapidFuzz是个轻量库（不像sentence_transformers要下载几百MB的
模型），直接用真实依赖跑，测的就是"真实模糊匹配算法+真实知识库数据"组合起来的行为，
不是在测"我的代码逻辑走向对不对"这种更抽象的东西。

这几个用例不是随便写的，是开发笔记第五节里两轮真实踩坑纠偏（插字改写匹配不上、
关键词碰撞误判）的回归验证——如果以后改动了FUZZY_MATCH_THRESHOLD或者SCORE_MARGIN
这些参数，这份测试能立刻告诉你有没有破坏掉已经验证过的行为。

运行方式：需要backend/requirements.txt里的rapidfuzz已经装好（cd到agri-agent目录后
`pip install -r backend/requirements.txt`），然后 `python backend/tests/test_pest_knowledge_tool.py`
或者用pytest跑。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge


def test_exact_keyword_match_returns_correct_entry():
    """知识库原文写法：应该毫无悬念地匹配上"""
    results = match_pest_knowledge("水稻", "最近发现叶子发黄，还有点卷")
    keywords_hit = [kw for r in results for kw in r["keywords"]]
    assert "叶子发黄" in keywords_hit
    print("测试通过：知识库原文写法能准确匹配")


def test_synonym_rewrite_matches_without_false_collision():
    """换字不改意思（'叶片发黄'↔'叶子发黄'）：应该能捞回来，
    且不应该误伤字面相似但含义不同的'叶片发黏'（通用类目，蚜虫蜜露）——
    这是笔记第五节第5/6步实测踩坑后加SCORE_MARGIN排序过滤修复的那个问题，
    这条测试是那次修复的回归验证"""
    results = match_pest_knowledge("水稻", "我家水稻叶片发黄了")

    causes = [c for r in results for c in r["causes"]]
    assert any("缺氮肥" in c or "纹枯病" in c or "排水不良" in c for c in causes)

    # 关键断言：不能混进"叶片发黏"（蚜虫蜜露）那条无关记录
    all_keywords = [kw for r in results for kw in r["keywords"]]
    assert "叶片发黏" not in all_keywords
    print("测试通过：换字说法能匹配到正确记录，且没有被字面相似的无关记录误伤")


def test_insertion_rewrite_is_a_known_accepted_limitation():
    """插字改写（'叶子有点黄'相对'叶子发黄'插入了新字）：这是编辑距离类模糊匹配的
    固有边界（插入字符会把后面字符的对齐位置全部错开，字面相似度掉得比换字更多），
    笔记第五节记录过这是"已知的、接受的局限，不是bug"。这条测试不是要断言"必须匹配
    失败"，而是诚实地把这个边界用代码固定下来——如果以后升级到向量语义检索解决了
    这个问题，这条测试会失败，到时候应该更新测试断言，而不是被意外的行为变化搞懵"""
    results = match_pest_knowledge("水稻", "我家水稻叶子有点黄")
    assert results == [], (
        "如果这条断言开始失败，说明匹配算法的行为变了（可能是升级到了语义检索），"
        "去开发笔记第五节确认一下是不是预期之内的改进，再更新这条测试"
    )
    print("测试通过：插字改写这个已知边界的行为符合预期（模糊匹配确实捞不回来）")


def test_generic_category_matches_regardless_of_crop():
    """'通用'类目的记录（不针对特定作物的症状，比如白粉病）应该对任何作物的
    查询都生效，不应该被作物过滤挡在外面"""
    results = match_pest_knowledge("番茄", "叶片有白色粉末，像是撒了层面粉")
    keywords_hit = [kw for r in results for kw in r["keywords"]]
    assert "叶片有白色粉末" in keywords_hit
    print("测试通过：通用类目记录不受作物过滤影响，任何作物都能命中")


def test_crop_filter_excludes_other_crops_specific_entries():
    """症状文本用的是小麦专属记录的关键词，但查询的crop是水稻——
    不应该跨作物匹配到小麦专属的那条记录（作物过滤要生效）"""
    wheat_specific_results = match_pest_knowledge("小麦", "秧苗徒长细高")
    wheat_keywords = [kw for r in wheat_specific_results for kw in r["keywords"]]

    # 先确认这个关键词确实是某个具体作物专属的（不是"通用"），这条测试才有意义
    rice_only_results = match_pest_knowledge("水稻", "秧苗徒长细高")
    rice_keywords = [kw for r in rice_only_results for kw in r["keywords"]]
    assert "秧苗徒长细高" in rice_keywords  # 这条记录本身是水稻专属的（恶苗病）

    wheat_wrong_crop_results = match_pest_knowledge("小麦", "水稻秧苗徒长细高瘦弱")
    wheat_wrong_keywords = [kw for r in wheat_wrong_crop_results for kw in r["keywords"]]
    assert "秧苗徒长细高" not in wheat_wrong_keywords
    print("测试通过：作物过滤生效，不会跨作物匹配到其他作物专属的记录")


def test_no_match_returns_empty_list_not_error():
    """完全不沾边的症状描述，应该老老实实返回空列表，不应该报错，
    也不应该为了凑结果硬塞一个不相关的记录"""
    results = match_pest_knowledge("水稻", "完全无关的一段话，跟任何症状都不沾边xyz123")
    assert results == []
    print("测试通过：查不到时返回空列表，不报错也不硬凑结果")


if __name__ == "__main__":
    test_exact_keyword_match_returns_correct_entry()
    test_synonym_rewrite_matches_without_false_collision()
    test_insertion_rewrite_is_a_known_accepted_limitation()
    test_generic_category_matches_regardless_of_crop()
    test_crop_filter_excludes_other_crops_specific_entries()
    test_no_match_returns_empty_list_not_error()
    print("\n全部测试通过")
