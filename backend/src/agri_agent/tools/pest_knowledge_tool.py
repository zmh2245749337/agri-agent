# src/agri_agent/tools/pest_knowledge_tool.py
import json
from pathlib import Path

from rapidfuzz import fuzz

# parents[3]：从 tools/ 往上数三层到项目根目录（agri-agent/），
# 知识库现在存成 JSON 数据文件（data/pest_knowledge/knowledge.json），
# 不再是 Python 模块，所以直接拼路径读文件，不需要 sys.path 那层技巧了
KNOWLEDGE_FILE = Path(__file__).resolve().parents[3] / "data" / "pest_knowledge" / "knowledge.json"

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    PEST_KNOWLEDGE_BASE = json.load(f)

# 模糊匹配的相似度阈值（0-100分）。分数越高要求越严格。
# 实测过：编辑距离类的模糊匹配对"换一两个字"（比如"叶片发黄"↔"叶子发黄"，
# 约75分）容忍度还可以，但对"中间插了几个字"（比如"叶子有点黄"↔"叶子发黄"，
# 只有约50分）效果很差——插字会把后面字符的对齐位置全部错开，字面相似度
# 掉得比换字更多，哪怕语义上完全一样。70这个阈值是权衡后的结果：能捞回
# "换字不改意思"这类情况，但"插字改写"这类情况本质上超出了模糊匹配的能力
# 范围，需要真正的语义检索（向量模型）才能解决，这个局限是已知的、接受的，
# 不是bug。
FUZZY_MATCH_THRESHOLD = 70

# 实测发现的第二个问题：知识库里如果存在两个关键词本身长得很像（哪怕含义完全
# 不同，比如"叶片发黄"vs"叶片发黏"，只差一个字），命中其中一个的输入大概率
# 也会误伤另一个——因为它们本身编辑距离就很近，不管阈值怎么调都很难只让正确
# 的那条过线。SCORE_MARGIN 是缓解这个问题的办法：不再是"只要单个关键词单独
# 过线就收进结果"，而是先给每条候选记录算出"最匹配的那个关键词"的分数，最后
# 只保留和全场最高分差距不超过 SCORE_MARGIN 的记录——真正对的答案通常远高于
# 巧合性的误伤（比如原文完全命中是100分，误伤的"发黏"只有75分左右，差距25分
# 会被过滤掉），这样能把"分数勉强达标但明显不如最佳匹配"的噪音去掉。
SCORE_MARGIN = 10


def match_pest_knowledge(crop: str, symptom_text: str) -> list:
    """
    根据作物名称和症状描述文本，在本地知识库里做模糊关键词匹配。
    返回匹配到的知识条目列表（可能是空列表，表示没匹配到）。

    做法分两步：
    1. 先给每条候选记录算出"最匹配的那个关键词"的相似度分数（不再是逐个关键词
       独立判断"过线就收"，而是先收集所有候选记录的最佳分数）；
    2. 只保留分数达标、且和全场最高分差距不大的记录，过滤掉分数勉强达标但明显
       不如最佳匹配的巧合性误判（比如"发黄"和"发黏"字面相似但含义不同的情况）。
    """
    scored = []
    for entry in PEST_KNOWLEDGE_BASE:
        # 作物要么对上，要么这条记录是"通用"类型
        if entry["crop"] not in (crop, "通用"):
            continue
        # partial_ratio：在symptom_text里找和kw最相似的一段，算相似度分数，
        # 能容忍个别字符替换（比如"叶片"↔"叶子"），但不能容忍插入式改写
        # （比如"发黄"↔"有点黄"）——这是编辑距离算法的固有边界，见上面注释
        best_score = max(
            (fuzz.partial_ratio(kw, symptom_text) for kw in entry["keywords"]),
            default=0,
        )
        if best_score >= FUZZY_MATCH_THRESHOLD:
            scored.append((best_score, entry))

    if not scored:
        return []

    top_score = max(score for score, _ in scored)
    return [entry for score, entry in scored if top_score - score <= SCORE_MARGIN]


if __name__ == "__main__":
    # 标准说法：关键词表里原本就有的写法
    print("标准说法：", match_pest_knowledge("水稻", "最近发现叶子发黄，还有点卷"))

    # 换字不改意思：关键词表里没有，但字面上只是换了一两个字，模糊匹配应该能捞回来，
    # 之前这组测试意外多匹配上了一条无关的"叶片发黏"记录，加了排序过滤后应该只剩正确的那条
    print("换字说法：", match_pest_knowledge("水稻", "我家水稻叶片发黄了"))

    # 插字改写：字面上插入了新字，模糊匹配大概率捞不回来，这是已知边界，不是bug
    print("插字说法：", match_pest_knowledge("水稻", "我家水稻叶子有点黄"))