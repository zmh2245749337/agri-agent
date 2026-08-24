# src/agri_agent/tools/policy_match_tool.py
import json
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# parents[3]：从 tools/ 往上数三层到项目根目录（agri-agent/）
POLICY_FILE = Path(__file__).resolve().parents[3] / "data" / "policies" / "policies.json"

with open(POLICY_FILE, "r", encoding="utf-8") as f:
    POLICIES = json.load(f)

# ---------- 预处理：程序启动时只做一次，不放进每次查询的函数里 ----------

def _tokenize(text: str) -> list:
    """用jieba分词，BM25需要分词后的词语列表，不能直接吃整句话"""
    return list(jieba.cut(text))


# 每条政策拼一段"可检索文本"（标题+作物+地区+正文），分词后交给BM25建索引
_corpus_texts = [f"{p['title']} {p['crop']} {p['region']} {p['content']}" for p in POLICIES]
_tokenized_corpus = [_tokenize(t) for t in _corpus_texts]
_bm25 = BM25Okapi(_tokenized_corpus)

# BGE中文向量模型，第一次运行会自动从HuggingFace下载（几百MB）
_embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 提前把每条政策的正文都算成向量存起来，避免每次查询都重新计算全部政策的向量
# （只有查询本身的向量需要现算，这是向量检索能扛住大数据量的关键：政策库不变的话，
# 政策向量只用算一次；如果每次查询都重新embedding全部政策，数据量大了会很慢）
_policy_embeddings = _embed_model.encode(
    [p["content"] for p in POLICIES], normalize_embeddings=True
)

# ---------- 可调参数 ----------
# 政策库从7条扩充到28条之后，把TOP_K_COARSE从5调大到10——语料库变大后，
# 如果粗筛还只留5条，容易漏掉一些BM25排名不是最靠前、但地区/作物其实精确匹配
# 的候选（比如关键词表述和政策原文不完全一致，BM25打分偏低，但语义和规则都对得上）。
# 留10条候选进入下一步的地区硬过滤+向量精排，能兼顾召回率，同时数据量还没大到
# 需要专门优化性能的地步，暴力算完全够快
TOP_K_COARSE = 10  # 第一阶段粗筛保留几条候选进入精排
TOP_K_FINAL = 3    # 最终返回几条结果
W_VECTOR = 0.6     # 向量语义相似度权重
W_BM25 = 0.2       # BM25关键词匹配分权重
W_RULE = 0.2       # 规则加分权重（地区/作物精确匹配）

# 最终得分低于这个值就算"不够相关"，直接过滤掉，哪怕TOP_K_FINAL还没凑满也不硬凑。
# 这是给"政策库还很小"这个情况兜底的：库里政策条数少，粗筛阶段经常连不相关的政策
# 都会被迫进入候选（因为矮子里拔将军也要凑够TOP_K_COARSE条），如果不设下限，最终
# 结果也会出现"矮子里拔将军"式的弱相关甚至不相关结果被硬塞进返回列表。0.4是参照
# 实测数据定的：真正相关的结果基本都在0.6以上，明显不相关的误伤案例在0.3左右，
# 0.4取在两者中间。等以后政策库条数上来了，弱相关结果会被更强的候选自然挤出
# TOP_K_FINAL，这个阈值的意义会减弱，但作为兜底不会有副作用，可以一直留着。
MIN_SCORE = 0.4


def match_policy(crop: str, region: str, need: str) -> list:
    """
    两阶段检索：
    1. 关键词粗筛（BM25召回）：从全部政策里用关键词匹配打分，选出最相关的TOP_K_COARSE条候选。
    2. 地区硬过滤：候选里地区对不上（且不是"全国"通用政策）的直接剔除，不参与打分。
    3. 向量语义精排：对剩下的候选，计算查询和政策正文的BGE向量相似度。
    4. 加权组合打分：最终得分 = 向量相似度*0.6 + BM25分数*0.2 + 规则加分*0.2（作物是否匹配），
       再用MIN_SCORE过滤掉弱相关结果，返回Top-K并附上match_reason，保证可解释性。
    """
    query = f"{crop} {region} {need}"

    # ---------- 第一阶段：关键词粗筛 ----------
    tokenized_query = _tokenize(query)
    bm25_scores = _bm25.get_scores(tokenized_query)  # 对全部政策逐条打分，分数越高越相关
    coarse_indices = np.argsort(bm25_scores)[::-1][:TOP_K_COARSE]  # 取分数最高的几条索引

    # ---------- 第二阶段：向量语义精排 ----------
    query_embedding = _embed_model.encode(query, normalize_embeddings=True)

    # ---------- 第三步：加权组合打分 ----------
    max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1  # 避免除以0，把BM25分数归一化到0-1

    results = []
    for idx in coarse_indices:
        policy = POLICIES[idx]

        # 地区硬过滤：地区对不上（且不是"全国"通用政策）的候选，直接跳过，
        # 完全不参与后面的打分——地区是硬约束条件，之前用"扣0.5分"实现过，
        # 但实测发现语料库小、话题同质时，向量相似度的背景噪声本身就有
        # 0.3~0.5这个量级，扣分之后仍能压线通过MIN_SCORE（比如"云南咖啡"
        # 查询命中了"广西甘蔗"政策）。软惩罚治标不治本，所以改成硬过滤：
        # 地区不对的候选连打分的机会都没有，从根上断绝噪声翻盘的可能
        region_match = (
            policy["region"] == "全国" or policy["region"] in region or region in policy["region"]
        )
        if not region_match:
            continue

        # 向量已经normalize过，点积就等于余弦相似度，范围大致在0~1之间
        vector_score = float(np.dot(query_embedding, _policy_embeddings[idx]))
        bm25_score_norm = float(bm25_scores[idx] / max_bm25)

        # 走到这里说明地区已经过关，规则加分只需要再看作物是否沾边——
        # 作物不像地区那么"非此即彼"（比如通用类政策适用所有作物），
        # 所以作物这块保留"沾边加分、不沾边不扣分"的软性处理
        reasons = [f"地区匹配（{policy['region']}）"]
        rule_score = 0.5
        if policy["crop"] == "不限" or policy["crop"] in crop or crop in policy["crop"]:
            rule_score += 0.5
            reasons.append(f"作物匹配（{policy['crop']}）")
        if vector_score > 0.5:
            reasons.append(f"语义相似度{vector_score:.2f}")

        final_score = W_VECTOR * vector_score + W_BM25 * bm25_score_norm + W_RULE * rule_score

        results.append({
            **policy,
            "match_score": round(final_score, 3),
            "match_reason": "；".join(reasons),
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    results = [r for r in results if r["match_score"] >= MIN_SCORE]  # 弱相关结果不硬凑数
    return results[:TOP_K_FINAL]


if __name__ == "__main__":
    print("查询：水稻 / 湖南省 / 水稻种植补贴")
    for r in match_policy(crop="水稻", region="湖南省", need="水稻种植补贴"):
        print(f"  [{r['match_score']}] {r['title']} —— {r['match_reason']}")

    print("\n查询：甘蔗 / 广西壮族自治区 / 甘蔗机械化补贴")
    for r in match_policy(crop="甘蔗", region="广西壮族自治区", need="甘蔗机械化补贴"):
        print(f"  [{r['match_score']}] {r['title']} —— {r['match_reason']}")
