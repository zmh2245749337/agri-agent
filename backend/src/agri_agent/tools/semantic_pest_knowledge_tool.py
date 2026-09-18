"""小型病虫害知识库的本地向量语义召回。

复用政策检索已经加载的 BGE 中文模型，不新增模型或网络依赖。字面检索为空时
才调用本模块；高于阈值的候选直接返回，模糊样本继续交给长上下文兜底。
"""
import numpy as np

from agri_agent.tools.pest_knowledge_tool import PEST_KNOWLEDGE_BASE
from agri_agent.tools.policy_match_tool import _embed_model


SEMANTIC_MATCH_THRESHOLD = 0.68
SEMANTIC_BORDERLINE_THRESHOLD = 0.64
SEMANTIC_BORDERLINE_LEAD = 0.03
SEMANTIC_SCORE_MARGIN = 0.04
SEMANTIC_TOP_K = 2

_entry_texts = [
    f"症状：{'；'.join(entry['keywords'])}；病因：{'；'.join(entry['causes'])}"
    for entry in PEST_KNOWLEDGE_BASE
]
_entry_embeddings = None


def _embeddings():
    global _entry_embeddings
    if _entry_embeddings is None:
        _entry_embeddings = _embed_model.encode(_entry_texts, normalize_embeddings=True)
    return _entry_embeddings


def semantic_match_pest_knowledge(crop: str, symptom_text: str) -> list:
    """按作物硬过滤后做 BGE 语义召回，只返回高置信且接近最高分的候选。"""
    candidate_indices = [
        index
        for index, entry in enumerate(PEST_KNOWLEDGE_BASE)
        if entry["crop"] in (crop, "通用")
    ]
    if not candidate_indices:
        return []

    query_embedding = _embed_model.encode(symptom_text, normalize_embeddings=True)
    embeddings = _embeddings()
    scored = sorted(
        (
            (float(np.dot(query_embedding, embeddings[index])), index)
            for index in candidate_indices
        ),
        reverse=True,
    )
    top_score = scored[0][0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    high_confidence = top_score >= SEMANTIC_MATCH_THRESHOLD
    clear_borderline_lead = (
        top_score >= SEMANTIC_BORDERLINE_THRESHOLD
        and top_score - runner_up_score >= SEMANTIC_BORDERLINE_LEAD
    )
    if not (high_confidence or clear_borderline_lead):
        return []

    matched = []
    for score, index in scored:
        if len(matched) >= SEMANTIC_TOP_K or top_score - score > SEMANTIC_SCORE_MARGIN:
            break
        entry = dict(PEST_KNOWLEDGE_BASE[index])
        entry["match_score"] = round(score, 3)
        entry["match_reason"] = "作物硬过滤后BGE语义召回"
        matched.append(entry)
    return matched
