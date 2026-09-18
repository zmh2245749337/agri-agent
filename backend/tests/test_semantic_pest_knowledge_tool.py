import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.tools.semantic_pest_knowledge_tool import semantic_match_pest_knowledge


def _causes(results: list) -> str:
    return " ".join(cause for entry in results for cause in entry["causes"])


def test_semantic_match_recovers_independent_diagnosis_paraphrases():
    cases = [
        ("水稻", "叶面慢慢冒出褐色梭子形斑块，湿度大时扩得很快", "稻瘟病"),
        ("玉米", "叶片上的灰褐斑越拉越长，看着像一个个梭子", "大斑病"),
        ("黄瓜", "叶面出现受叶脉限制的黄斑，背面有灰黑色霉层", "霜霉病"),
        ("白菜", "菜心从根部开始变软出水，闻起来有明显臭味", "软腐病"),
    ]

    for crop, symptom, expected in cases:
        assert expected in _causes(semantic_match_pest_knowledge(crop, symptom))


def test_semantic_match_rejects_unrelated_symptom_below_threshold():
    assert semantic_match_pest_knowledge("苹果", "果实吃起来比往年酸很多") == []
