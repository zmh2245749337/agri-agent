# eval/diagnosis_retrieval_eval.py
"""
诊断模块两条技术路线的量化对比实验：
1. match_pest_knowledge  —— RapidFuzz模糊字符串匹配（tools/pest_knowledge_tool.py）
2. long_context_diagnose —— 长上下文直接推理（tools/long_context_diagnosis_tool.py）

README的Roadmap里一直留着"模糊匹配 vs 向量检索的效果对比实验"这一项，这次
一并做了，但选择对比的第二条路线是"长上下文"而不是"向量检索"——原因见
long_context_diagnosis_tool.py顶部注释：77条记录的知识库规模下，向量检索/
GraphRAG这类更重的基础设施投入产出比不高，长上下文直接推理才是跟数据规模
匹配的、诚实的技术选择。

这个脚本要跑起来需要：
1. .env里配置好LLM_API_KEY/LLM_MODEL_ID/LLM_BASE_URL（long_context_diagnose
   要真实调用大模型，每条测试用例都会发一次请求，测试集有14条，会有14次
   真实API调用，注意这一点，不要在配额紧张的时候跑）
2. 装好了backend/requirements.txt里的rapidfuzz（match_pest_knowledge本身
   不需要网络，瞬间跑完，但需要真实的rapidfuzz包，不能用沙箱里的桩实现）

用法：
    cd backend
    python eval/diagnosis_retrieval_eval.py

跑完会在终端打印整体命中率、分类别命中情况、以及两种方法结果不一致的用例，
同时把完整报告写到 eval/results/diagnosis_retrieval_report.md。
"""
import sys
import time
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agri_agent.core.my_llm import MyLLM
from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge
from agri_agent.tools.long_context_diagnosis_tool import long_context_diagnose

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# 人工构造的带标注测试集，覆盖五类场景：
# - "精确匹配"：症状描述跟知识库关键词几乎一致，两种方法都应该能命中，用来做
#   sanity check（如果连这个都不过，说明代码本身有bug，不是技术路线的差异）
# - "近义改写"：换一两个字但字面结构没变（比如"叶片发黄"→"叶子发黄"），
#   RapidFuzz的partial_ratio对这种情况容忍度还可以，理论上也应该命中
# - "插字改写"：中间插入了新词（比如"叶子发黄"→"叶子有点发黄"），这是
#   pest_knowledge_tool.py注释里明确写过的RapidFuzz已知边界——编辑距离类
#   算法处理不好这种情况。这批用例是本次实验最核心的假设：长上下文推理应该
#   能命中插字改写，RapidFuzz大概率命中不了
# - "作物不匹配"：症状关键词本身存在于知识库，但用户说的作物对不上（比如
#   拿水稻的"叶子发黄"关键词套在玉米头上），两种方法都应该返回空——这条测的
#   是长上下文推理会不会"因为更灵活，反而在该拒绝的时候不拒绝"（幻觉风险，
#   这是长上下文/大模型直接推理这条路线最需要警惕的失败模式）
# - "真阴性"：知识库里压根没有对应症状，两种方法都应该返回空
#
# expected_causes_contains为None表示这条用例期望两种方法都返回空列表；
# 否则期望返回的匹配条目里，至少有一条的causes字段包含这个子串
TEST_CASES = [
    # ---- 精确匹配（sanity check） ----
    {"category": "精确匹配", "crop": "小麦", "symptom_text": "叶片有橙黄色条状锈斑",
     "expected_causes_contains": "条锈病"},
    {"category": "精确匹配", "crop": "黄瓜", "symptom_text": "叶片表面有白色粉状物",
     "expected_causes_contains": "白粉病"},
    {"category": "精确匹配", "crop": "通用", "symptom_text": "叶片发黏",
     "expected_causes_contains": "蚜虫"},

    # ---- 近义改写（换字不改意思，RapidFuzz预期能命中） ----
    {"category": "近义改写", "crop": "水稻", "symptom_text": "我家水稻叶片发黄了",
     "expected_causes_contains": "纹枯病"},
    {"category": "近义改写", "crop": "黄瓜", "symptom_text": "叶子跟撒了面粉一样",
     "expected_causes_contains": "白粉病"},
    {"category": "近义改写", "crop": "玉米", "symptom_text": "叶片有排孔状的虫眼",
     "expected_causes_contains": "玉米螟"},

    # ---- 插字改写（核心假设：RapidFuzz预期命中不了，长上下文预期能命中） ----
    {"category": "插字改写", "crop": "水稻", "symptom_text": "我家水稻叶子有点黄",
     "expected_causes_contains": "纹枯病"},
    {"category": "插字改写", "crop": "水稻", "symptom_text": "叶片卷曲，而且尖端发白干枯",
     "expected_causes_contains": "稻纵卷叶螟"},
    {"category": "插字改写", "crop": "玉米", "symptom_text": "叶子上慢慢长出了一节一节的褐色梭形斑纹",
     "expected_causes_contains": "大斑病"},
    {"category": "插字改写", "crop": "番茄", "symptom_text": "叶子背面长了一层白色的霉，叶片上还有些水浸状的暗斑",
     "expected_causes_contains": "晚疫病"},
    {"category": "插字改写", "crop": "柑橘", "symptom_text": "新梢部位这几天慢慢开始发黄了",
     "expected_causes_contains": "黄龙病"},
    {"category": "插字改写", "crop": "蔬菜", "symptom_text": "叶子卷了，长得也不太好看，有点畸形",
     "expected_causes_contains": "蚜虫传播病毒病"},

    # ---- 作物不匹配（测幻觉风险：症状关键词命中，但作物应该拦住） ----
    # 注意：这两条特意选的是知识库里没有"通用"类等价条目的症状关键词——
    # 如果换成有通用等价条目的症状（比如"白粉病"，通用类目下也有一条），
    # 结果应该是命中通用条目而不是空，那样反而不是在测"作物过滤"这件事了
    {"category": "作物不匹配", "crop": "玉米", "symptom_text": "叶子发黄",
     "expected_causes_contains": None},
    {"category": "作物不匹配", "crop": "番茄", "symptom_text": "秧苗徒长细高",
     "expected_causes_contains": None},

    # ---- 真阴性（知识库里压根没有对应症状） ----
    {"category": "真阴性", "crop": "水稻", "symptom_text": "植株突然提前开花，稻穗比往年小很多",
     "expected_causes_contains": None},
    {"category": "真阴性", "crop": "苹果", "symptom_text": "果实的味道变得特别酸涩",
     "expected_causes_contains": None},
]


def _is_hit(matched: list, expected_causes_contains) -> bool:
    if expected_causes_contains is None:
        return matched == []
    return any(expected_causes_contains in "".join(entry["causes"]) for entry in matched)


def run_eval(method_name: str, match_fn) -> dict:
    """跑一遍完整测试集，返回这个方法的整体统计和逐条明细。match_fn签名统一
    要求是match_fn(crop, symptom_text) -> list，两个待测函数签名天然一致
    （long_context_diagnose多出来的llm参数用functools.partial提前绑定好）"""
    details = []
    hit_count = 0
    total_latency = 0.0

    for case in TEST_CASES:
        start = time.time()
        try:
            matched = match_fn(case["crop"], case["symptom_text"])
        except Exception as e:
            matched = []
            print(f"[WARN] {method_name}在用例「{case['symptom_text']}」上执行失败：{e}")
        latency = time.time() - start
        total_latency += latency

        hit = _is_hit(matched, case["expected_causes_contains"])
        hit_count += hit
        details.append({
            "category": case["category"],
            "crop": case["crop"],
            "symptom_text": case["symptom_text"],
            "expected": case["expected_causes_contains"] or "（应为空）",
            "hit": hit,
            "matched_count": len(matched),
            "latency": latency,
        })

    return {
        "method_name": method_name,
        "accuracy": hit_count / len(TEST_CASES),
        "hit_count": hit_count,
        "total_cases": len(TEST_CASES),
        "avg_latency": total_latency / len(TEST_CASES),
        "details": details,
    }


def _category_breakdown(fuzzy_result: dict, long_context_result: dict) -> list:
    categories = sorted(set(d["category"] for d in fuzzy_result["details"]))
    rows = []
    for cat in categories:
        fuzzy_cases = [d for d in fuzzy_result["details"] if d["category"] == cat]
        lc_cases = [d for d in long_context_result["details"] if d["category"] == cat]
        rows.append({
            "category": cat,
            "fuzzy": f"{sum(d['hit'] for d in fuzzy_cases)}/{len(fuzzy_cases)}",
            "long_context": f"{sum(d['hit'] for d in lc_cases)}/{len(lc_cases)}",
        })
    return rows


def _print_report(fuzzy_result: dict, long_context_result: dict) -> None:
    print("\n" + "=" * 60)
    print("诊断模块检索方法对比实验结果")
    print("=" * 60)
    for result in (fuzzy_result, long_context_result):
        print(f"\n【{result['method_name']}】")
        print(f"  整体命中率：{result['hit_count']}/{result['total_cases']} "
              f"({result['accuracy']:.0%})")
        print(f"  平均耗时：{result['avg_latency'] * 1000:.1f}ms")

    print("\n分类别命中情况：")
    print(f"{'类别':<10}{'RapidFuzz':<14}{'长上下文':<14}")
    for row in _category_breakdown(fuzzy_result, long_context_result):
        print(f"{row['category']:<10}{row['fuzzy']:<14}{row['long_context']:<14}")

    print("\n两种方法结果不一致的用例（重点核查对象）：")
    diffs = 0
    for f_detail, l_detail in zip(fuzzy_result["details"], long_context_result["details"]):
        if f_detail["hit"] != l_detail["hit"]:
            diffs += 1
            print(f"  [{f_detail['category']}] {f_detail['crop']}「{f_detail['symptom_text']}」"
                  f"→ RapidFuzz{'命中' if f_detail['hit'] else '未命中'} / "
                  f"长上下文{'命中' if l_detail['hit'] else '未命中'}")
    if diffs == 0:
        print("  （无，两种方法在所有用例上结果完全一致）")


def _write_markdown_report(fuzzy_result: dict, long_context_result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "diagnosis_retrieval_report.md"

    lines = ["# 诊断模块检索方法对比实验报告", ""]
    lines.append("| 方法 | 命中率 | 平均耗时 |")
    lines.append("| --- | --- | --- |")
    for result in (fuzzy_result, long_context_result):
        lines.append(
            f"| {result['method_name']} | {result['hit_count']}/{result['total_cases']} "
            f"({result['accuracy']:.0%}) | {result['avg_latency'] * 1000:.1f}ms |"
        )
    lines.append("")
    lines.append("## 分类别命中情况")
    lines.append("")
    lines.append("| 类别 | RapidFuzz | 长上下文 |")
    lines.append("| --- | --- | --- |")
    for row in _category_breakdown(fuzzy_result, long_context_result):
        lines.append(f"| {row['category']} | {row['fuzzy']} | {row['long_context']} |")
    lines.append("")
    lines.append("## 逐条明细")
    lines.append("")
    lines.append("| 类别 | 作物 | 症状描述 | RapidFuzz | 长上下文 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for f_detail, l_detail in zip(fuzzy_result["details"], long_context_result["details"]):
        lines.append(
            f"| {f_detail['category']} | {f_detail['crop']} | {f_detail['symptom_text']} | "
            f"{'✅命中' if f_detail['hit'] else '❌未命中'} | "
            f"{'✅命中' if l_detail['hit'] else '❌未命中'} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    print(f"共{len(TEST_CASES)}条测试用例，开始跑RapidFuzz模糊匹配（不需要网络，瞬间完成）...")
    fuzzy_result = run_eval("RapidFuzz模糊匹配", match_pest_knowledge)

    print("开始跑长上下文直接推理（需要真实网络+LLM API，每条用例都要调用一次大模型，会比较慢）...")
    llm = MyLLM()
    long_context_result = run_eval("长上下文直接推理", partial(long_context_diagnose, llm=llm))

    _print_report(fuzzy_result, long_context_result)
    out_path = _write_markdown_report(fuzzy_result, long_context_result)
    print(f"\n完整报告已写入：{out_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
