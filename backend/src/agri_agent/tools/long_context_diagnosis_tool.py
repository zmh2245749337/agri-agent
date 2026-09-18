# src/agri_agent/tools/long_context_diagnosis_tool.py
"""
诊断模块的另一条技术路线：不做检索，把整个知识库原文一次性喂给大模型，
让模型自己在长上下文里做推理、判断哪些条目命中——对应"长上下文能力替代
检索步骤"这个思路的一个最朴素落地版本。

跟match_pest_knowledge（RapidFuzz模糊字符串匹配，见pest_knowledge_tool.py）
是解决同一个问题的两条不同技术路线，特意保留两条路线同时存在，是为了做
eval/量化对比实验（backend/eval/diagnosis_retrieval_eval.py），而不是凭
感觉判断"哪个更好"。

这不是GraphRAG。知识库目前只有77条记录，规模小，为了这么小的数据去构建
知识图谱、引入图数据库，投入产出比很低，容易显得是为了赶技术热点而做，而不是
真的解决问题。"长上下文直接推理"才是跟这个数据规模匹配的、诚实的技术选择：
77条记录全部拼起来也就几千字，完全在主流大模型的上下文窗口之内，不需要额外
的基础设施。

代价也很直接：每次诊断都要把全量知识库当输入token吃一遍，延迟和成本明显
高于RapidFuzz这种近乎零成本的字符串匹配——这个权衡本身就是eval实验要
量化出来的核心结论之一，不是"长上下文一定更好"，而是"两条路线各自的
准确率/成本曲线长什么样"。
"""
import json
import re
import sys
from threading import Lock
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from agri_agent.tools.pest_knowledge_tool import PEST_KNOWLEDGE_BASE
from agri_agent.core.my_llm import MyLLM

# 模型偶尔会不听话地在JSON数组外面加解释文字，或者用代码块围栏包一层，
# 直接json.loads(raw_text)容易炸，这里先用正则抓出形如[1, 2, 3]的子串再解析。
# 字符类里带上"-"是为了不让负数（哪怕是prompt里没要求过的、模型偶尔瞎写的
# 越界/非法index）导致整个正则匹配失败——负数本身会在下面的越界检查里被
# 过滤掉，但"能不能先把数组解析出来"和"数组里的值合不合法"应该是两步独立的
# 判断，不能让后者的问题反过来让前者也失败
_JSON_ARRAY_PATTERN = re.compile(r"\[[-\d,\s]*\]")
_RESULT_CACHE: dict[tuple[str, str], list] = {}
_CACHE_LOCK = Lock()


def _build_context_block() -> str:
    """把整个知识库编号列出来，每条记录带作物/关键词/病因，格式化成大模型
    好读的文本块。不包含recommendations字段——判断"是否命中"只需要作物和
    症状特征这两类信息，处理建议对匹配判断没有帮助，不塞进prompt多耗token"""
    lines = []
    for idx, entry in enumerate(PEST_KNOWLEDGE_BASE):
        keywords = "、".join(entry["keywords"])
        causes = "、".join(entry["causes"])
        lines.append(f"[{idx}] 作物={entry['crop']}；症状关键词={keywords}；对应病因={causes}")
    return "\n".join(lines)


def _parse_indices(raw_text: str) -> list:
    """从模型回复文本里提取JSON数组并解析成整数列表，容错模型输出格式不完全
    听话的情况；解析失败或者格式不对，安全返回空列表而不是抛异常中断流程"""
    if not raw_text:
        return []
    match = _JSON_ARRAY_PATTERN.search(raw_text)
    if not match:
        return []
    try:
        indices = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(indices, list):
        return []
    return [i for i in indices if isinstance(i, int)]


def long_context_diagnose(crop: str, symptom_text: str, llm=None) -> list:
    """
    诊断模块的长上下文直接推理版本：不做任何预先过滤/检索，把全量知识库
    连同作物、症状一起交给大模型，让模型自己在长上下文里判断哪些条目命中，
    返回值格式跟match_pest_knowledge完全一致（知识条目列表），方便eval
    实验直接拿两边的返回结果做对比，也方便以后要换掉diagnosis_agent.py
    里的检索方式时，可以直接原地替换、不用改调用方代码。

    llm参数允许调用方传入共享的MyLLM实例（避免每次调用都新建一个client连接），
    不传就用默认配置新建一个——用法上跟项目里其它Agent的llm参数是同一个习惯。
    """
    cache_key = (crop.strip(), " ".join(symptom_text.split()))
    use_cache = llm is None or isinstance(llm, MyLLM)
    if use_cache:
        with _CACHE_LOCK:
            cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            return [dict(entry) for entry in cached]

    llm = llm or MyLLM()
    context_block = _build_context_block()
    prompt = f"""下面是一份农作物病虫害知识库，每条记录的格式是"[编号] 作物=...；症状关键词=...；对应病因=..."：

{context_block}

用户种植的作物是"{crop}"，描述的症状是"{symptom_text}"。
请判断上面哪些编号的记录，作物和症状都能对应上用户的描述：
- 作物必须匹配，或者知识库记录里的作物是"通用"（"通用"类记录适用于任何作物）；
- 症状不要求逐字一致，只要语义上是同一种现象即可，比如"叶子有点黄"和"叶片发黄"应该算同一种症状，
  但含义明显不同的现象（比如"发黄"和"发黏"）不能算命中；
- 如果没有任何记录能同时满足作物和症状条件，就返回空数组，不要为了"给个结果"而勉强凑一个不太确定的匹配。

严格只输出一个JSON数组，比如[2, 5]或者[]，不要输出任何其他文字、不要用代码块包裹、不要解释理由。
"""
    raw_text = llm.invoke([{"role": "user", "content": prompt}])
    indices = _parse_indices(raw_text)

    matched = []
    for i in indices:
        if 0 <= i < len(PEST_KNOWLEDGE_BASE):
            matched.append(PEST_KNOWLEDGE_BASE[i])
    if use_cache:
        with _CACHE_LOCK:
            _RESULT_CACHE[cache_key] = [dict(entry) for entry in matched]
    return matched


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("精确匹配：", long_context_diagnose("小麦", "叶片有橙黄色条状锈斑"))
    print("插字改写（RapidFuzz已知命中不了的用例）：", long_context_diagnose("水稻", "我家水稻叶子有点黄"))
    print("作物不匹配（应该返回空）：", long_context_diagnose("玉米", "叶子发黄"))
