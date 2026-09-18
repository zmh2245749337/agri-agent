"""构建 AgriAgent V2 结果导向挑战集。

旧版 ``e2e_tasks.jsonl`` 是模板回归集，主要检查固定路由、槽位和工具轨迹。
这套挑战集不把内部 route 当成标准答案，而是验证对话结束后是否完成业务目标：

* 最终上下文中的作物、城市、地区是否正确；
* 诊断、天气、政策能力是否真正执行；
* 工具或子 Agent 的结果中是否包含可核验的知识/政策/天气证据；
* 缺参时是否先追问，边界问题是否避免误调用工具。

覆盖矩阵不是先拍一个总数再凑题：

    7 种能力组合 × 4 种交互状态 × 4 种表达风格 = 112
    再加 8 个 no-action 边界场景 = 120

所有问题均为合成挑战场景，不代表线上用户流量。事实标准来自仓库知识条目、政策
来源和冻结天气响应；用户表达为独立改写，不直接把现有回归集问题复制过来。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"

FAMILIES = (
    "diagnosis",
    "weather",
    "policy",
    "diagnosis_weather",
    "diagnosis_policy",
    "weather_policy",
    "diagnosis_weather_policy",
)
INTERACTIONS = ("complete", "missing", "correction", "carryover")
STYLES = ("standard", "colloquial", "implicit", "noisy")

DOMAIN_TO_TOOL = {
    "diagnosis": "diagnose_crop_disease",
    "weather": "get_weather_forecast",
    "policy": "search_subsidy_policy",
}


# 症状描述刻意不直接复制知识库第一关键词。expected_term 只用于在工具/子 Agent
# 的原始证据中核对命中的知识条目，不用于要求最终自然语言逐字复述。
DIAGNOSIS_CASES = [
    {"crop": "水稻", "symptom": "叶子颜色越来越淡，还夹着几片黄叶，田里排水也不太顺", "expected_term": "纹枯病"},
    {"crop": "水稻", "symptom": "稻秆里面像被钻过，能看到孔，有些穗直接白了", "expected_term": "二化螟"},
    {"crop": "水稻", "symptom": "叶面慢慢冒出褐色梭子形斑块，湿度大时扩得很快", "expected_term": "稻瘟病"},
    {"crop": "水稻", "symptom": "稻叶被卷成细筒，里面还能看到虫子啃过的痕迹", "expected_term": "稻纵卷叶螟"},
    {"crop": "小麦", "symptom": "麦叶上是一道道橙黄粉末，手碰以后会沾到颜色", "expected_term": "条锈病"},
    {"crop": "小麦", "symptom": "麦叶和茎表面像蒙了白灰，擦掉以后过几天又长出来", "expected_term": "白粉病"},
    {"crop": "玉米", "symptom": "叶子被咬出成排的小孔，茎上也能找到蛀口和虫粪", "expected_term": "玉米螟"},
    {"crop": "玉米", "symptom": "叶片上的灰褐斑越拉越长，看着像一个个梭子", "expected_term": "大斑病"},
    {"crop": "大豆", "symptom": "新叶深浅颜色相间，还皱巴巴的，整株明显长不高", "expected_term": "花叶病毒病"},
    {"crop": "番茄", "symptom": "叶背有一层浅色霉，叶和茎上的暗斑像被水泡过", "expected_term": "晚疫病"},
    {"crop": "番茄", "symptom": "果子最下面先黑一小块，后来逐渐往里塌，浇水不太均匀", "expected_term": "脐腐病"},
    {"crop": "黄瓜", "symptom": "叶子正面出现受叶脉限制的黄斑，背面发灰发黑", "expected_term": "霜霉病"},
    {"crop": "黄瓜", "symptom": "叶面像被撒了一层面粉，擦掉后还会重新出现", "expected_term": "白粉病"},
    {"crop": "白菜", "symptom": "菜心从根部开始变软出水，闻起来有明显臭味", "expected_term": "软腐病"},
    {"crop": "辣椒", "symptom": "辣椒果上有往里陷的斑，斑面带着一圈圈纹路和黑点", "expected_term": "炭疽病"},
    {"crop": "茄子", "symptom": "下部叶子先黄，接着整株发蔫，剖开茎里面已经变褐", "expected_term": "黄萎病"},
    {"crop": "柑橘", "symptom": "嫩梢先黄，叶片颜色一块深一块浅，沿叶脉也在黄化", "expected_term": "黄龙病"},
    {"crop": "葡萄", "symptom": "叶面有油渍一样的黄斑，翻到背面能看到白色霜层", "expected_term": "霜霉病"},
    {"crop": "花生", "symptom": "植株突然整棵蔫下去，叶子还挂着不落，很快就枯死", "expected_term": "青枯病"},
    {"crop": "马铃薯", "symptom": "叶片出现湿漉漉的暗绿斑，背面有白霉，薯块也开始凹陷", "expected_term": "晚疫病"},
    {"crop": "苹果", "symptom": "果面从一个褐点开始腐烂，后来形成一圈圈轮纹，枝干上也有类似病斑", "expected_term": "轮纹病"},
    {"crop": "茶叶", "symptom": "叶子边缘先出现半圆褐斑，斑块中间逐渐发灰，叶缘开始枯黄", "expected_term": "茶炭疽病"},
    {"crop": "甘蔗", "symptom": "蔗株顶端抽出一根黑色细鞭，植株变矮，旁边分蘖反而增多", "expected_term": "黑穗病"},
    {"crop": "蔬菜", "symptom": "新叶卷得厉害，长出来的形状也不正常，嫩叶上还能看到小虫", "expected_term": "蚜虫"},
]


# 每项都对应 policies.json 中的一条真实记录。source 是可核验的 gold evidence；
# city 只为需要同时查询天气的组合场景提供城市槽位。
POLICY_CASES = [
    {"crop": "水稻", "region": "广东省", "city": "广州", "need": "种粮和耕地补贴", "source": "tuliu.com/read-155581", "title_term": "耕地地力保护补贴"},
    {"crop": "水稻", "region": "湖南省", "city": "长沙", "need": "耕地地力保护补贴", "source": "tuliu.com/read-155581", "title_term": "耕地地力保护补贴"},
    {"crop": "甘蔗", "region": "广西壮族自治区", "city": "南宁", "need": "糖料蔗种植补助", "source": "yntw.com/2026/05/37904", "title_term": "糖料蔗"},
    {"crop": "小麦", "region": "河南省", "city": "郑州", "need": "最低收购价政策", "source": "lswz.gov.cn", "title_term": "小麦最低收购价格"},
    {"crop": "棉花", "region": "新疆维吾尔自治区", "city": "乌鲁木齐", "need": "棉花目标价格政策", "source": "ndrc.gov.cn", "title_term": "棉花目标价格"},
    {"crop": "苹果", "region": "甘肃省", "city": "兰州", "need": "苹果产业扶持", "source": "gsjn.gov.cn", "title_term": "静宁苹果"},
    {"crop": "玉米", "region": "吉林省", "city": "长春", "need": "生产者补贴", "source": "agri.jl.gov.cn", "title_term": "生产者补贴"},
    {"crop": "大豆", "region": "辽宁省", "city": "沈阳", "need": "生产者补贴", "source": "nyncj.yingkou.gov.cn", "title_term": "生产者补贴"},
    {"crop": "油菜", "region": "湖北省", "city": "武汉", "need": "油菜轮作项目", "source": "jiangxia.gov.cn", "title_term": "油菜轮作"},
    {"crop": "蔬菜", "region": "福建省", "city": "福州", "need": "温室大棚补贴", "source": "nynct.fujian.gov.cn", "title_term": "温室大棚"},
    {"crop": "玉米", "region": "内蒙古自治区", "city": "呼和浩特", "need": "生产者补贴和轮作补助", "source": "finance.sina.com.cn", "title_term": "生产者补贴"},
    {"crop": "小麦", "region": "陕西省", "city": "西安", "need": "耕地地力保护补贴", "source": "nynct.shaanxi.gov.cn", "title_term": "耕地地力保护补贴"},
    {"crop": "玉米", "region": "云南省", "city": "昆明", "need": "带状复合种植补助", "source": "nync.yn.gov.cn", "title_term": "带状复合种植"},
    {"crop": "水稻", "region": "安徽省", "city": "合肥", "need": "农机购置补贴", "source": "ny.huaibei.gov.cn", "title_term": "农机购置"},
    {"crop": "水稻", "region": "江苏省", "city": "南京", "need": "农机购置补贴", "source": "yixing.gov.cn", "title_term": "农机购置"},
    {"crop": "玉米", "region": "山西省", "city": "太原", "need": "农机报废更新补贴", "source": "xwboo.com", "title_term": "报废更新"},
    {"crop": "大豆", "region": "黑龙江省", "city": "哈尔滨", "need": "农机报废更新补贴", "source": "nynct.hlj.gov.cn", "title_term": "报废更新"},
    {"crop": "不限", "region": "山东省", "city": "青岛", "need": "农业保险保费补贴", "source": "cfsn.cn", "title_term": "农业保险"},
    {"crop": "茶叶", "region": "云南省", "city": "昆明", "need": "绿色高产高效项目", "source": "nync.yn.gov.cn", "title_term": "绿色高产高效"},
    {"crop": "大豆", "region": "吉林省", "city": "长春", "need": "耕地轮作补贴", "source": "agri.jl.gov.cn", "title_term": "耕地轮作"},
]

# 同时包含诊断和政策时，只选择知识库里确实有对应诊断条目的政策作物，避免为了
# 拼覆盖矩阵造出“棉花症状却拿番茄条目当标准答案”这种伪 gold label。
COMPOSITE_POLICY_INDICES = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19)


LOCATIONS = [
    ("湖南省", "长沙"), ("广东省", "广州"), ("江苏省", "南京"),
    ("四川省", "成都"), ("湖北省", "武汉"), ("河南省", "郑州"),
    ("山东省", "济南"), ("安徽省", "合肥"), ("江西省", "南昌"),
    ("陕西省", "西安"), ("云南省", "昆明"), ("黑龙江省", "哈尔滨"),
    ("辽宁省", "沈阳"), ("福建省", "福州"), ("贵州省", "贵阳"),
    ("山西省", "太原"), ("甘肃省", "兰州"), ("吉林省", "长春"),
    ("广西壮族自治区", "南宁"), ("内蒙古自治区", "呼和浩特"),
]


STYLE_PREFIX = {
    "standard": "",
    "colloquial": "麻烦帮我看看，",
    "implicit": "我不太确定该先处理哪一件事，",
    "noisy": "补充一下，前两天浇过水，家里人说法也不一致；",
}

SLOT_LABELS = {"crop": "作物", "city": "城市", "region": "地区"}


def _domains(family: str) -> list[str]:
    return [name for name in ("diagnosis", "weather", "policy") if name in family]


def _base_values(index: int, family: str) -> dict:
    policy_index = index % len(POLICY_CASES)
    if "diagnosis" in family and "policy" in family:
        policy_index = COMPOSITE_POLICY_INDICES[index % len(COMPOSITE_POLICY_INDICES)]
    policy = POLICY_CASES[policy_index]
    diagnosis = DIAGNOSIS_CASES[index % len(DIAGNOSIS_CASES)]
    if "diagnosis" in family and "policy" in family:
        matching = [item for item in DIAGNOSIS_CASES if item["crop"] == policy["crop"]]
        if not matching:
            raise AssertionError(f"复合场景缺少作物 {policy['crop']} 的诊断 gold 条目")
        diagnosis = matching[index % len(matching)]
    region, city = LOCATIONS[index % len(LOCATIONS)]
    if "policy" in family:
        crop = policy["crop"] if policy["crop"] != "不限" else diagnosis["crop"]
        region = policy["region"]
        city = policy["city"]
    else:
        crop = diagnosis["crop"]
    return {
        "crop": crop,
        "symptom": diagnosis["symptom"],
        "diagnosis_term": diagnosis["expected_term"],
        "region": region,
        "city": city,
        "need": policy["need"],
        "policy_source": policy["source"],
        "policy_title_term": policy["title_term"],
    }


def _request(family: str, style: str, values: dict, omit: str | None = None) -> str:
    crop = "" if omit == "crop" else values["crop"]
    city = "" if omit == "city" else values["city"]
    region = "" if omit == "region" else values["region"]
    symptom = values["symptom"]
    if omit == "crop":
        # 缺作物场景不能在症状里继续泄露作物名（例如删掉“小麦”却保留
        # “麦叶”）。只去除作物身份线索，症状本身和难度保持不变。
        crop_neutral_terms = {
            "麦叶": "叶片",
            "稻叶": "叶片",
            "稻秆": "茎秆",
            "蔗株": "植株",
        }
        for specific, neutral in crop_neutral_terms.items():
            symptom = symptom.replace(specific, neutral)
    need = values["need"]
    prefix = STYLE_PREFIX[style]

    if family == "diagnosis":
        body = f"{crop + '最近' if crop else '地里的作物最近'}{symptom}，这更像什么问题，应该先怎么处理？"
    elif family == "weather":
        place = city or "我这里"
        body = f"{place}明后两天适不适合打药？请结合降雨和风力说明。"
    elif family == "policy":
        place = region or "当地"
        body = f"我在{place}种{crop}，想确认能不能申请{need}，请给出政策来源。"
    elif family == "diagnosis_weather":
        if city:
            body = f"我在{city}种的{crop}{symptom}，还想知道明天能不能马上打药，请一起判断。"
        else:
            body = f"我种的{crop}{symptom}，还想知道明天能不能马上打药，请一起判断。"
    elif family == "diagnosis_policy":
        place = region or "当地"
        body = f"我在{place}种{crop}，现在{symptom}；处理病害的同时，也帮我查一下{need}。"
    elif family == "weather_policy":
        if city:
            location = f"{region}{city}" if region else city
            body = f"我在{location}种{crop}，想看近期天气并确认{need}，请整理成安排。"
        else:
            place_region = region or "当地"
            body = f"我在{place_region}种{crop}，想看当地近期天气并确认{need}，请整理成安排。"
    else:
        if region and city:
            location = f"{region}{city}"
        else:
            location = region or city or "当地"
        body = f"我在{location}种{crop}，现在{symptom}；请同时判断病害、近期天气和{need}，给出有依据的处理顺序。"

    if style == "implicit":
        body = body.replace("请一起判断", "别只回答其中一件事")
        body = body.replace("请整理成安排", "我得决定这几天先做什么")
        body = body.replace("请同时判断", "我需要把这几件事放在一起考虑：")
    elif style == "noisy":
        body += " 地块不大，暂时不考虑换作物；无关信息可以忽略。"
    return prefix + body


def _slot_for_interaction(family: str) -> str:
    if family == "diagnosis":
        return "crop"
    if family == "weather":
        return "city"
    if family in {"policy", "diagnosis_policy"}:
        return "region"
    if family == "diagnosis_weather":
        return "city"
    if family == "weather_policy":
        return "city"
    # 三能力问题同时需要城市和省级地区。保留城市却省略省份时，系统可以可靠地
    # 由城市推出省份，不应强迫用户重复回答；因此这里真正省略城市。
    return "city"


def _alternate(slot: str, values: dict) -> str:
    alternatives = {
        "crop": "玉米" if values["crop"] != "玉米" else "水稻",
        "city": "成都" if values["city"] != "成都" else "武汉",
        "region": "四川省" if values["region"] != "四川省" else "湖北省",
    }
    return alternatives[slot]


def _evidence(family: str, values: dict) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    if "diagnosis" in family:
        evidence["diagnosis"] = [values["diagnosis_term"]]
    if "weather" in family:
        evidence["weather"] = [values["city"]]
    if "policy" in family:
        # title/source 任一出现即可；评分器对每个领域使用 any-of 语义。
        evidence["policy"] = [values["policy_source"], values["policy_title_term"]]
    return evidence


def _goal(family: str, values: dict) -> dict:
    context = {"crop": values["crop"]} if "diagnosis" in family or "policy" in family else {}
    if "weather" in family:
        context["city"] = values["city"]
    if "policy" in family:
        context["region"] = values["region"]
    return {
        "context": context,
        "required_domains": _domains(family),
        "evidence_any": _evidence(family, values),
        "no_action": False,
    }


def _turns(family: str, interaction: str, style: str, values: dict) -> list[dict]:
    if interaction == "complete":
        return [{"user": _request(family, style, values)}]

    slot = _slot_for_interaction(family)
    if interaction == "missing":
        return [
            {
                "user": _request(family, style, values, omit=slot),
                "checkpoint": {
                    "missing_contains": [slot],
                    "forbid_domain_actions": True,
                },
            },
            {"user": f"补充一下，{SLOT_LABELS[slot]}是{values[slot]}，请继续处理刚才的问题。"},
        ]

    if interaction == "correction":
        wrong_values = dict(values)
        wrong_values[slot] = _alternate(slot, values)
        setup_parts = []
        if "diagnosis" in family or "policy" in family:
            setup_parts.append(f"作物是{wrong_values['crop']}")
        if "weather" in family:
            setup_parts.append(f"城市是{wrong_values['city']}")
        if "policy" in family:
            setup_parts.append(f"地区按{wrong_values['region']}算")
        return [
            {"user": f"先记一下：{'，'.join(setup_parts)}。稍后我再说具体需求。"},
            {
                "user": (
                    f"刚才的{SLOT_LABELS[slot]}说错了，正确的是{values[slot]}，不要沿用旧值。"
                    + _request(family, style, values, omit=slot)
                )
            },
        ]

    # carryover：第一轮只建立上下文，第二轮不重复关键槽位，检查状态继承。
    setup_parts = []
    if "diagnosis" in family or "policy" in family:
        setup_parts.append(f"作物是{values['crop']}")
    if "weather" in family:
        setup_parts.append(f"城市是{values['city']}")
    if "policy" in family:
        setup_parts.append(f"地区按{values['region']}算")
    omitted = _slot_for_interaction(family)
    return [
        {"user": f"先记一下：{'，'.join(setup_parts)}。后面的问题都按这些信息处理。"},
        {"user": _request(family, style, values, omit=omitted)},
    ]


def _scenario(family: str, interaction: str, style: str, index: int) -> dict:
    values = _base_values(index, family)
    return {
        "id": f"v2_{family}_{interaction}_{style}",
        "category": family,
        "interaction": interaction,
        "style": style,
        "source": "coverage_matrix_and_repository_facts",
        "provenance": {
            "facts": "repository_knowledge_policy_and_frozen_weather",
            "utterance": "independently_rewritten_synthetic_challenge",
            "online_user_traffic": False,
        },
        "turns": _turns(family, interaction, style, values),
        "goal": _goal(family, values),
    }


BOUNDARY_SCENARIOS = [
    ("boundary_smalltalk", "你好，先介绍一下你能提供哪些农业方面的帮助。"),
    ("boundary_thanks", "谢谢，刚才的问题已经解决了，先不用再查任何东西。"),
    ("boundary_negation", "不用查天气，也不要调用工具，我只是想知道轮作为什么能减轻连作障碍。"),
    ("boundary_out_of_scope", "帮我起草一份土地租赁合同，再算一下违约金。"),
    ("boundary_vague", "地里好像不太对，你先别乱查，告诉我还需要补充什么。"),
    ("boundary_no_symptom", "我准备明年种番茄，现在还没播种，也没有病害症状，先说说整地原则。"),
    ("boundary_cancel", "前面的补贴查询取消，不用继续，也不要联网搜索。"),
    ("boundary_meta", "你为什么有时需要追问城市或地区？只解释原因，不要真的查询。"),
]


def build_scenarios() -> list[dict]:
    scenarios: list[dict] = []
    index = 0
    for family in FAMILIES:
        for interaction in INTERACTIONS:
            for style in STYLES:
                scenarios.append(_scenario(family, interaction, style, index))
                index += 1

    for scenario_id, user in BOUNDARY_SCENARIOS:
        scenarios.append({
            "id": scenario_id,
            "category": "boundary_no_action",
            "interaction": "single_turn",
            "style": "handwritten",
            "source": "handwritten_boundary_case",
            "provenance": {
                "facts": "business_boundary",
                "utterance": "handwritten",
                "online_user_traffic": False,
            },
            "turns": [{"user": user}],
            "goal": {
                "context": {},
                "required_domains": [],
                "evidence_any": {},
                "no_action": True,
            },
        })

    if len(scenarios) != 120:
        raise AssertionError(f"V2 挑战集应为 120 项，实际 {len(scenarios)}")
    ids = [item["id"] for item in scenarios]
    if len(ids) != len(set(ids)):
        raise AssertionError("V2 挑战集存在重复 ID")
    return scenarios


def _canonical_hash(scenarios: list[dict]) -> str:
    payload = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in scenarios
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    dataset_path = DATA_DIR / "outcome_challenge_v2.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in scenarios) + "\n",
        encoding="utf-8",
    )

    categories = Counter(item["category"] for item in scenarios)
    interactions = Counter(item["interaction"] for item in scenarios)
    styles = Counter(item["style"] for item in scenarios)
    manifest = {
        "version": 2,
        "tasks": len(scenarios),
        "turns": sum(len(item["turns"]) for item in scenarios),
        "construction": (
            "7种能力组合×4种交互状态×4种表达风格，加8项边界场景；"
            "事实来自仓库知识/政策和冻结天气，问题为独立合成改写，不代表线上流量"
        ),
        "primary_scoring": (
            "最终上下文、业务能力执行、可核验证据与非错误输出；内部route仅记录不计主成功"
        ),
        "categories": dict(sorted(categories.items())),
        "interactions": dict(sorted(interactions.items())),
        "styles": dict(sorted(styles.items())),
        "sha256": _canonical_hash(scenarios),
    }
    manifest_path = DATA_DIR / "outcome_challenge_v2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(dataset_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
