"""构建 AgriAgent 端到端场景评测集。

数据集使用仓库内病虫害知识条目、政策覆盖地区和常见城市作为真实槽位来源，
再通过固定场景模板组合成可复现任务。它是受控的农业场景集，不冒充线上用户流量。
"""

from __future__ import annotations

import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

REGION_CITY = [
    ("湖南省", "长沙"),
    ("广东省", "广州"),
    ("江苏省", "南京"),
    ("四川省", "成都"),
    ("湖北省", "武汉"),
    ("河南省", "郑州"),
    ("山东省", "济南"),
    ("安徽省", "合肥"),
    ("江西省", "南昌"),
    ("陕西省", "西安"),
    ("云南省", "昆明"),
    ("黑龙江省", "哈尔滨"),
    ("辽宁省", "沈阳"),
    ("福建省", "福州"),
    ("贵州省", "贵阳"),
    ("山西省", "太原"),
    ("甘肃省", "兰州"),
    ("吉林省", "长春"),
]

DIRECT_QUERIES = [
    "你好，你能提供哪些农业帮助？",
    "水稻一般什么时候播种？",
    "玉米适合什么样的土壤？",
    "小麦和玉米轮作要注意什么？",
    "新手种番茄最容易忽略什么？",
    "花生一亩地通常需要多少种子？",
    "水稻分蘖期管理重点是什么？",
    "大豆怎么施肥比较合理？",
    "柑橘幼树日常管理要注意什么？",
    "蔬菜连作为什么容易出问题？",
    "现在适合播种玉米吗？",
    "谢谢，先到这里。",
    "水稻育秧和直播有什么区别？",
    "葡萄修剪一般要注意哪些原则？",
    "棉花生长周期大概多久？",
    "种植前为什么要做土壤检测？",
]

CLARIFY_QUERIES = [
    "帮我看看。",
    "这个怎么办？",
    "地里有点问题。",
    "这情况正常吗？",
    "你帮我分析一下。",
    "有个农业问题想请教。",
    "最近感觉不太对。",
    "能不能帮我处理一下？",
    "我拿不准该怎么办。",
    "这两天有点麻烦。",
    "你看看需要怎么弄。",
    "我这里出了点状况。",
    "不知道下一步怎么办。",
    "帮忙判断一下。",
    "有些异常，但说不清楚。",
    "想请你给点建议。",
]

WEATHER_ASKS = [
    "明天适合打药吗？",
    "未来三天会下雨吗？",
    "这周适合灌溉吗？",
    "后天能不能收割？",
    "最近有没有大风天气？",
    "明天气温大概怎么样？",
]

POLICY_CROPS = ["水稻", "小麦", "玉米", "大豆", "花生", "油菜"]
POLICY_ASKS = [
    "有哪些种植补贴？",
    "补贴申请条件是什么？",
    "想了解今年的农业扶持政策。",
    "耕地地力保护补贴怎么申请？",
]


def _load_diagnosis_pairs() -> list[tuple[str, str]]:
    rows = json.loads(
        (BACKEND / "data/pest_knowledge/knowledge.json").read_text(encoding="utf-8")
    )
    pairs: list[tuple[str, str]] = []
    for row in rows:
        crop = str(row.get("crop") or "").strip()
        keywords = row.get("keywords") or []
        if not crop or crop == "通用" or not keywords:
            continue
        pair = (crop, str(keywords[0]).strip())
        if pair not in pairs:
            pairs.append(pair)
    if len(pairs) < 20:
        raise RuntimeError("病虫害知识库不足 20 组可用作物/症状")
    return pairs


def _expect(
    route: str,
    capabilities: list[str] | None = None,
    *,
    context: dict | None = None,
    missing_fields: list[str] | None = None,
    tools: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict:
    return {
        "route": route,
        "capabilities": capabilities or [],
        "context": context or {},
        "missing_fields": missing_fields or [],
        "tools": tools or [],
        "tasks": tasks or [],
    }


def _single(task_id: str, category: str, user: str, expect: dict) -> dict:
    return {
        "id": task_id,
        "category": category,
        "source": "repository_knowledge_and_scenario_template",
        "turns": [{"user": user, "expect": expect}],
    }


def build_tasks() -> list[dict]:
    diagnosis_pairs = _load_diagnosis_pairs()
    tasks: list[dict] = []

    for idx, query in enumerate(DIRECT_QUERIES, 1):
        tasks.append(_single(
            f"direct_{idx:03d}",
            "direct",
            query,
            _expect("direct"),
        ))

    for idx, query in enumerate(CLARIFY_QUERIES, 1):
        tasks.append(_single(
            f"clarify_{idx:03d}",
            "clarify",
            query,
            _expect("clarify", missing_fields=["request"]),
        ))

    diagnosis_forms = [
        "我种的{crop}最近{symptom}，帮我查一下是什么问题。",
        "{crop}{symptom}，应该怎么处理？",
        "田里的{crop}出现{symptom}，请帮我诊断。",
    ]
    for idx, (crop, symptom) in enumerate(diagnosis_pairs[:20], 1):
        query = diagnosis_forms[(idx - 1) % len(diagnosis_forms)].format(
            crop=crop, symptom=symptom
        )
        tasks.append(_single(
            f"diagnosis_{idx:03d}",
            "single_tool_diagnosis",
            query,
            _expect(
                "react",
                ["diagnosis"],
                context={"crop": crop},
                tools=["diagnose_crop_disease"],
            ),
        ))

    for idx in range(20):
        region, city = REGION_CITY[idx % len(REGION_CITY)]
        ask = WEATHER_ASKS[(idx + idx // len(REGION_CITY)) % len(WEATHER_ASKS)]
        tasks.append(_single(
            f"weather_{idx + 1:03d}",
            "single_tool_weather",
            f"我在{city}种地，{ask}",
            _expect(
                "react",
                ["weather"],
                context={"city": city},
                tools=["get_weather_forecast"],
            ),
        ))

    for idx in range(20):
        region, _ = REGION_CITY[idx % len(REGION_CITY)]
        crop = POLICY_CROPS[idx % len(POLICY_CROPS)]
        ask = POLICY_ASKS[idx % len(POLICY_ASKS)]
        tasks.append(_single(
            f"policy_{idx + 1:03d}",
            "single_tool_policy",
            f"我在{region}种{crop}，{ask}",
            _expect(
                "react",
                ["policy"],
                context={"crop": crop, "region": region},
                tools=["search_subsidy_policy"],
            ),
        ))

    # 24 个复合任务：诊断+天气、天气+政策、诊断+天气+政策各 8 个。
    for idx in range(8):
        crop, symptom = diagnosis_pairs[20 + idx]
        region, city = REGION_CITY[idx]
        tasks.append(_single(
            f"planning_dw_{idx + 1:03d}",
            "multi_agent_diagnosis_weather",
            f"我在{city}种的{crop}{symptom}，再看看明天是否适合打药，给我一个处理安排。",
            _expect(
                "planning",
                ["diagnosis", "weather"],
                context={"crop": crop, "city": city},
                tasks=["diagnosis", "weather"],
            ),
        ))

    for idx in range(8):
        crop = POLICY_CROPS[idx % len(POLICY_CROPS)]
        region, city = REGION_CITY[8 + idx]
        tasks.append(_single(
            f"planning_wp_{idx + 1:03d}",
            "multi_agent_weather_policy",
            f"我在{region}{city}种{crop}，帮我同时查近期天气和可申请的补贴，整理成行动建议。",
            _expect(
                "planning",
                ["weather", "policy"],
                context={"crop": crop, "city": city, "region": region},
                tasks=["weather", "policy"],
            ),
        ))

    for idx in range(8):
        crop, symptom = diagnosis_pairs[28 + idx]
        region, city = REGION_CITY[(idx + 4) % len(REGION_CITY)]
        tasks.append(_single(
            f"planning_all_{idx + 1:03d}",
            "multi_agent_all",
            f"我在{region}{city}种{crop}，现在{symptom}。请结合病害、天气和补贴给一份完整计划。",
            _expect(
                "planning",
                ["diagnosis", "weather", "policy"],
                context={"crop": crop, "city": city, "region": region},
                tasks=["diagnosis", "weather", "policy"],
            ),
        ))

    # 24 个两轮任务，验证缺参追问、状态保存与补槽后继续执行。
    for idx in range(8):
        crop, symptom = diagnosis_pairs[36 + idx]
        tasks.append({
            "id": f"multiturn_diagnosis_{idx + 1:03d}",
            "category": "multi_turn_slot_diagnosis",
            "source": "repository_knowledge_and_scenario_template",
            "turns": [
                {
                    "user": f"叶片出现{symptom}，应该怎么办？",
                    "expect": _expect(
                        "clarify", ["diagnosis"], missing_fields=["crop"]
                    ),
                },
                {
                    "user": crop,
                    "expect": _expect(
                        "react",
                        ["diagnosis"],
                        context={"crop": crop},
                        tools=["diagnose_crop_disease"],
                    ),
                },
            ],
        })

    for idx in range(8):
        region, city = REGION_CITY[idx]
        tasks.append({
            "id": f"multiturn_weather_{idx + 1:03d}",
            "category": "multi_turn_slot_weather",
            "source": "scenario_template",
            "turns": [
                {
                    "user": "明天适合打药吗？",
                    "expect": _expect(
                        "clarify", ["weather"], missing_fields=["city"]
                    ),
                },
                {
                    "user": city,
                    "expect": _expect(
                        "react",
                        ["weather"],
                        context={"city": city},
                        tools=["get_weather_forecast"],
                    ),
                },
            ],
        })

    for idx in range(8):
        crop = POLICY_CROPS[idx % len(POLICY_CROPS)]
        region, _ = REGION_CITY[idx + 8]
        tasks.append({
            "id": f"multiturn_policy_{idx + 1:03d}",
            "category": "multi_turn_slot_policy",
            "source": "scenario_template",
            "turns": [
                {
                    "user": f"种{crop}能申请哪些补贴？",
                    "expect": _expect(
                        "clarify",
                        ["policy"],
                        context={"crop": crop},
                        missing_fields=["region"],
                    ),
                },
                {
                    "user": region,
                    "expect": _expect(
                        "react",
                        ["policy"],
                        context={"crop": crop, "region": region},
                        tools=["search_subsidy_policy"],
                    ),
                },
            ],
        })

    if len(tasks) != 140:
        raise AssertionError(f"端到端任务数应为 140，实际为 {len(tasks)}")
    return tasks


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    dataset_path = DATA_DIR / "e2e_tasks.jsonl"
    with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    distribution: dict[str, int] = {}
    turns = 0
    for task in tasks:
        distribution[task["category"]] = distribution.get(task["category"], 0) + 1
        turns += len(task["turns"])

    manifest = {
        "version": 1,
        "tasks": len(tasks),
        "turns": turns,
        "construction": "仓库知识库真实槽位与固定农业场景模板组合；不代表线上用户流量",
        "coverage": distribution,
        "scoring_unit": "一个任务的全部轮次同时满足路由、能力、状态、工具/子Agent轨迹和非错误输出才算成功",
    }
    (DATA_DIR / "e2e_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"written: {dataset_path}")


if __name__ == "__main__":
    main()
