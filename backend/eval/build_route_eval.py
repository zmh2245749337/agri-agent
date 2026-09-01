# backend/eval/build_route_eval.py
"""
构建统一主图「路由准确率」评测集。

为什么要有这个评测集
--------------------
AgriGraph 的第一步是路由器：把用户这句话分到 direct / react / planning / clarify
四条路线之一，并提取需要的农业能力。这一步错了，后面工具调用、并发编排、追问
全都跟着错，所以它是整张图里最值得单独量化的环节。原来的 eval 只覆盖了诊断检索
（14 条人工用例），路由这一层一直没有数字。

标注怎么来（重要，面试会被问）
------------------------------
不是手工标 200 条。做法是「模板 × 真实槽位值」程序化生成：
- 作物名、症状关键词直接取自 data/pest_knowledge/knowledge.json 的真实条目；
- 省级地区取自 data/policies/policies.json 的真实 region 字段；
- 城市名用常见城市级地名（路由器要求 city 只能是城市级，省级只能进 region）。
模板决定了这条样本的正确路由，所以标签在构造时就确定，不需要事后人工标注。

这跟 CAIL2018、CrossWOZ 那类「用公开数据集自带标注」的思路是一回事：
样本量要上去，标注就不能靠手工。代价是模板生成的句子比真实用户提问规整，
指标会偏乐观——所以另外手写了一个 challenge 难例集（见下），两个分开报告，
不混在一起充数。

标签依据
--------
标签严格按 agri_graph.py 里 ROUTER_PROMPT 自己写的规则定，不按我们的主观理解：
- direct：寒暄、能力介绍、一般农艺知识、对上一条回答的追问
- react：单一农业能力查询
- planning：明确要计划/方案/安排，或同时需要 ≥2 个能力
- clarify：用户意图本身无法判断（注意：字段缺失不算 clarify，
           ROUTER_PROMPT 明确写了「字段是否齐全最终会由程序再次校验」）

已知局限（要主动交代，不要等被问）
----------------------------------
clarify 是四类里定义最模糊的一类。「帮我看看」「这个怎么办」这种句子判成
clarify 还是 direct，本身就有争议空间——模型判成 direct 也不能算错得离谱。
所以 clarify 这一类的准确率参考价值低于其他三类，报告时应该分类别看，
不要只报一个总数。这里不为了让指标好看去改标签：标签一旦跟着结果调，
整套评测就失去意义了。
如果标签依据的是我们自己想当然的规则，测出来的就不是路由器准不准，
而是路由器同不同意我们的误解。

用法
----
    cd backend
    python eval/build_route_eval.py

输出：
    eval/data/route_eval.jsonl            主评测集（模板生成）
    eval/data/route_eval_challenge.jsonl  难例集（人工手写）
    eval/data/route_eval_manifest.json    构造说明与分布
"""
import json
import random
from pathlib import Path

SEED = 42
BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"

# 城市级地名。路由器规则里明确要求 city 只能填城市级，省级只能进 region，
# 所以天气类样本必须用城市名，不能用省名——这条本身也是个易错点。
# 省 → 省会。planning 类样本同时出现省和市，必须地理上对得上，
# 否则会生成「河南省南宁」「湖北省杭州」这种不存在的组合，句子不自然、
# 也会给路由器制造本不该有的噪声。
PROVINCE_CAPITAL = {
    "湖南省": "长沙", "广东省": "广州", "江苏省": "南京", "四川省": "成都",
    "湖北省": "武汉", "河南省": "郑州", "山东省": "济南", "安徽省": "合肥",
    "江西省": "南昌", "陕西省": "西安", "云南省": "昆明", "黑龙江省": "哈尔滨",
    "辽宁省": "沈阳", "福建省": "福州", "贵州省": "贵阳", "山西省": "太原",
    "甘肃省": "兰州", "吉林省": "长春", "海南省": "海口",
    "广西壮族自治区": "南宁", "内蒙古自治区": "呼和浩特", "新疆维吾尔自治区": "乌鲁木齐",
}
CITIES = sorted(set(PROVINCE_CAPITAL.values()))

WEATHER_ASKS = [
    "明天会下雨吗", "这几天天气怎么样", "明天适合打药吗", "最近有没有大风",
    "后天气温多少", "这周适合灌溉吗", "明天能不能收割", "接下来三天有雨吗",
]

POLICY_ASKS = [
    "有什么种植补贴", "能申请哪些补贴", "补贴怎么申请", "有没有扶持政策",
    "农机购置补贴怎么算", "种粮补贴的条件是什么",
]

PLAN_ASKS = [
    "帮我做个接下来的安排", "给我一份种植方案", "帮我规划一下这几天该干什么",
    "能不能出个完整的处理计划", "帮我理一下下一步怎么做",
]

# direct：寒暄、能力介绍、一般农艺知识。
# 注意「现在适不适合播种」按 ROUTER_PROMPT 属于农艺问题，不是天气查询——
# 这是刻意放进主集的一个易混样本。
DIRECT_TEMPLATES = [
    "你好", "谢谢你", "辛苦了", "好的，我知道了", "再见",
    "你能帮我做什么", "你都有哪些功能", "你是怎么工作的",
    "{crop}一般什么时候播种比较好", "{crop}亩产大概多少",
    "{crop}和玉米哪个更耐旱", "{crop}适合什么样的土壤",
    "现在适合种{crop}吗", "{crop}的生长周期是多久",
    "{crop}一亩地大概要多少种子", "{crop}怎么施肥比较合理",
    "{crop}和水稻能不能轮作", "新手种{crop}要注意什么",
]

# clarify：意图本身无法判断，不是缺字段。
CLARIFY_TEMPLATES = [
    "帮我看看", "这个怎么办", "有点问题", "我想问点事",
    "麻烦了", "看一下呢", "你帮我处理一下", "这样行不行",
    "我这边有点情况", "能不能帮个忙", "有个事想请教", "你看看呢",
    "帮忙分析下", "这情况正常吗", "该怎么弄", "我拿不准",
    "想咨询一下", "有点搞不懂",
]


def load_pest_pairs():
    """从真实病虫害知识库取 (作物, 症状关键词) 组合。"""
    items = json.loads((BACKEND / "data/pest_knowledge/knowledge.json").read_text(encoding="utf-8"))
    pairs = []
    for it in items:
        crop = it.get("crop")
        kws = it.get("keywords") or []
        if not crop or crop == "通用" or not kws:
            continue
        pairs.append((crop, kws[0]))
    # 同作物同症状去重
    return sorted(set(pairs))


def load_regions_crops():
    """从真实政策库取省级地区与作物。"""
    items = json.loads((BACKEND / "data/policies/policies.json").read_text(encoding="utf-8"))
    regions, crops = set(), set()
    for it in items:
        r = (it.get("region") or "").strip()
        # 只要省级：跳过「全国」和带市/区的细粒度条目
        if r and r != "全国" and "市" not in r and "/" not in r:
            regions.add(r)
        c = (it.get("crop") or "").strip()
        for part in c.replace("（", "/").replace("）", "").split("/"):
            part = part.strip()
            if part and part != "不限" and len(part) <= 4:
                crops.add(part)
    return sorted(regions), sorted(crops)


def build_main(rng):
    pest_pairs = load_pest_pairs()
    regions, policy_crops = load_regions_crops()
    all_crops = sorted({c for c, _ in pest_pairs} | set(policy_crops))
    rows = []

    def add(query, route, caps, kind):
        rows.append({
            "query": query,
            "expect_route": route,
            "expect_capabilities": caps,
            "kind": kind,
            "source": "template",
        })

    # ---- direct ----
    for t in DIRECT_TEMPLATES:
        if "{crop}" in t:
            for crop in rng.sample(all_crops, min(6, len(all_crops))):
                add(t.format(crop=crop), "direct", [], "direct_agronomy")
        else:
            add(t, "direct", [], "direct_smalltalk")

    # ---- react / diagnosis ----
    diag_forms = [
        "我家的{crop}{sym}，是怎么回事",
        "{crop}{sym}，要怎么处理",
        "种的{crop}最近{sym}，帮我看看什么原因",
    ]
    for crop, sym in rng.sample(pest_pairs, min(9, len(pest_pairs))):
        for f in diag_forms:
            add(f.format(crop=crop, sym=sym), "react", ["diagnosis"], "react_diagnosis")

    # ---- react / weather ----
    for city in rng.sample(CITIES, 12):
        for ask in rng.sample(WEATHER_ASKS, 2):
            add(f"{city}{ask}", "react", ["weather"], "react_weather")

    # ---- react / policy ----
    for region in rng.sample(regions, min(10, len(regions))):
        crop = rng.choice(policy_crops)
        for ask in rng.sample(POLICY_ASKS, 2):
            add(f"我在{region}种{crop}，{ask}", "react", ["policy"], "react_policy")

    # ---- planning：显式要方案，或同时需要两种以上能力 ----
    for _ in range(18):
        crop, sym = rng.choice(pest_pairs)
        region = rng.choice([r for r in regions if r in PROVINCE_CAPITAL] or regions)
        city = PROVINCE_CAPITAL.get(region, "长沙")
        add(f"我在{region}{city}种{crop}，最近{sym}，{rng.choice(PLAN_ASKS)}",
            "planning", ["diagnosis", "weather", "policy"], "planning_explicit")
    for _ in range(18):
        crop, sym = rng.choice(pest_pairs)
        city = rng.choice(CITIES)
        add(f"我的{crop}{sym}，{city}明天的天气也帮我看一下",
            "planning", ["diagnosis", "weather"], "planning_multi_capability")

    # ---- clarify ----
    for t in CLARIFY_TEMPLATES:
        add(t, "clarify", [], "clarify_vague")
    for q in ["嗯……", "在吗", "有人吗", "我这个情况", "说来话长",
              "刚遇到个事", "想问问", "在忙吗", "打扰一下", "有空吗"]:
        add(q, "clarify", [], "clarify_vague")

    # 去重 + 打散
    seen, uniq = set(), []
    for r in rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        uniq.append(r)
    rng.shuffle(uniq)
    for i, r in enumerate(uniq, 1):
        r["id"] = f"main_{i:04d}"
    return uniq


def build_challenge():
    """
    人工手写难例集。

    ⚠️ 标签修正说明（2026-08-27，在首次跑出结果之后做的修正，必须披露）
    ------------------------------------------------------------------
    首版把「江苏明天天气怎么样」这类只给了省级地名的样本标成了 react，跑出来
    react 一类 0/7。复核后确认是**标注错误，不是模型判错**：

      本评测测的是 _route_node 的最终路由，而最终路由 = 路由器判断 + 程序字段校验。
      _REQUIRED_FIELDS 规定 weather 必须有城市级 city（省级会被 _is_province_only
      挡掉）、diagnosis 必须有 crop+symptom、policy 必须有 crop+region；
      任一必填字段缺失时，agri_graph.py 会把最终路由强制改成 clarify。

    也就是说「只给了省、没给市」的天气请求，系统判 clarify（追问城市）是**正确行为**。
    首版标签只看「用户想问什么能力」，忽略了字段校验这一步，属于标注规则本身写错。

    修正方式是按规则统一重标，不是逐条挑着改：
      期望能力所需的必填字段在文本或历史里齐全 → react / planning
      任一必填字段缺失                          → clarify
    每条样本用 provides 字段显式写出它提供了哪些槽位，标签由规则推导，可复核。

    这条修正发生在看到结果之后，所以在这里写明；报告和面试中都应主动交代，
    不能当作从一开始就是这么标的。除此之外没有为了指标好看而改动任何标签。
    """
    def derive(caps, provides, explicit_plan=False):
        """按 _REQUIRED_FIELDS 规则推导最终期望路由。"""
        required = {"diagnosis": ("crop", "symptom"),
                    "weather": ("city",),
                    "policy": ("crop", "region")}
        if not caps:
            return "direct"
        missing = [f for c in caps for f in required[c] if f not in provides]
        if missing:
            return "clarify"
        if explicit_plan or len(caps) >= 2:
            return "planning"
        return "react"

    def C(q, caps, kind, provides=(), hist=None, explicit_plan=False, force_route=None):
        return {"query": q,
                "expect_capabilities": list(caps),
                "expect_route": force_route or derive(caps, set(provides), explicit_plan),
                "provides": list(provides),
                "kind": kind, "source": "handwritten", "history": hist or []}

    rows = [
        # 指代承接：单看这一句不完整，必须结合上文，ROUTER_PROMPT 要求不能因此判 clarify
        C("那具体怎么做", [], "challenge_anaphora", hist=[
            {"role": "user", "content": "我的水稻叶子发黄怎么办"},
            {"role": "assistant", "content": "可能是缺氮肥或纹枯病，建议先检查田间排水。"}]),
        C("为什么是这个原因", [], "challenge_anaphora", hist=[
            {"role": "user", "content": "我的小麦叶片有橙黄色条状锈斑"},
            {"role": "assistant", "content": "这是典型的条锈病症状。"}]),
        C("刚才那句什么意思", [], "challenge_anaphora", hist=[
            {"role": "user", "content": "长沙明天适合打药吗"},
            {"role": "assistant", "content": "明天有阵雨，不建议打药。"}]),
        C("还没播种", [], "challenge_slot_answer", hist=[
            {"role": "user", "content": "我想种水稻"},
            {"role": "assistant", "content": "请问现在处于什么生长阶段？"}]),
        C("湖南省", ["policy"], "challenge_slot_answer", provides=("crop", "region"), hist=[
            {"role": "user", "content": "种水稻有什么补贴"},
            {"role": "assistant", "content": "请问您在哪个省？"}]),
        # 只给省级地名：weather 必填 city 缺失 → 系统正确地追问城市
        C("江苏明天天气怎么样", ["weather"], "challenge_province_as_city"),
        C("湖南这几天有雨吗", ["weather"], "challenge_province_as_city"),
        C("广东省下周适合收割吗", ["weather"], "challenge_province_as_city"),
        # 农时陷阱：不是天气查询
        C("现在适合播种玉米吗", [], "challenge_agronomy_not_weather"),
        C("这个时候种花生晚不晚", [], "challenge_agronomy_not_weather"),
        C("水稻现在能移栽了吗", [], "challenge_agronomy_not_weather"),
        # 隐含多能力，没说「方案」，且槽位齐全
        C("我的番茄叶子有黄斑，广州最近的天气会不会加重", ["diagnosis", "weather"],
          "challenge_implicit_multi", provides=("crop", "symptom", "city")),
        C("河南种小麦能领补贴吗，郑州这两天下不下雨", ["policy", "weather"],
          "challenge_implicit_multi", provides=("crop", "region", "city")),
        C("柑橘落果厉害，昆明天气和云南的补贴一起帮我看看",
          ["diagnosis", "weather", "policy"], "challenge_implicit_multi",
          provides=("crop", "symptom", "city", "region")),
        # 结束语，不该触发任何能力
        C("好的谢谢，先这样", [], "challenge_closing", hist=[
            {"role": "user", "content": "我在湖南种水稻有什么补贴"},
            {"role": "assistant", "content": "湖南有耕地地力保护补贴等。"}]),
        C("明白了，辛苦", [], "challenge_closing", hist=[
            {"role": "user", "content": "长沙明天下雨吗"},
            {"role": "assistant", "content": "明天有小雨。"}]),
        # 模糊但有农业指向：意图本身判不出
        C("我的地里有点不对劲", [], "challenge_vague_agri", force_route="clarify"),
        C("今年收成怕是不行", [], "challenge_vague_agri", force_route="clarify"),
        # 跨领域 / 超出范围
        C("帮我写一份种植合同", [], "challenge_out_of_scope"),
        C("农产品现在什么价格", [], "challenge_out_of_scope"),
        # 迂回表述：有作物但没有明确症状 → diagnosis 必填 symptom 缺失
        C("最近雨水多，我那块水稻田是不是该排水了", ["diagnosis"],
          "challenge_indirect_single", provides=("crop",)),
        # 想问天气但没给城市 → 必填 city 缺失
        C("打算下周喷药，得先看看天", ["weather"], "challenge_indirect_single"),
        # 政策：作物和省份都有
        C("听说种大豆国家给钱，我在安徽", ["policy"], "challenge_indirect_single",
          provides=("crop", "region")),
        # 显式要方案且槽位齐全
        C("给我一份水稻纹枯病的处理方案", ["diagnosis"], "challenge_plan_single_cap",
          provides=("crop", "symptom"), explicit_plan=True),
        # 否定 / 拒绝
        C("不用查天气了", [], "challenge_negation", hist=[
            {"role": "user", "content": "我在长沙种水稻"},
            {"role": "assistant", "content": "需要我查一下长沙的天气吗？"}]),
    ]
    for i, r in enumerate(rows, 1):
        r["id"] = f"chal_{i:03d}"
    return rows


def main():
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = build_main(rng)
    chal_rows = build_challenge()

    def dump(rows, name):
        path = DATA_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    dump(main_rows, "route_eval.jsonl")
    dump(chal_rows, "route_eval_challenge.jsonl")

    def dist(rows):
        d = {}
        for r in rows:
            d[r["expect_route"]] = d.get(r["expect_route"], 0) + 1
        return dict(sorted(d.items()))

    manifest = {
        "seed": SEED,
        "main": {
            "count": len(main_rows),
            "construction": "模板 × 真实槽位值程序化生成；标签由模板决定，无人工标注",
            "slot_sources": {
                "crop/symptom": "data/pest_knowledge/knowledge.json 真实条目",
                "region": "data/policies/policies.json 真实 region 字段（省级）",
                "city": "常见城市级地名列表（路由器要求 city 必须为城市级）",
            },
            "route_distribution": dist(main_rows),
        },
        "challenge": {
            "count": len(chal_rows),
            "construction": "人工手写，覆盖指代承接、省市混淆、农时陷阱、隐含多能力、结束语、迂回表述、否定",
            "label_correction": "2026-08-27 首次跑出结果后发现首版标签规则有误："
                                "本评测测的是程序字段校验之后的最终路由，必填字段缺失时系统会强制改判 clarify，"
                                "而首版只按『用户想问什么能力』标注，把只给省级地名的天气请求标成了 react。"
                                "已按 _REQUIRED_FIELDS 规则统一重标（每条用 provides 字段显式声明所提供槽位，"
                                "标签由规则推导可复核）。此修正发生在看到结果之后，须主动披露；"
                                "除此之外未因指标好看改动任何标签。",
            "route_distribution": dist(chal_rows),
        },
        "label_basis": "严格按 agri_graph.py 中 ROUTER_PROMPT 自述的四条路由规则标注；"
                       "字段缺失不判 clarify（ROUTER_PROMPT 明确由程序另行校验）",
        "caveat": "模板生成的句子比真实用户提问规整，主集指标偏乐观；"
                  "challenge 集单独报告，不与主集合并计数",
    }
    (DATA_DIR / "route_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"main     : {len(main_rows):4d}  {dist(main_rows)}")
    print(f"challenge: {len(chal_rows):4d}  {dist(chal_rows)}")
    print(f"written to {DATA_DIR}")


if __name__ == "__main__":
    main()
