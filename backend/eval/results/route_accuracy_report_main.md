# 统一主图路由准确率评测（main）

> 只评测路由器本身，不触发下游工具；每条样本一次模型调用。

## 总体

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 203 |
| 路由准确率（含兜底样本） | 83.74% |
| 路由准确率（仅模型成功返回的 179 条） | 82.68% |
| 能力识别 F1 | 87.80% |
| 能力集合完全匹配率 | 91.13% |
| 无能力轮误触发率 ↓ | 0.00% （0/96） |
| 保守兜底触发次数 | 24 |
| 异常样本数 | 0 |
| 平均单次路由耗时 | 1.957s |

## 分类别准确率

| 路由 | 样本数 | 准确率 |
| --- | ---: | ---: |
| direct | 68 | 100.00% |
| react | 71 | 90.14% |
| planning | 36 | 52.78% |
| clarify | 28 | 67.86% |

## 混淆矩阵（行=期望，列=预测）

| 期望 \ 预测 | direct | react | planning | clarify |
| --- | ---: | ---: | ---: | ---: |
| direct | 68 | 0 | 0 | 0 |
| react | 0 | 64 | 0 | 7 |
| planning | 0 | 15 | 19 | 2 |
| clarify | 9 | 0 | 0 | 19 |

## 典型错误样例

- `main_0001`（clarify_vague）「有点搞不懂」期望 **clarify**，预测 **direct**
- `main_0003`（react_diagnosis）「我家的茄子叶片自下而上发黄萎蔫，是怎么回事」期望 **react**，预测 **clarify**
- `main_0010`（clarify_vague）「这样行不行」期望 **clarify**，预测 **direct**
- `main_0019`（planning_explicit）「我在广东省广州种柑橘，最近叶片和果实有木栓状隆起斑，帮我理一下下一步怎么做」期望 **planning**，预测 **react**
- `main_0021`（planning_explicit）「我在甘肃省兰州种水稻，最近叶片有褐色斑点，帮我理一下下一步怎么做」期望 **planning**，预测 **react**
- `main_0028`（planning_explicit）「我在辽宁省沈阳种水稻，最近秧苗徒长细高，帮我理一下下一步怎么做」期望 **planning**，预测 **react**
- `main_0034`（planning_explicit）「我在黑龙江省哈尔滨种茶叶，最近叶片有水泡状凸起，能不能出个完整的处理计划」期望 **planning**，预测 **react**
- `main_0037`（clarify_vague）「在忙吗」期望 **clarify**，预测 **direct**
- `main_0059`（planning_explicit）「我在湖北省武汉种白菜，最近叶片正面黄色病斑，给我一份种植方案」期望 **planning**，预测 **react**
- `main_0065`（clarify_vague）「有空吗」期望 **clarify**，预测 **direct**
- `main_0068`（planning_explicit）「我在山东省济南种香蕉，最近叶片自下而上黄化萎蔫，帮我理一下下一步怎么做」期望 **planning**，预测 **react**
- `main_0069`（clarify_vague）「打扰一下」期望 **clarify**，预测 **direct**
- `main_0082`（react_diagnosis）「我家的黄瓜叶片正面有黄色多角斑，是怎么回事」期望 **react**，预测 **clarify**
- `main_0087`（clarify_vague）「我拿不准」期望 **clarify**，预测 **direct**
- `main_0097`（planning_explicit）「我在新疆维吾尔自治区乌鲁木齐种茄子，最近叶片自下而上发黄萎蔫，给我一份种植方案」期望 **planning**，预测 **react**
