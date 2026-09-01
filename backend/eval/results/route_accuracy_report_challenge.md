# 统一主图路由准确率评测（challenge）

> 只评测路由器本身，不触发下游工具；每条样本一次模型调用。

## 总体

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 25 |
| 路由准确率（含兜底样本） | 56.00% |
| 路由准确率（仅模型成功返回的 20 条） | 60.00% |
| 能力识别 F1 | 78.57% |
| 能力集合完全匹配率 | 76.00% |
| 无能力轮误触发率 ↓ | 14.29% （2/14） |
| 保守兜底触发次数 | 5 |
| 异常样本数 | 0 |
| 平均单次路由耗时 | 2.821s |

## 分类别准确率

| 路由 | 样本数 | 准确率 |
| --- | ---: | ---: |
| direct | 12 | 83.33% |
| react | 7 | 0.00% |
| planning | 4 | 75.00% |
| clarify | 2 | 50.00% |

## 混淆矩阵（行=期望，列=预测）

| 期望 \ 预测 | direct | react | planning | clarify |
| --- | ---: | ---: | ---: | ---: |
| direct | 10 | 1 | 0 | 1 |
| react | 3 | 0 | 0 | 4 |
| planning | 0 | 1 | 3 | 0 |
| clarify | 1 | 0 | 0 | 1 |

## 典型错误样例

- `chal_005`（challenge_slot_answer）「湖南省」期望 **react**，预测 **direct**
- `chal_006`（challenge_province_as_city）「江苏明天天气怎么样」期望 **react**，预测 **clarify**
- `chal_007`（challenge_province_as_city）「湖南这几天有雨吗」期望 **react**，预测 **clarify**
- `chal_008`（challenge_province_as_city）「广东省下周适合收割吗」期望 **react**，预测 **clarify**
- `chal_018`（challenge_vague_agri）「今年收成怕是不行」期望 **clarify**，预测 **direct**
- `chal_019`（challenge_out_of_scope）「帮我写一份种植合同」期望 **direct**，预测 **clarify**
- `chal_021`（challenge_indirect_single）「最近雨水多，我那块水稻田是不是该排水了」期望 **react**，预测 **direct**
- `chal_022`（challenge_indirect_single）「打算下周喷药，得先看看天」期望 **react**，预测 **clarify**
- `chal_023`（challenge_indirect_single）「听说种大豆国家给钱，我在安徽」期望 **react**，预测 **direct**
- `chal_024`（challenge_plan_single_cap）「给我一份水稻纹枯病的处理方案」期望 **planning**，预测 **react**
- `chal_025`（challenge_negation）「不用查天气了」期望 **direct**，预测 **react**
