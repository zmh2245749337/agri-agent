# AgriAgent 结果导向挑战评测（v2_final）

> 评测最终业务状态、能力执行和可核验证据，不要求固定内部路由；场景为合成挑战集，不代表线上流量。

- 数据集 SHA-256：`ba461e030de2232cf34283267b449b04e224ad15755bac5197b742ed6434e1e0`
- 场景任务：120 项；对话：204 轮

## 总体结果

| 指标 | 结果 |
| --- | ---: |
| 业务结果成功率 | 92.50% |
| 缺参检查点通过率 | 89.29% |
| 最终上下文正确率 | 100.00% |
| 所需业务能力完成率 | 100.00% |
| 可核验证据通过率 | 92.50% |
| 边界场景无误调用率 | 100.00% |
| 非错误输出率 | 100.00% |
| 平均单轮延迟 | 23.216s |
| P95 单轮延迟 | 94.189s |
| 运行时异常数 | 0 |

## 按业务类别

| 分组 | 任务数 | 成功率 |
| --- | ---: | ---: |
| boundary_no_action | 8 | 100.00% |
| diagnosis | 16 | 93.75% |
| diagnosis_policy | 16 | 93.75% |
| diagnosis_weather | 16 | 93.75% |
| diagnosis_weather_policy | 16 | 75.00% |
| policy | 16 | 100.00% |
| weather | 16 | 100.00% |
| weather_policy | 16 | 87.50% |

## 按交互状态

| 分组 | 任务数 | 成功率 |
| --- | ---: | ---: |
| carryover | 28 | 100.00% |
| complete | 28 | 89.29% |
| correction | 28 | 96.43% |
| missing | 28 | 82.14% |
| single_turn | 8 | 100.00% |

## 按表达风格

| 分组 | 任务数 | 成功率 |
| --- | ---: | ---: |
| colloquial | 28 | 89.29% |
| handwritten | 8 | 100.00% |
| implicit | 28 | 100.00% |
| noisy | 28 | 82.14% |
| standard | 28 | 96.43% |

## 失败样本（前30项）

- `v2_diagnosis_complete_colloquial`：diagnosis 缺少可核验证据，候选=['二化螟']
- `v2_diagnosis_weather_missing_noisy`：diagnosis 缺少可核验证据，候选=['大斑病']
- `v2_diagnosis_policy_missing_colloquial`：diagnosis 缺少可核验证据，候选=['二化螟']
- `v2_weather_policy_missing_standard`：weather 缺少可核验证据，候选=['乌鲁木齐']；缺参追问：expected contains=['city'], actual=[], pending=None；缺参前提前执行业务能力：['policy', 'weather']
- `v2_weather_policy_missing_noisy`：weather 缺少可核验证据，候选=['沈阳']；缺参追问：expected contains=['city'], actual=[], pending=None；缺参前提前执行业务能力：['policy', 'weather']
- `v2_diagnosis_weather_policy_complete_colloquial`：diagnosis 缺少可核验证据，候选=['二化螟']
- `v2_diagnosis_weather_policy_complete_noisy`：policy 缺少可核验证据，候选=['nynct.hlj.gov.cn', '报废更新']
- `v2_diagnosis_weather_policy_missing_noisy`：weather 缺少可核验证据，候选=['长沙']；缺参追问：expected contains=['city'], actual=[], pending=None；缺参前提前执行业务能力：['diagnosis', 'policy', 'weather']
- `v2_diagnosis_weather_policy_correction_noisy`：diagnosis 缺少可核验证据，候选=['大斑病']

## 指标边界

- 业务结果成功要求最终上下文、所需能力执行、可核验证据、边界约束和非错误输出同时通过。
- route、节点名称和工具顺序只保存在明细中，不作为主成功条件。
- 天气默认使用冻结响应；政策与诊断使用仓库中的真实本地知识数据。
- 不使用 LLM-as-Judge 自动评价自由文本建议的农业专业水平。
