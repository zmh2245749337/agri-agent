# AgriAgent 端到端场景评测（main）

> 通过公开 `run` 接口真实执行 LangGraph 主流程；任务由仓库知识槽位与农业场景模板构造，不代表线上用户流量。

## 总体结果

| 指标 | 结果 |
| --- | ---: |
| 场景任务数 | 140 |
| 对话轮次数 | 164 |
| 端到端任务完成率 | 98.57% |
| 单轮执行成功率 | 98.78% |
| 路由准确率 | 98.78% |
| 能力集合完全匹配率 | 99.39% |
| 状态字段准确率 | 100.00% |
| 缺参追问正确率 | 100.00% |
| 工具调用 F1 | 99.41% |
| Planning 子任务完成率 | 100.00% |
| 无动作场景误触发率 ↓ | 1.79% |
| 非错误输出率 | 100.00% |
| 平均端到端延迟 | 7.313s |
| P95 端到端延迟 | 16.207s |
| 运行时异常数 | 0 |

## 分场景结果

| 场景 | 任务数 | 任务完成率 | 单轮成功率 |
| --- | ---: | ---: | ---: |
| clarify | 16 | 100.00% | 100.00% |
| direct | 16 | 87.50% | 87.50% |
| multi_agent_all | 8 | 100.00% | 100.00% |
| multi_agent_diagnosis_weather | 8 | 100.00% | 100.00% |
| multi_agent_weather_policy | 8 | 100.00% | 100.00% |
| multi_turn_slot_diagnosis | 8 | 100.00% | 100.00% |
| multi_turn_slot_policy | 8 | 100.00% | 100.00% |
| multi_turn_slot_weather | 8 | 100.00% | 100.00% |
| single_tool_diagnosis | 20 | 100.00% | 100.00% |
| single_tool_policy | 20 | 100.00% | 100.00% |
| single_tool_weather | 20 | 100.00% | 100.00% |

## 失败样本（前 20 条）

- `direct_004`（direct）：route: expected=('direct', 'react')；capabilities: expected=([], ['policy'])；tools: expected=([], ['search_subsidy_policy'])
- `direct_010`（direct）：route: expected=('direct', 'clarify')；missing_fields: expected=([], ['request'])

## 指标边界

- 任务完成要求该任务全部轮次同时通过路由、能力、状态、工具/子 Agent 轨迹与非错误输出检查。
- 该评测衡量系统执行正确性，不使用 LLM-as-Judge 判断最终建议的农业专业质量。
- 外部模型、天气和 A2A 服务异常不会从分母中剔除。
