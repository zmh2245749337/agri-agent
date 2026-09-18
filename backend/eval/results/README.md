本目录保存可审计的评测产物：

- `e2e_agent_report_main.md`：140 个农业场景、164 轮对话的端到端汇总与失败样本。
- `e2e_agent_main.json`：端到端评测逐任务原始记录，用于复核指标。
- `outcome_agent_report_v2_final.md`：V2最终结果导向汇总与9项失败样本，业务结果成功率为92.50%。
- `outcome_agent_v2_final.json`：V2最终逐场景状态、业务能力、证据和原始轨迹记录。
- 运行 `python eval/diagnosis_retrieval_eval.py` 后还会生成 `diagnosis_retrieval_report.md`，用于比较 RapidFuzz 模糊匹配与长上下文推理。

端到端报告由真实模型与高德 MCP 调用生成，但场景数据来自仓库知识槽位与固定模板，不代表线上用户流量，也不评价最终回答的农业专业质量。

V2默认用冻结天气响应替代实时MCP数据，主要指标不要求固定内部路由，而是同时检查最终上下文、业务能力执行、可核验证据、边界约束和非错误输出。V2同样是合成挑战集，不应解释成线上任务成功率。
