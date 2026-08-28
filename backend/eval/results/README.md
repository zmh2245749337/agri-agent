本目录保存可审计的评测产物：

- `e2e_agent_report_main.md`：140 个农业场景、164 轮对话的端到端汇总与失败样本。
- `e2e_agent_main.json`：端到端评测逐任务原始记录，用于复核指标。
- 运行 `python eval/diagnosis_retrieval_eval.py` 后还会生成 `diagnosis_retrieval_report.md`，用于比较 RapidFuzz 模糊匹配与长上下文推理。

端到端报告由真实模型与高德 MCP 调用生成，但场景数据来自仓库知识槽位与固定模板，不代表线上用户流量，也不评价最终回答的农业专业质量。
