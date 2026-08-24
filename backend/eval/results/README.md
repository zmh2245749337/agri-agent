跑 `python eval/diagnosis_retrieval_eval.py` 之后，`diagnosis_retrieval_report.md` 会自动生成在这个目录下，内容是 RapidFuzz 模糊匹配 vs 长上下文直接推理 两种诊断检索方法的量化对比报告（命中率、耗时、分类别明细）。

这个目录里目前只放这份说明文件——报告本身依赖真实网络 + LLM API 才能生成，不在仓库里预先放一份，避免被误当成"已经跑过的真实结果"。
