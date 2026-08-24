# src/agri_agent/agents/diagnosis_agent.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from agri_agent.core.my_llm import MyLLM
from agri_agent.core.my_agent import MyAgent
from agri_agent.tools.pest_knowledge_tool import match_pest_knowledge


class CropDiagnosisAgent:
    def __init__(self, llm=None):
        self.llm = llm or MyLLM()
        self.agent = MyAgent("CropDiagnosisAgent", self.llm)

    def run(self, crop: str, symptom_text: str) -> str:
        matched = match_pest_knowledge(crop, symptom_text)

        if not matched:
            knowledge_text = "本地知识库未匹配到相关记录，请基于常识给出通用建议，并提示用户描述更详细的症状。"
        else:
            lines = []
            for entry in matched:
                lines.append(
                    f"可能病因：{'、'.join(entry['causes'])}；建议：{'、'.join(entry['recommendations'])}"
                )
            knowledge_text = "\n".join(lines)

        prompt = f"""你是一个农作物病虫害诊断顾问。用户种植的作物是"{crop}"，描述的症状是"{symptom_text}"。
下面是本地知识库匹配到的相关记录，请结合这些信息，用简洁的语言（3句话以内）给出诊断和处理建议：

{knowledge_text}
"""
        return self.agent.run(prompt)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    agent = CropDiagnosisAgent()
    print(agent.run("水稻", "最近发现叶子发黄"))