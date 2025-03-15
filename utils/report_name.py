from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import os

class ReportName(BaseModel):
    report_name: str = Field(
        description=f"A short, descriptive file name for the report.",
    )

file_name_instructions = """
Generate a short descriptive file name for this topic: {topic} 

1. The file name should be lowercase, with spaces replaced by underscores.
2. The file name should not contain any other special characters.
3. The file name should be short and descriptive and specific to the topic.
4. Aim for a file name that is 2-5 words long.
"""

def generate_report_name(path, topic):
    """
    Generate a report name for the given topic.
    """
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_message = file_name_instructions.format(topic=topic)
    name = llm.with_structured_output(ReportName).invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the file name.")]
    )

    file_name = f"{name.report_name}.md"
    path = os.path.join(path, file_name)
    return path
