from research_assistant import builder
from dotenv import load_dotenv
from utils.report_name import generate_report_name

load_dotenv()

topic = "How to consume Snowflake data from Amazon Lake Formation, including how to manage data access security.  Use AWS architect, security architect and data engineer roles as personas."
path = "/Users/neil/src/ai_tools/reports"

file_name = generate_report_name(path, topic)
print(f"Generated file name: {file_name}")

print("Compiling graph...")
graph = builder.compile()

print("Generating research report...")
results = graph.invoke({"topic": topic, "human_feedback": "approve"}, {"recursion_limit": 50})
content = results["final_report"]

with open(file_name, "w") as f:
    f.write(content)

print(f"Report saved to {file_name}")
