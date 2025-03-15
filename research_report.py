from research_assistant import builder

topic = "How to consume S3 data from Snowflake"

print("Generating research report...")
graph = builder.compile()
results = graph.invoke({"topic": topic, "human_feedback": "approve"})
content = results["final_report"]

report_file_name = f"/Users/neil/src/ai_tools/reports/{topic.replace(' ', '_').lower()}.md"
with open(report_file_name, "w") as f:
    f.write(content)

print(f"Report saved to {report_file_name}")
