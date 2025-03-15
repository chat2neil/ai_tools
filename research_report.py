from research_assistant import builder

topic = "How to consume Snowflake data from Amazon Lake Formation, including how to manage data access security.  Use AWS architect, security architect and data engineer roles as personas."

print("Generating research report...")
graph = builder.compile()
results = graph.invoke({"topic": topic, "human_feedback": "approve"}, {"recursion_limit": 50})
content = results["final_report"]

report_file_name = f"/Users/neil/src/ai_tools/reports/{topic.replace(' ', '_').lower()}.md"
with open(report_file_name, "w") as f:
    f.write(content)

print(f"Report saved to {report_file_name}")
