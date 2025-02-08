import os
from dotenv import load_dotenv
import json
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o", temperature=0, tools=tools)

def chatbot(state: dict) -> dict:
    return {'messages': [llm.invoke(state['messages'])]}

def tools_called(state: dict) -> dict:
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")
    
    outputs = []
    for tool_call in message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(
            tool_call["args"]
        )
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def are_tools_called(state: dict) -> dict:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools_called"
    return END

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", tools_called)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", are_tools_called)
builder.add_edge("tools", "chatbot")

graph = builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    elif user_input.strip() == "":
        continue

    stream_graph_updates(user_input)
