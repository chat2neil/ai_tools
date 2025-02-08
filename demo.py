# %%
from dotenv import load_dotenv
from pprint import pprint
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.globals import *
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage

from langchain_openai import ChatOpenAI

load_dotenv()

class MessagesState(MessagesState):
    pass

llm = ChatOpenAI(model="gpt-4o")

# %%
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [multiply, add, divide]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

# %%
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def call_ai(state: MessagesState):
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node(call_ai)
builder.add_node(ToolNode(tools))

builder.add_edge(START, "call_ai")
builder.add_conditional_edges("call_ai", tools_condition)
builder.add_edge("tools", "call_ai")

graph = builder.compile(checkpointer=MemorySaver())
graph


