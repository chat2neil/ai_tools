# %%
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

# %%
### LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# %%
### Schema

class Persona(BaseModel):
    role: str = Field(
        description="Role of the persona in the context of the topic.",
    )
    key_interests: str = Field(
        description="Comprehensive bullet list of primary interests, concerns, and motives.",
    )

    @property
    def description(self) -> str:
        return f"Role: {self.role}\nKey interests: {self.key_interests}\n"


class Perspectives(BaseModel):
    personas: List[Persona] = Field(
        description="List of personas who are primary stakeholders for the topic.",
    )


class GeneratePersonasState(TypedDict):
    topic: str  # Research topic
    human_feedback_on_personas: str  # Human feedback
    personas: List[Persona]  # Personas to consider for this topic


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Persona  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    human_feedback_on_personas: str  # Human feedback
    personas: List[Persona]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


# %%
### Creating personas, with human feedback

def create_personas(state: GeneratePersonasState):
    """Create personas who are would primarily be interested in the topic"""

    topic = state["topic"]
    max_personas = 3
    human_feedback_on_personas = state.get("human_feedback_on_personas", "")

    print(f"Creating personas for the topic: {topic}")
    print(f"Human feedback on personas: {human_feedback_on_personas}")

    create_personas_prompt = PromptTemplate.from_file(
        "./research_assistant/prompts/create_personas.txt"
    )
    system_message = create_personas_prompt.format(
        topic=topic,
        human_feedback_on_personas=human_feedback_on_personas,
        max_personas=max_personas,
    )

    personas = llm.with_structured_output(Perspectives).invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of personas.")]
    )

    return {"personas": personas.personas}


def human_feedback(state: GeneratePersonasState):
    """No-op node that should be interrupted on"""
    pass


# %%
### Generate analyst question

def generate_question(state: InterviewState):
    """Node to generate a question"""

    question_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/question_instructions.txt"
    )

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.description)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


# Search query writing
def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    search_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/search_instructions.txt"
    )

    # Search
    tavily_search = TavilySearchResults(max_results=3)

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [SystemMessage(content=search_instructions)] + state["messages"]
    )

    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia"""

    search_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/search_instructions.txt"
    )

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [SystemMessage(content=search_instructions)] + state["messages"]
    )

    # Search
    search_docs = WikipediaLoader(
        query=search_query.search_query, load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


# Generate expert answer
def generate_answer(state: InterviewState):
    """Node to answer a question"""

    answer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/answer_instructions.txt"
    )

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(
        goals=analyst.description, context=context
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"


# Write a summary (section of the final report) of the interview
def write_section(state: InterviewState):
    """Node to write a section"""

    section_writer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/section_writer_instructions.txt"
    )

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


def initiate_information_gathering(state: ResearchGraphState):
    """Conditional edge to initiate a series of parallel information gathering processes
     via Send() API or return to create_personas"""

    # Check if human feedback
    human_feedback_on_personas = state.get("human_feedback_on_personas", "approve")
    
    if human_feedback_on_personas.lower() != "approve":
        # Return to create_personas
        return "create_personas"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        return END
    # else:
    #     topic = state["topic"]
    #     return [
    #         Send(
    #             "gather_information",
    #             {
    #                 "analyst": analyst,
    #                 "messages": [
    #                     HumanMessage(
    #                         content=f"So you said you were writing an article on {topic}?"
    #                     )
    #                 ],
    #             },
    #         )
    #         for analyst in state["personas"]
    #     ]


def write_report(state: ResearchGraphState):
    """Write the final report body"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    report_writer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/report_writer_instructions.txt"
    )
    system_message = report_writer_instructions.format(
        topic=topic, context=formatted_str_sections
    )
    report = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Write a report based upon these memos.")]
    )
    return {"content": report.content}


# Write the introduction or conclusion
intro_conclusion_instructions = PromptTemplate.from_file(
    "./research_assistant/prompts/intro_conclusion_instructions.txt"
)


def write_introduction(state: ResearchGraphState):
    """Node to write the introduction"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    intro_conclusion_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/intro_conclusion_instructions.txt"
    )
    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Node to write the conclusion"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    intro_conclusion_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/intro_conclusion_instructions.txt"
    )
    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""

    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"]
        + "\n\n---\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# %%
# Research information gathering stage
gathering_process_builder = StateGraph(InterviewState)
gathering_process_builder.add_node("ask_question", generate_question)
gathering_process_builder.add_node("search_web", search_web)
gathering_process_builder.add_node("search_wikipedia", search_wikipedia)
gathering_process_builder.add_node("answer_question", generate_answer)
gathering_process_builder.add_node("save_interview", save_interview)
gathering_process_builder.add_node("write_section", write_section)

# Flow
gathering_process_builder.add_edge(START, "ask_question")
gathering_process_builder.add_edge("ask_question", "search_web")
gathering_process_builder.add_edge("ask_question", "search_wikipedia")
gathering_process_builder.add_edge("search_web", "answer_question")
gathering_process_builder.add_edge("search_wikipedia", "answer_question")
gathering_process_builder.add_conditional_edges(
    "answer_question", route_messages, ["ask_question", "save_interview"]
)
gathering_process_builder.add_edge("save_interview", "write_section")
gathering_process_builder.add_edge("write_section", END)


# %%
# Create graph
builder = StateGraph(ResearchGraphState)
builder.add_node("create_personas", create_personas)
builder.add_node("human_feedback", human_feedback)
# builder.add_node("gather_information", gathering_process_builder.compile())
# builder.add_node("write_report",write_report)
# builder.add_node("write_introduction",write_introduction)
# builder.add_node("write_conclusion",write_conclusion)
# builder.add_node("finalize_report",finalize_report)

# Create personas and seek feedback from user until approved
# Then move to information gathering
builder.add_edge(START, "create_personas")
builder.add_edge("create_personas", "human_feedback")
# builder.add_conditional_edges("human_feedback", initiate_information_gathering, ["create_personas", "gather_information"])
builder.add_conditional_edges("human_feedback", initiate_information_gathering, ["create_personas", END])

# After gathering information, write the report
# builder.add_edge("gather_information", "write_report")
# builder.add_edge("gather_information", "write_introduction")
# builder.add_edge("gather_information", "write_conclusion")
# builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
# builder.add_edge("finalize_report", END)

builder.add_edge("create_personas", END)

# Compile
graph = builder.compile(interrupt_before=['human_feedback'])
# graph = builder.compile()

# %%
graph

# %%
# Test the graph
# state = {
#     "topic": "Snowflake data lakehouse capabilities",
#     "human_feedback_on_personas": "make sure a data scientist is included",
#     "personas": [],
#     "sections": [],
#     "introduction": "",
#     "content": "",
#     "conclusion": "",
#     "final_report": ""
# }

# graph.invoke(state)

# %%
