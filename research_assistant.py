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
    concerns: List[str] = Field(
        description="Comprehensive list of interests, concerns, and motives.",
    )

    @property
    def description(self) -> str:
        concerns_bullets = "\n - ".join(self.concerns)
        return f"Role: {self.role}\nConcerns: {concerns_bullets}\n"


class TopicPerspectives(BaseModel):
    personas: List[Persona] = Field(
        description="List of personas who are primary stakeholders for the topic.",
    )


class GeneratePersonasState(TypedDict):
    topic: str  # Research topic
    human_feedback: str  # Human feedback
    personas: List[Persona]  # Personas to consider for this topic


class GatherInformationState(TypedDict):
    context: Annotated[list, operator.add]  # Source docs
    persona_role: str        # The persona with an interest
    topic: str          # The topic being researched
    concern: str        # The specific concern we are researching
    question: str       # The question we are asking
    information: str    # The information gathered
    sections: list      # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    human_feedback: str  # Human feedback
    personas: List[Persona]  # persona asking questions
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
    human_feedback = state.get("human_feedback", "")

    create_personas_prompt = PromptTemplate.from_file(
        "./research_assistant/prompts/create_personas.txt"
    )
    system_message = create_personas_prompt.format(
        topic=topic,
        human_feedback=human_feedback
    )

    personas = llm.with_structured_output(TopicPerspectives).invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of personas.")]
    )

    return {"personas": personas.personas}


def get_human_feedback(state: GeneratePersonasState):
    """No-op node that should be interrupted on"""
    pass

def start_gathering_information(state: ResearchGraphState):
    """Conditional edge to initiate a series of parallel information gathering processes
     via Send() API or return to create_personas"""

    # Check if human feedback
    human_feedback = state.get("human_feedback", "approve")
    
    if human_feedback.lower() != "approve":
        # Return to create_personas
        return "create_personas"

    # Otherwise kick off informations in parallel via Send() API
    else:
        topic = state["topic"]
        return [
            Send(
                "gather_information",
                {
                    "persona_role": persona.role,
                    "topic": topic,
                    "concern": concern,
                },
            )
            for persona in state["personas"] for concern in persona.concerns
        ]

# %%
### Gather information from the perspective of each persona

def generate_question(state: GatherInformationState):
    """Generate a question to gather information about a particular persona and one of their concerns"""

    topic = state["topic"]
    persona_role = state["persona_role"]
    concern = state["concern"]
    
    question_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/generate_question.txt"
    )
    system_message = question_instructions.format(topic=topic, persona_role=persona_role, concern=concern)
    question = llm.invoke([SystemMessage(content=system_message), HumanMessage("Compose the question.")])

    return {"question": question.content}


# Search query writing
def search_web(state: GatherInformationState):
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


def search_wikipedia(state: GatherInformationState):
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
def generate_answer(state: GatherInformationState):
    """Node to answer a question"""

    answer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/answer_instructions.txt"
    )

    # Get state
    persona = state["persona"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(
        goals=persona.description, context=context
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_information(state: GatherInformationState):
    """Save information"""

    # Get messages
    messages = state["messages"]

    # Convert information to a string
    information = get_buffer_string(messages)

    # Save to informations key
    return {"information": information}


def ask_another_question(state: GatherInformationState, name: str = "expert"):
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
        return "save_information"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_information"
    
    return "generate_question"


# %%
### Write report

def write_section(state: GatherInformationState):
    """Node to write a section"""

    section_writer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/section_writer_instructions.txt"
    )

    # Get state
    information = state["information"]
    context = state["context"]
    persona = state["persona"]

    # Write section using either the gathered source docs from information (context) or the information itself (information)
    system_message = section_writer_instructions.format(focus=persona.description)
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


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
gather_info_builder = StateGraph(GatherInformationState)
gather_info_builder.add_node("generate_question", generate_question)
# gather_info_builder.add_node("search_web", search_web)
# gather_info_builder.add_node("search_wikipedia", search_wikipedia)
# gather_info_builder.add_node("answer_question", generate_answer)
# gather_info_builder.add_node("save_information", save_information)
# gather_info_builder.add_node("write_section", write_section)

# Flow
gather_info_builder.add_edge(START, "generate_question")
# gather_info_builder.add_edge("generate_question", "search_web")
# gather_info_builder.add_edge("generate_question", "search_wikipedia")
# gather_info_builder.add_edge("search_web", "answer_question")
# gather_info_builder.add_edge("search_wikipedia", "answer_question")
# gather_info_builder.add_conditional_edges(
#     "answer_question", ask_another_question, ["generate_question", "save_information"]
# )
# gather_info_builder.add_edge("save_information", "write_section")
# gather_info_builder.add_edge("write_section", END)
gather_info_builder.add_edge("generate_question", END)

# %%
# Create graph
builder = StateGraph(ResearchGraphState)
builder.add_node("create_personas", create_personas)
builder.add_node("get_human_feedback", get_human_feedback)
builder.add_node("gather_information", gather_info_builder.compile())
# builder.add_node("write_report",write_report)
# builder.add_node("write_introduction",write_introduction)
# builder.add_node("write_conclusion",write_conclusion)
# builder.add_node("finalize_report",finalize_report)

# Create personas and seek feedback from user until approved
# Then move to information gathering
builder.add_edge(START, "create_personas")
builder.add_edge("create_personas", "get_human_feedback")
builder.add_conditional_edges("get_human_feedback", start_gathering_information, ["create_personas", "gather_information"])

# After gathering information, write the report
# builder.add_edge("gather_information", "write_report")
# builder.add_edge("gather_information", "write_introduction")
# builder.add_edge("gather_information", "write_conclusion")
# builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
# builder.add_edge("finalize_report", END)

builder.add_edge("gather_information", END)

# Compile
graph = builder.compile(interrupt_before=['get_human_feedback'])

# %%
graph

# %%
# Test the graph
# state = {
#     "topic": "Snowflake data lakehouse capabilities",
#     "human_feedback": "make sure a data scientist is included",
#     "personas": [],
#     "sections": [],
#     "introduction": "",
#     "content": "",
#     "conclusion": "",
#     "final_report": ""
# }

# graph.invoke(state)

# %%
