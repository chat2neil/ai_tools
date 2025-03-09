# %%
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
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

from langgraph.types import Command, Send
from langgraph.graph import END, MessagesState, START, StateGraph

# %%
### Max extent of search.
MAX_PERSONAS = 1
MAX_CONCERNS = 2
MAX_SEARCH_RESULTS = 3

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
        description=f"List of {MAX_CONCERNS} key interests, concerns, and motives.",
    )

    @property
    def description(self) -> str:
        concerns_bullets = "\n - ".join(self.concerns)
        return f"Role: {self.role}\nConcerns: {concerns_bullets}\n"


class TopicPerspectives(BaseModel):
    personas: List[Persona] = Field(
        description=f"List of {MAX_PERSONAS} personas who are primary stakeholders for the topic.",
    )


class GeneratePersonasState(TypedDict):
    topic: str  # Research topic
    human_feedback: str  # Human feedback
    personas: List[Persona]  # Personas to consider for this topic


class GatherInformationSubgraphState(MessagesState):
    """
    This interim state is where the detailed questions are crafted and
    the information is gathered by searching the web.
    """

    persona: Persona  # The persona with an interest
    topic: str  # The topic being researched
    concern_index: int  # Pointer to current concern in loop
    concern: str  # The current concern we are researching
    search_results: Annotated[list, operator.add]  # List of web pages found
    all_answers: list  # Consolidate list of answers to use to write the section
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Web search query.")


class ResearchGraphState(TypedDict):
    """
    This is the state for the overall research graph.
    """

    topic: str  # Research topic
    human_feedback: str  # Human feedback
    personas: List[Persona]  # List of personas asking questions
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
        personas = state["personas"]
        return [
            Send(
                "gather_information",
                {
                    "persona": persona, 
                    "topic": topic,
                    "concern": persona.concerns[0],
                    "concern_index": 0
                 },
            )
            for persona in personas
        ]


# %%
### Gather information from the perspective of each persona

def generate_question(state: GatherInformationSubgraphState):
    """Generate a question to gather information about a particular persona and one of their concerns"""

    topic = state["topic"]
    persona_role = state["persona"].role
    concern = state["concern"]

    question_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/generate_question.txt"
    )
    system_message = question_instructions.format(
        topic=topic, persona_role=persona_role, concern=concern
    )
    question = llm.invoke(
        [SystemMessage(content=system_message), HumanMessage("Compose the question.")]
    )
    question.name = "question"

    return {"messages": [question]}


def generate_web_search_query(state: GatherInformationSubgraphState):
    """Generate web search criteria from a question"""

    question = state["messages"][-1].content
    topic = state["topic"]

    search_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/web_search_query.txt"
    ).format(question=question, topic=topic)
    query = llm.invoke(
        [SystemMessage(content=search_instructions)]
    )
    query.name = "web_search_query"

    return {"messages": [query]}


def search_web(state: GatherInformationSubgraphState):
    """Search the web using Tavily search service"""

    query = state["messages"][-1].content

    tavily_search = TavilySearchResults(max_results = MAX_SEARCH_RESULTS)
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"search_results": [formatted_search_docs]}


def search_wikipedia(state: GatherInformationSubgraphState):
    """Search wikipedia"""

    query = state["messages"][-1].content

    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"search_results": [formatted_search_docs]}


def generate_answer(state: GatherInformationSubgraphState):
    """Use the question and search results to generate an answer"""

    persona_role = state["persona"].role
    question = [q for q in state["messages"] if q.name == "question"][-1].content
    topic = state["topic"]
    search_results = state["search_results"]

    answer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/generate_answer.txt"
    )
    system_message = answer_instructions.format(
        persona_role=persona_role, question=question, topic=topic, search_results=search_results
    )
    answer = llm.invoke([SystemMessage(content=system_message)])
    answer.name = "answer"

    return {"messages": [answer]}


def next_concern(state: GatherInformationSubgraphState) -> Command[Literal["generate_question", "consolidate_answers"]]:
    """Select the persona's next concern to gather information for"""

    persona = state["persona"]
    concern_index = state["concern_index"]

    concern_index += 1
    if len(persona.concerns) > concern_index:
        next_concern = persona.concerns[concern_index]
        return Command(
            update={
                "concern_index": concern_index,
                "concern": next_concern,
                "search_results": []
            },
            goto="generate_question"
        )
    else:
        return Command(goto="consolidate_answers")


def consolidate_answers(state: GatherInformationSubgraphState):
    """Combine all answers into a single information string"""

    answers = [ans for ans in state["messages"] if ans.name == "answer"]
    all_answers = get_buffer_string(answers)
    return {"all_answers": all_answers}


def write_section(state: GatherInformationSubgraphState):
    """Node to write a section"""

    persona = state["persona"]
    topic = state["topic"]
    all_answers = state["all_answers"]

    # Write section using either the gathered source docs from information (context) or the information itself (information)
    section_writer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/write_section.txt"
    )
    system_message = section_writer_instructions.format(
        persona_role=persona.role, topic=topic, answers=all_answers
    )
    section = llm.invoke(
        [SystemMessage(content=system_message)]
    )

    # Append it to state
    return {"sections": [section.content]}


# %%
### Write final report


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
gather_info_builder = StateGraph(GatherInformationSubgraphState)
gather_info_builder.add_node("generate_question", generate_question)
gather_info_builder.add_node("generate_web_search_query", generate_web_search_query)
gather_info_builder.add_node("search_web", search_web)
gather_info_builder.add_node("search_wikipedia", search_wikipedia)
gather_info_builder.add_node("generate_answer", generate_answer)
gather_info_builder.add_node("next_concern", next_concern)
gather_info_builder.add_node("consolidate_answers", consolidate_answers)
gather_info_builder.add_node("write_section", write_section)

# Flow
gather_info_builder.add_edge(START, "generate_question")
gather_info_builder.add_edge("generate_question", "generate_web_search_query")
gather_info_builder.add_edge("generate_web_search_query", "search_web")
gather_info_builder.add_edge("generate_web_search_query", "search_wikipedia")
gather_info_builder.add_edge("search_web", "generate_answer")
gather_info_builder.add_edge("search_wikipedia", "generate_answer")

gather_info_builder.add_edge("generate_answer", "next_concern")
# gather_info_builder.add_conditional_edges("generate_answer", next_concern, ["generate_question", "consolidate_answers"])

gather_info_builder.add_edge("consolidate_answers", "write_section")
gather_info_builder.add_edge("write_section", END)

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
builder.add_conditional_edges(
    "get_human_feedback",
    start_gathering_information,
    ["create_personas", "gather_information"],
)

# After gathering information, write the report
# builder.add_edge("gather_information", "write_report")
# builder.add_edge("gather_information", "write_introduction")
# builder.add_edge("gather_information", "write_conclusion")
# builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
# builder.add_edge("finalize_report", END)

builder.add_edge("gather_information", END)

# Compile
graph = builder.compile(interrupt_before=["get_human_feedback"])

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
