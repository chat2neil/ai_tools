import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.types import Command, Send
from langgraph.graph import END, MessagesState, START, StateGraph

### Max extent of search.
MAX_PERSONAS = 3
MAX_CONCERNS = 3
MAX_SEARCH_RESULTS = 5

### LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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


### Creating personas, with human feedback

def create_personas(state: GeneratePersonasState):
    """Create personas who are would primarily be interested in the topic"""

    topic = state["topic"]
    human_feedback = state.get("human_feedback", "")

    create_personas_prompt = PromptTemplate.from_file(
        "./research_assistant/prompts/create_personas.txt"
    )
    system_message = create_personas_prompt.format(
        topic=topic, human_feedback=human_feedback
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
                    "concern_index": 0,
                },
            )
            for persona in personas
        ]


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
    query = llm.invoke([SystemMessage(content=search_instructions)])
    query.name = "web_search_query"

    return {"messages": [query]}


def search_web(state: GatherInformationSubgraphState):
    """Search the web using Tavily search service"""

    query = state["messages"][-1].content

    tavily_search = TavilySearchResults(max_results=MAX_SEARCH_RESULTS)
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
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
        persona_role=persona_role,
        question=question,
        topic=topic,
        search_results=search_results,
    )
    answer = llm.invoke([SystemMessage(content=system_message)])
    answer.name = "answer"

    return {"messages": [answer]}


def next_concern(
    state: GatherInformationSubgraphState,
) -> Command[Literal["generate_question", "consolidate_answers"]]:
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
                "search_results": [],
            },
            goto="generate_question",
        )
    else:
        return Command(goto="consolidate_answers")


def consolidate_answers(state: GatherInformationSubgraphState):
    """Combine all answers into a single information string"""

    answers = [ans for ans in state["messages"] if ans.name == "answer"]
    all_answers = get_buffer_string(answers)
    return {"all_answers": all_answers}


def write_section(state: GatherInformationSubgraphState):
    """Write a section of the report that summarises the gathered information for a persona"""

    persona = state["persona"]
    concerns = "\n- ".join(persona.concerns)
    topic = state["topic"]
    all_answers = state["all_answers"]

    # Write section using either the gathered source docs from information (context) or the information itself (information)
    section_writer_instructions = PromptTemplate.from_file(
        "./research_assistant/prompts/write_section.txt"
    )
    system_message = section_writer_instructions.format(
        persona_role=persona.role, topic=topic, concerns=concerns, answers=all_answers
    )
    section = llm.invoke([SystemMessage(content=system_message)])

    # Append it to state
    return {"sections": [section.content]}


### Write final report

def combine_sections_into_report_body(state: ResearchGraphState):
    """Consolidate the sections into a final report body"""
    
    sections = state["sections"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    return { "content": formatted_str_sections }


# Write the introduction or conclusion
def write_introduction(state: ResearchGraphState):
    """Write the introduction"""

    topic = state["topic"]
    content = state["content"]
    prompt_template = PromptTemplate.from_file(
        "./research_assistant/prompts/write_introduction.txt"
    )
    instructions = prompt_template.format(
        content=content, topic=topic
    )
    intro = llm.invoke(
        [SystemMessage(content=instructions), HumanMessage(content=f"Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Write the conclusion"""

    topic = state["topic"]
    content = state["content"]
    prompt_template = PromptTemplate.from_file(
        "./research_assistant/prompts/write_conclusion.txt"
    )
    instructions = prompt_template.format(
        content=content, topic=topic
    )
    conclusion = llm.invoke(
        [SystemMessage(content=instructions), HumanMessage(content=f"Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def combine_header_body_and_conclusion(state: ResearchGraphState):
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


# Research information gathering stage
gather_info = StateGraph(GatherInformationSubgraphState)
gather_info.add_node("generate_question", generate_question)
gather_info.add_node("generate_web_search_query", generate_web_search_query)
gather_info.add_node("search_web", search_web)
gather_info.add_node("generate_answer", generate_answer)
gather_info.add_node("next_concern", next_concern)
gather_info.add_node("consolidate_answers", consolidate_answers)
gather_info.add_node("write_section", write_section)
gather_info.add_edge(START, "generate_question")
gather_info.add_edge("generate_question", "generate_web_search_query")
gather_info.add_edge("generate_web_search_query", "search_web")
gather_info.add_edge("search_web", "generate_answer")
gather_info.add_edge("generate_answer", "next_concern")
gather_info.add_edge("consolidate_answers", "write_section")
gather_info.add_edge("write_section", END)

# Create graph
builder = StateGraph(ResearchGraphState)
builder.add_node("create_personas", create_personas)
builder.add_node("get_human_feedback", get_human_feedback)
builder.add_node("gather_information", gather_info.compile())
builder.add_node("combine_sections_into_report_body",combine_sections_into_report_body)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("combine_header_body_and_conclusion",combine_header_body_and_conclusion)

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
builder.add_edge("gather_information", "combine_sections_into_report_body")
builder.add_edge("combine_sections_into_report_body", "write_introduction")
builder.add_edge("combine_sections_into_report_body", "write_conclusion")
builder.add_edge(["write_conclusion", "write_introduction"], "combine_header_body_and_conclusion")
builder.add_edge("combine_header_body_and_conclusion", END)

graph = builder.compile(interrupt_before=["get_human_feedback"])
graph
