import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT, get_structure_prompt
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries
from langchain_core.messages import HumanMessage, AIMessage

# Number of query to process in parallel for each batch
# Change depending on the performance of the system
BATCH_SIZE = 3

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    print("--- Generating research queries ---")
    user_instructions = state["user_instructions"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    
    query_writer_prompt = RESEARCH_QUERY_WRITER_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    )
    
    # Using local Deepseek R1 model with Ollama
    result = invoke_ollama(
        model='deepseek-r1:7b',
        system_prompt=query_writer_prompt,
        user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
        output_format=Queries
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # result = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=query_writer_prompt,
    #     user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
    #     output_format=Queries
    # )

    return {"research_queries": result.queries}

def initiate_query_research(state: ResearcherState):
    queries = state["research_queries"]
    current_position = state["current_position"]
    batch_end = min(current_position, len(queries))
    batch_start = max(0, batch_end - BATCH_SIZE)
    current_batch = queries[batch_start:batch_end]
    
    return [
        Send("search_and_summarize_query", {"query": s})
        for s in current_batch
    ]

def search_queries(state: ResearcherState):
    print("--- Searching queries ---")
    current_position = state.get("current_position", 0)
    # Add search_summaries if not present
    if "search_summaries" not in state:
        state["search_summaries"] = []
    return {"current_position": current_position + BATCH_SIZE}


def check_more_queries(state: ResearcherState) -> Literal["search_queries", "generate_final_answer"]:
    """Check if there are more queries to process"""
    current_position = state.get("current_position", 0)
    if current_position < len(state["research_queries"]):
        return "search_queries"
    return "generate_final_answer"

def retrieve_rag_documents(state: QuerySearchState):
    """Retrieve documents from the RAG database."""
    print("--- Retrieving documents ---")
    query = state["query"]
    vectorstore = get_or_create_vector_db()
    if vectorstore is None:
        return {"retrieved_documents": None}
    vectorstore_retreiver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    documents = vectorstore_retreiver.invoke(query)

    return {"retrieved_documents": documents}

def evaluate_retrieved_documents(state: QuerySearchState):
    print("--- Evaluating retrieved documents ---")
    if state["retrieved_documents"] is None:
        return {"are_documents_relevant": False}
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    print("Retrieved documents:", retrieved_documents)
    
    evaluation_prompt = RELEVANCE_EVALUATOR_PROMPT.format(
        query=query,
        documents=format_documents_with_metadata(retrieved_documents)
    )
    
    # Using local Deepseek R1 model with Ollama
    evaluation = invoke_ollama(
        model='deepseek-r1:7b',
        system_prompt=evaluation_prompt,
        user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
        output_format=Evaluation
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # evaluation = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=evaluation_prompt,
    #     user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
    #     output_format=Evaluation
    # )

    return {"are_documents_relevant": evaluation.is_relevant}

def route_research(state: QuerySearchState, config: RunnableConfig) -> Literal["summarize_query_research", "web_research", "__end__"]:
    """ Route the research based on the documents relevance """

    if state["are_documents_relevant"]:
        print("Documents are relevant. Proceeding to summarize.")
        return "summarize_query_research"
    elif config["configurable"].get("enable_web_search", False):
        print("Documents are not relevant. Finding web search results.")
        return "web_research"
    else:
        print("Skipping query due to irrelevant documents and web search disabled.")
        return "__end__"

def web_research(state: QuerySearchState):
    print("--- Web research ---")
    output = tavily_search(state["query"])
    search_results = output["results"]

    return {"web_search_results": search_results}

def summarize_query_research(state: QuerySearchState):
    print("--- Summarizing query research ---")
    query = state["query"]
    information = None
    if state["are_documents_relevant"]:
        information = state["retrieved_documents"]
    else:
        information = state["web_search_results"]

    summary_prompt = SUMMARIZER_PROMPT.format(
        query=query,
        docmuents=information
    )
    
    summary = invoke_ollama(
        model='deepseek-r1:7b',
        system_prompt=summary_prompt,
        user_prompt=f"Generate a summary for this query: {query}"
    )
    summary = parse_output(summary)["response"]
    print("Summary of query search:", summary)
    return {
        "query": query,  # Include query for tracking
        "search_summaries": [summary]
    }

def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    print("--- Generating final answer ---")
    
    # Validate required state
    if not state.get("search_summaries"):
        return {
            "final_answer": "Unable to generate answer: No research summaries available"
        }
    
    report_structure = config["configurable"].get("report_structure", "")
    structure_name = config["configurable"].get("structure_name", "none")
    structure_prompt = get_structure_prompt(structure_name)

    # Format chat history for the prompt
    chat_history_str = ""
    if "chat_history" in state and state["chat_history"]:
        formatted_messages = []
        for msg in state["chat_history"]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_messages.append(f"{role}: {msg.content}")
        chat_history_str = "\n".join(formatted_messages)
    
    answer_prompt = REPORT_WRITER_PROMPT.format(
        chat_history=chat_history_str,
        structure_instruction=structure_prompt["instruction"],
        instruction=state["user_instructions"],
        report_structure=report_structure,
        information="\n\n---\n\n".join(state["search_summaries"]),
        structure_guidelines=structure_prompt["guidelines"]
    )

    # Using local Deepseek R1 model with Ollama
    result = invoke_ollama(
        model='deepseek-r1:7b',
        system_prompt=answer_prompt,
        user_prompt=f"Generate a research summary using the provided information and chat history."
    )
    # Remove thinking part (reasoning between <think> tags)
    parsing_result = parse_output(result)
    # answer = parsing_result["response"]
    return {"final_answer": parsing_result}

def route_web_research(state: QuerySearchState) -> Literal["summarize_query_research", "__end__"]:
    """Route after web research based on results"""
    if state.get("web_search_results"):
        return "summarize_query_research"
    return "__end__"

# Create subghraph for searching each query
query_search_subgraph = StateGraph(QuerySearchState, input=QuerySearchStateInput, output=QuerySearchStateOutput)

# Define subgraph nodes for searching the query
query_search_subgraph.add_node(retrieve_rag_documents)
query_search_subgraph.add_node(evaluate_retrieved_documents)
query_search_subgraph.add_node(web_research)
query_search_subgraph.add_node(summarize_query_research)

# Set entry point and define transitions for the subgraph
query_search_subgraph.add_edge(START, "retrieve_rag_documents")
query_search_subgraph.add_edge("retrieve_rag_documents", "evaluate_retrieved_documents")
query_search_subgraph.add_conditional_edges(
    "evaluate_retrieved_documents",
    route_research,
    {
        "summarize_query_research": "summarize_query_research",
        "web_research": "web_research",
        "__end__": END,
    },
)
query_search_subgraph.add_conditional_edges(
    "web_research",
    route_web_research,
    {"summarize_query_research": "summarize_query_research", "__end__": END},
)
query_search_subgraph.add_edge("summarize_query_research", END)

# Create main research agent graph
researcher_graph = StateGraph(ResearcherState, input=ResearcherStateInput, output=ResearcherStateOutput, config_schema=Configuration)

# Define main researcher nodes
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", query_search_subgraph.compile())
researcher_graph.add_node(generate_final_answer)

# Define transitions for the main graph
researcher_graph.add_edge(START, "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "search_queries")
researcher_graph.add_conditional_edges("search_queries", initiate_query_research, ["search_and_summarize_query"])
researcher_graph.add_conditional_edges("search_and_summarize_query", check_more_queries)
researcher_graph.add_edge("generate_final_answer", END)

# Compile the researcher graph
researcher = researcher_graph.compile()
