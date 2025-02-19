import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class ResearcherState(TypedDict):
    user_instructions: str
    research_queries: list[str]
    search_summaries: Annotated[list, operator.add]
    current_position: int
    final_answer: str
    chat_history: List[BaseMessage]

class ResearcherStateInput(TypedDict):
    user_instructions: str
    chat_history: List[BaseMessage]

class ResearcherStateOutput(TypedDict):
    final_answer: str

class QuerySearchState(TypedDict):
    query: str
    web_search_results: list
    retrieved_documents: list
    are_documents_relevant: bool
    search_summaries: list[str]

class QuerySearchStateInput(TypedDict):
    query: str

class QuerySearchStateOutput(TypedDict):
    query: str
    search_summaries: list[str]