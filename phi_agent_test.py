from phi.agent import Agent, RunResponse
from pydantic import BaseModel, Field
from typing import List, Literal
from phi.model.ollama import Ollama
from phi.tools.yfinance import YFinanceTools
from phi.tools.tavily import TavilyTools
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.json import JSONKnowledgeBase
from phi.knowledge.text import TextKnowledgeBase
from phi.embedder.ollama import OllamaEmbedder
from phi.document.chunking.recursive import RecursiveChunking
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REASONING_MODEL_ID = "deepseek-r1:7b"
TOOL_MODEL_ID = "llama3.2"


class StockPrice(BaseModel):
    finance_summary: str = Field(
        ...,
        examples=["The financial performance of stock is ..."],
        desription="The summary of the company financial situation",
    )
    # ticker: str = Field(..., examples=["NVDA", "AMD"], description="The stock ticker")
    # price: float = Field(
    #     ..., examples=[100.0, 200.0], description="The predicted stock price"
    # )
    # currency: str = Field(
    #     ..., examples=["USD", "EUR"], description="The currency of the stock price"
    # )
    # sentiment: Literal["positive", "negative", "neutral"] = Field(
    #     ..., description="The sentiment of the stock financial performance"
    # )
    # confidence_score: float = Field(
    #     ...,
    #     ge=-1,
    #     le=1,
    #     examples=[0.0, 1.0],
    #     description="The confidence score of the stock financial performance",
    # )


# knowledge_base = TextKnowledgeBase(
#     path="/home/leihan-chen/Downloads/cp8323_data/documents",
#     vector_db=ChromaDb(
#         collection="DEF14A_10K",
#         embedder=OllamaEmbedder(model="nomic-embed-text"),
#         persistent_client=True,
#     ),
#     num_documents=3,
#     chunking_strategy=RecursiveChunking(),
# )

# local_knowledge_agent = Agent(
#     name="Local Knowledge Agent",
#     role="Access local knowledge base",
#     model=Ollama(id=MODEL_ID),
#     knowledge=knowledge_base,
#     search_knowledge=True,
# )
# local_knowledge_agent.knowledge.load(recreate=False)

finance_agent = Agent(
    name="Finance Agent",
    role="Analyze financial data",
    model=Ollama(id=TOOL_MODEL_ID),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            stock_fundamentals=True,
        )
    ],
    instructions=["Use tables to display data"],
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals",
    show_tool_calls=True,
    markdown=True,
)

# finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id=TOOL_MODEL_ID),
    tools=[TavilyTools()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    # team=[local_knowledge_agent, web_agent, finance_agent],
    team=[web_agent, finance_agent],
    mode="coordinate",
    send_team_context_to_members=True,
    model=Ollama(id=TOOL_MODEL_ID),
    reasoning_model=Ollama(id=REASONING_MODEL_ID),
    reasoning=True,
    show_full_reasoning=True,
    show_members_responses=True,
    instructions=[
        # "First, search relevant info in vector database and provide a summary.",
        "First, finding relevant information in the web about company financial report online.",
        "Then, ask yfinancial tool to provide the latest news, analyst recommendations, stock price and other metric.",
        "Finally, provide a summary of current stock financial performance with structure output.",
    ],
    show_tool_calls=True,
    # structured_outputs=True,
    # output_model=StockPrice,
    markdown=True,
    debug_mode=True,
    stream=True,
)

# agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
agent_team.print_response(
    "Please provide a financial summary with listing financial metricfor AAPL performance in March 2025",
    stream=True,
)
