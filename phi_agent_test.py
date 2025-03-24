from phi.agent import Agent, RunResponse
from pydantic import BaseModel, Field
from typing import List
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

MODEL_ID = "deepseek-r1:7b"


class StockPrice(BaseModel):
    summary: str = Field(
        ...,
        examples=["The reason of my prediction bases on ..."],
        desription="The summary of the prediction process",
    )
    ticker: str = Field(..., examples=["NVDA", "AMD"], description="The stock ticker")
    price: float = Field(
        ..., examples=[100.0, 200.0], description="The predicted stock price"
    )
    currency: str = Field(
        ..., examples=["USD", "EUR"], description="The currency of the stock price"
    )


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
    model=Ollama(id=MODEL_ID),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id=MODEL_ID),
    tools=[TavilyTools()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    # team=[local_knowledge_agent, web_agent, finance_agent],
    team=[web_agent, finance_agent],
    model=Ollama(id=MODEL_ID),
    instructions=[
        # "First, search relevant info in vector database and provide a summary.",
        # "Then, using summary to keep finding relevant information in the web.",
        "Then, ask yfinancial tool to provide the latest news and analyst recommendations.",
        "Finally, provide a summary and prediction with structure output.",
    ],
    show_tool_calls=True,
    structured_outputs=True,
    markdown=True,
    debug_mode=True,
    output_model=StockPrice,
)

# agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
agent_team.print_response(
    "Please provide a prediction for AMD stock price in April 2025",
    stream=True,
)
