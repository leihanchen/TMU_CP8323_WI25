from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.yfinance import YFinanceTools
from phi.tools.tavily import TavilyTools

finance_agent = Agent(
    name="Finance Agent",
    model=Ollama(id="llama3.2"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id="llama3.2"),
    tools=[TavilyTools()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Ollama(id="llama3.2"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    reasoning=True,
    structured_outputs=True,
    markdown=True,
)

# agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
agent_team.print_response("Please show me your sentiment anlysis on NVDIA and predict its stock price in April 2025, please give me specific prediction metrix and your analysis based on existing news and analyst_recommendations", stream=True, show_full_reasoning=True)
