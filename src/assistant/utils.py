import os
import re
import shutil
from ollama import chat, Client
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_community.document_loaders import CSVLoader, TextLoader, PDFPlumberLoader, JSONLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.assistant.vector_db import add_documents

class Evaluation(BaseModel):
    is_relevant: bool

class Queries(BaseModel):
    queries: list[str]


class StockPrice(BaseModel):
    stock_summary: str = Field(
        ...,
        examples=["The stock financial situation is ..."],
        desription="The summary of the stock financial performance ",
    )
    ticker: str = Field(
        ..., examples=["NVDA", "AMD"], description="The stock ticker", required=True
    )
    price: float = Field(
        ...,
        examples=[100.0, 200.0],
        description="The predicted stock price",
        required=True,
    )
    currency: str = Field(
        ..., examples=["USD", "EUR"], description="The currency of the stock price"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the stock financial performance")
    confidence_score: float = Field(..., ge=-1.0, le=1.0, examples=[0.0, 1.0], description="The confidence score of the stock sentiment")
    think: Optional[str] = Field(description="Think though how to answer the question, non reasoning model should be empty")


def parse_output(text):
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL).group(1).strip()
    output = re.search(r'</think>\s*(.*?)$', text, re.DOTALL).group(1).strip()

    return {
        "reasoning": think,
        "response": output
    }

def parse_stock_price(stockprice: StockPrice):
    return stockprice.model_dump()

def extract_symbol_and_date(query):
    symbol = re.search(r'\b[A-Z]{2,5}\b', query).group(0)
    date = re.search(r'\d{4}-\d{2}-\d{2}', query).group(0)
    return symbol, date

def format_documents_with_metadata(documents):
    """
    Convert a list of Documents into a formatted string including metadata.

    Args:
        documents: List of Document objects

    Returns:
        String containing document content and metadata
    """
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown source')
        formatted_doc = f"Source: {source}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_doc)

    return "\n\n---\n\n".join(formatted_docs)

def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Connect to remote Ollama instance
        host = os.getenv("OLLAMA_HOST", "http://141.117.231.104:11434")
        client = Client(host=host)
        response = client.chat(
            messages=messages,
            model=model,
            format=output_format.model_json_schema() if output_format else None
        )
        
        if output_format:
            return output_format.model_validate_json(response.message.content)
        else:
            return response.message.content
            
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        # Fallback to local instance if remote fails
        response = chat(
            messages=messages,
            model=model,
            format=output_format.model_json_schema() if output_format else None
        )
        
        if output_format:
            return output_format.model_validate_json(response.message.content)
        else:
            return response.message.content

def invoke_ollama_chat(model, system_prompt, user_prompt, temperature=0, output_format=None):
    ollama_model = ChatOllama(model=model, temperature=temperature)
    if output_format is not None:
        ollama_model = ollama_model.with_structured_output(output_format, method="json_schema")
    # Invoke LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = ollama_model.invoke(messages)
    if output_format is not None:
        return response
    else:
        return response.content


def invoke_vllm(model, system_prompt, user_prompt, output_format=None):
    from openai import OpenAI
    # Set the API base to your vLLM server endpoint.
    # Replace the URL with your actual vLLM endpoint (including the appropriate port and path, e.g., /v1).
    client = OpenAI(
        base_url="http://141.117.231.104:8000/v1",
        api_key="EMPTY",
    )

    # models = client.models.list()
    # model = models.data[0].id
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    response = response.choices[0]

    if output_format:
        return output_format.model_validate_json(response.message.content)
    else:
        return response.message.content

def invoke_llm(
    model,  # Specify the model name from OpenRouter
    system_prompt,
    user_prompt,
    output_format=None,
    temperature=0
):
        
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model, 
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base= "https://openrouter.ai/api/v1",
    )
    
    # If Response format is provided use structured output
    if output_format:
        llm = llm.with_structured_output(output_format)
    
    # Invoke LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)
    
    if output_format:
        return response
    return response.content # str response

def tavily_search(query, include_raw_content=True, max_results=3):
    """ Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    tavily_client = TavilyClient()
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content
    )

def get_report_structures(reports_folder="report_structures"):
    """
    Loads report structures from .md or .txt files in the specified folder.
    Each file should be named as 'report_name.md' or 'report_name.txt' and contain the report structure.
    Returns a dictionary of report structures.
    """
    report_structures = {}

    # Create the folder if it doesn't exist
    os.makedirs(reports_folder, exist_ok=True)

    try:
        # List all .md and .txt files in the folder
        for filename in os.listdir(reports_folder):
            if filename.endswith(('.md', '.txt')):
                report_name = os.path.splitext(filename)[0]  # Remove extension
                file_path = os.path.join(reports_folder, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        report_structures[report_name] = {
                            "content": content
                        }
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    except Exception as e:
        print(f"Error accessing reports folder: {str(e)}")

    return report_structures

def process_uploaded_files(uploaded_files, unprocessed_files_str):
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in unprocessed_files_str:
                continue
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_file_path = os.path.join(temp_folder, uploaded_file.name)

            # Save file temporarily
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Choose the appropriate loader
            if file_extension == "csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in ["txt", "md"]:
                loader = TextLoader(temp_file_path)
            elif file_extension == "pdf":
                loader = PDFPlumberLoader(temp_file_path)
            elif file_extension == "json":
                # Load JSON Lines format with jq schema to combine title and summary
                loader = JSONLoader(
                    file_path=temp_file_path,
                    jq_schema=".[]",  # Load each line as a separate document
                    json_lines=True,
                    # text_content=False,
                    # metadata_func=lambda metadata: {
                    #     "ticker": metadata.get("ticker", ""),
                    #     "date": metadata.get("date", ""),
                    #     "title": metadata.get("title", ""),
                    #     "source": "financial_news",
                    # },
                    # content_func=lambda data: (
                    #     f"Title: {data.get('title', '')}\n\n"
                    #     f"Summary: {data.get('summary', '')}\n\n"
                    #     f"Ticker: {data.get('ticker', '')}\n"
                    #     f"Date: {data.get('date', '')}"
                    # )
                )
            else:
                continue

            # Load and append documents
            docs = loader.load()
            add_documents(docs)

        return True
    finally:
        # Remove the temp folder and its contents
        shutil.rmtree(temp_folder, ignore_errors=True)
