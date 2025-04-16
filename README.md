# 🚀 **Financial RAG Chatbot with DeepSeek R1 & Langgraph**

This is a  **local adaptive RAG financial agent** using **LangGraph** and a local **DeepSeek R1 model** running on **Ollama**. This agent act like a deep researcher, designed to gather, analyze, and summarize information based on user instructions.  

## **How It Works** 

1. **Generating Research Queries** – The agent takes user input and formulates relevant research questions to find the most useful information.  

2. **Retrieving Documents** – It searches a local **Chroma database** to pull relevant documents related to the query.  

3. **Evaluating Relevance** – Each document is checked against the original query to ensure it contains meaningful and accurate information.  

4. **Expanding Search if Needed** – If the retrieved documents are not sufficient or relevant, the agent can **search the web** for additional sources.  

5. **Summarizing Findings** – After gathering all necessary information, the agent processes the data and extracts key insights.  

6. **Final Report Generation** – The summarized findings are sent to a **writer agent**, which structures the information into a **detailed and well-formatted report** based on a predefined format.  


## **Key Features**  

- **Dynamic Search Through Local Documents** – Efficiently retrieves relevant information from your internal documents.  
- **Advanced Insight Extraction** – Leverages the reasoning power of **DeepSeek R1** model to evaluate, analyze, and extract the most valuable insights from documents.  
- **Real-Time Web Search** – Expands research by accessing online sources using **[Tavily API](https://tavily.com/)** or **[DuckDuckGO](https://duckduckgo.com/)** when local documents are insufficient and irrelevant.  
- **Structured Report Generation** – Produces well-formatted reports based on your predefined reporting templates.

## System Flowchart

This is the detailed flow of the system:

<div align="center">
  <img src="https://github.com/user-attachments/assets/5e06e948-c853-47d1-b25e-e3c5ca96b60d" alt="Langgraph Local Deepseek RAG researcher">
</div>


## Tech Stack
- **[Ollama](https://ollama.com/)**: Runs the DeepSeek R1 model locally.
- **[LangGraph](https://www.langchain.com/langgraph)**: Builds AI agents and defines the researcher's workflow.
- **[ChromaDB](https://docs.trychroma.com/)**: Local vector database for RAG-based retrieval.
- **[Streamlit](https://docs.streamlit.io/)**: Provides a UI for interacting with the researcher.
- **[Tavily](https://tavily.com/)**: For searching the web.

## How to Run
### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- Ollama
- Tavily API key or DuckDuckGo for web searchs
- Necessary Python libraries (listed in `requirements.txt`)

### Setup
#### Clone the Repository
```bash
git clone https://github.com/kaymen99/local-rag-researcher-deepseek
cd local-rag-researcher-deepseek
```

#### Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Install Required Packages
```bash
pip install -r requirements.txt
```

#### Set Up Environment Variables
Create a `.env` file in the root directory and add necessary credentials:

```ini
# Tavily API key for SearchTool (optional)
TAVILY_API_KEY="your-api-key"
```

## Running the Application
### Step 1: Install and Run Ollama

Follow the official instructions to [install Ollama](https://ollama.com/download), then pull the DeepSeek R1 model (this project uses the 7b model but you can choose any other [models available](https://ollama.com/library/deepseek-r1)):

```bash
ollama pull deepseek-r1:7b
```

### Step 2: Launch the Streamlit App

Run the following command to start the UI:

```bash
streamlit run app.py
```

### Step 3: Visualize in LangGraph Studio (Optional)

Since the researcher is built with LangGraph, you can use **LangGraph Studio** to inspect the agent's workflow. To do this, run the following commands:  

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev
```

## Customization
### Modify Report Structures
- Add custom structures inside the `report_structures` folder.
- Select the preferred structure in the UI.

### Using an External LLM Provider  

By default, the researcher runs locally using the **DeepSeek R1 model** on **Ollama**. However, if you prefer to use a cloud-based LLM provider instead (such as **Cloud DeepSeek R1**, **OpenAI GPT-4o**, or **OpenAI o1**), follow these steps:  

1. **Modify the Code**:  
   - Go to `assistant/graph.py`.  
   - Comment the code invoking Ollama model.  
   - Uncomment the section of code that enables external LLM calls.  
   - `invoke_llm` uses **[OpenRouter](https://openrouter.ai)**, which provides access to multiple LLMs. You can choose your preferred model from their [list](https://openrouter.ai/models). 🚀  
   - You can also modify the `invoke_llm` function to use a single LLM provider instead of **OpenRouter** if you want.  

2. **Set Up API Keys**:  
   - Obtain OpenRouter API key from [here](https://openrouter.ai/settings/keys).  
   - Add these keys to your `.env` file in the following format:  

     ```env
     OPENROUTER_API_KEY=your_openai_key
     ```

## **📚 Further Reading & Resources**

* Langchain: Building a fully local "deep researcher" with DeepSeek-R1[see](https://www.youtube.com/watch?v=sGUjmyfof4Q) 

* Langchain: Building a fully local research assistant from scratch with Ollama [see](https://www.youtube.com/watch?v=XGuTzHoqlj8) 

* LangGraph Template: Multi-Agent RAG Research [see](https://www.youtube.com/watch?v=JLDLANs_m_w) 

* LangGraph Adaptative RAG implementation [see](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb)

## Contact
If you have any questions or suggestions, feel free to contact me at leihan.chen@torontomu.ca.