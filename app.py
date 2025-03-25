# Solving pysqlite issue with streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pyperclip
import streamlit as st
import streamlit_nested_layout
import pandas as pd
from src.assistant.graph import researcher
from src.assistant.utils import get_report_structures, process_uploaded_files
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pyngrok import ngrok
import os
import re
import io

load_dotenv()

REPORTS_PATH = "C:\\Users\\tejas\\Downloads\\nlpa1\\parsed_per_file_only_keywords"

def find_reports_for_ticker(ticker):
    """
    Searches for available reports for a given ticker.
    Returns a dictionary of {Year_ReportType: file_path}
    """
    reports = {}
    pattern = re.compile(f"^{ticker}_\d+_(10K|DEF14A|YFinance)\\_parsed.(txt|csv)$")

    for file in os.listdir(REPORTS_PATH):
        if pattern.match(file):
            year, report_type = file.split("_")[1:3]
            reports[f"{year} {report_type}"] = os.path.join(REPORTS_PATH, file)

    return reports

def generate_response(user_input, enable_web_search, report_structure, max_search_queries, chat_history):
    """
    Generate response using the researcher agent and stream steps
    """
    # Initialize state for the researcher
    initial_state = {
        "user_instructions": user_input,
        "chat_history": chat_history
    }

    # Langgraph researcher config
    config = {"configurable": {
        "enable_web_search": enable_web_search,
        "report_structure": report_structure,
        "max_search_queries": max_search_queries,
    }}

    # Create the status for the global "Researcher" process
    langgraph_status = st.status("**Researcher Running...**", state="running")

    # Force order of expanders by creating them before iteration
    with langgraph_status:
        generate_queries_expander = st.expander("Generate Research Queries", expanded=False)
        search_queries_expander = st.expander("Search Queries", expanded=True)
        final_answer_expander = st.expander("Generate Final Answer", expanded=False)
        final_thinking_expander = st.expander("Thinking", expanded=False)
        steps = []

        # Run the researcher graph and stream outputs
        for output in researcher.stream(initial_state, config=config):
            for key, value in output.items():
                expander_label = key.replace("_", " ").title()

                if key == "generate_research_queries":
                    with generate_queries_expander:
                        st.write(value)

                elif key.startswith("search_and_summarize_query"):
                    with search_queries_expander:
                        with st.expander(expander_label, expanded=False):
                            st.write(value)

                elif key == "generate_final_answer":
                    is_dict_instance = isinstance(value["final_answer"], dict)
                    # value_response = {"final_answer": value["final_answer"]["response"]} if is_dict_instance else {"final_answer":value["final_answer"]}
                    # value_reasoning = {"final_thinking": value["final_answer"]["reasoning"]} if is_dict_instance else {"final_thinking": value["final_answer"]}
                    value_reasoning = value["final_answer"]["think"] if is_dict_instance and (value["final_answer"].get("think") is not None) else "No reasoning generated"
                    if is_dict_instance:
                        value["final_answer"].pop("think", "Think trace not found")
                    value_response = value["final_answer"] if is_dict_instance else "No response generated"
                    with final_answer_expander:
                        st.write(value_response)
                    with final_thinking_expander:
                        st.write(value_reasoning)

                steps.append({"step": key, "content": value})

    # Update status to complete
    langgraph_status.update(state="complete", label="**Using Langgraph** (Thinking completed)")

    # Return the final report
    return steps[-1]["content"] if steps else "No response generated"

def clear_chat():
    st.session_state.messages = []
    st.session_state.processing_complete = False
    st.session_state.uploader_key = 0
    st.session_state.chat_history = []
    st.session_state.processed_files = set()  # Clear processed files set

def fetch_ticker():
    # URL of the Wikipedia page containing S&P 500 companies
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Read all tables on the page
    tables = pd.read_html(url)

    # The first table usually contains the list of S&P 500 companies
    sp500_table = tables[0]

    # Extract the 'Symbol' column and convert it to a list of strings
    tickers = sp500_table['Symbol'].tolist()

    return tickers

def main():
    st.set_page_config(page_title="DeepSeek RAG Financial Analysis", layout="wide")

    # Initialize session states
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_report_structure" not in st.session_state:
        st.session_state.selected_report_structure = None
    if "max_search_queries" not in st.session_state:
        st.session_state.max_search_queries = 3  # Default value of 3
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False
    if "file_status" not in st.session_state:
        st.session_state.file_status = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()  # Track processed files
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize ngrok connection flag
    if "ngrok_connected" not in st.session_state:
        st.session_state.ngrok_connected = False
        st.session_state.public_url = None

    # Title row with clear button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📄 RAG Financial Analysis with DeepSeek R1")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # Sidebar configuration
    st.sidebar.title("Settings")

    # Add report structure selector to sidebar
    report_structures = get_report_structures()
    default_report = "none"

    selected_structure = st.sidebar.selectbox(
        "Report Structure",
        options=list(report_structures.keys()),
        index=list(map(str.lower, report_structures.keys())).index(default_report),
        help="Select the structure for the generated report.",
        key="structure"
    )

    st.session_state.selected_report_structure = {
        "name": selected_structure,
        "content": report_structures[selected_structure]
    }

    # Maximum search queries input
    st.session_state.max_search_queries = st.sidebar.number_input(
        "Max Number of Search Queries",
        min_value=1,
        max_value=10,
        value=st.session_state.max_search_queries,
        help="Set the maximum number of search queries to be made. (1-10)"
    )
    
    enable_web_search = st.sidebar.checkbox("Enable Web Search", value=True)

    # Fetch tickers
    tickers = fetch_ticker()
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers, key="ticker")

    # Fetch available reports for the selected ticker from the internal dataset
    available_reports = find_reports_for_ticker(selected_ticker)

    # Load internal files as simulated Streamlit uploads
    internal_uploaded_files = []
    for name, path in available_reports.items():
        if os.path.isfile(path):
            with open(path, "rb") as f:
                content = f.read()
            uploaded_file = io.BytesIO(content)
            uploaded_file.name = os.path.basename(path)
            internal_uploaded_files.append(uploaded_file)

    # Combine with manually uploaded files
    user_uploaded_files = st.sidebar.file_uploader(
        "Upload Additional Documents",
        type=["pdf", "txt", "csv", "md", "json"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    uploaded_files = (user_uploaded_files or []) + internal_uploaded_files

    # Process uploaded files and integrate them with internal dataset reports
    if uploaded_files:
        process_button_placeholder = st.sidebar.empty()
        current_files = {f.name for f in uploaded_files}
        unprocessed_files = current_files - st.session_state.processed_files

        if unprocessed_files:
            with process_button_placeholder.container():
                process_clicked = st.button("Process Uploaded Files", use_container_width=True)

                if process_clicked:
                    with st.sidebar.status("Processing uploaded files...", expanded=False) as status:
                        if process_uploaded_files(uploaded_files, unprocessed_files):
                            st.session_state.processed_files.update(current_files)
                            st.session_state.file_status = status
                            status.update(
                                label=f"Processed {len(unprocessed_files)} new files successfully!",
                                state="complete",
                                expanded=False
                            )

    # Merge dataset reports & uploaded reports
    all_reports = available_reports.copy()
    for uploaded_file in uploaded_files or []:
        all_reports[uploaded_file.name] = uploaded_file

    # Display reports selection if any reports are available
    if all_reports:
        selected_report_key = st.sidebar.selectbox("Select Report", list(all_reports.keys()), key="sel_rep")
        selected_report_path = all_reports[selected_report_key]

        st.sidebar.write(f"Selected file: {selected_report_path}")

        # Read and display file content based on type
        if isinstance(selected_report_path, str):  # Internal dataset file
            if selected_report_path.endswith(".csv"):
                df = pd.read_csv(selected_report_path)
                st.write(df.head())  # Show a preview
            else:
                with open(selected_report_path, "r", encoding="utf-8") as f:
                    st.text_area("Report Content", f.read(), height=300)
        else:  # Uploaded file (streamlit file object)
            if selected_report_path.name.endswith(".csv"):
                df = pd.read_csv(selected_report_path)
                st.write(df.head())
            else:
                content = selected_report_path.getvalue().decode("utf-8")
                st.text_area("Report Content", content, height=300)
    else:
        st.sidebar.write("No reports available for this ticker.")
        
    # Connect ngrok only if not already connected
    if not st.session_state.ngrok_connected:
        ngrok.set_auth_token("2tXPwDst3Zu00YOYMDzYjCnQZ96_Sz2BVfU7jCHmg5bBmffc")
        # tunnel = ngrok.connect(8501, "http")
        # public_url = tunnel.public_url
        st.session_state.public_url = "https://403c-141-117-231-104.ngrok-free.app"
        st.session_state.ngrok_connected = True  # Set the flag to True
        

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Use the index to generate a unique key for each button
            if message["role"] == "assistant":
                if st.button("📋", key=f"copy_{idx}"):
                    pyperclip.copy(message["content"])

    # Chat input and response handling
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate and display assistant response
        report_structure = st.session_state.selected_report_structure["content"]
        assistant_response = generate_response(
            user_input, 
            enable_web_search, 
            report_structure,
            st.session_state.max_search_queries,
            st.session_state.chat_history
        )

        # Add assistant response to chat history
        if isinstance(assistant_response, dict):
            print("Predicted price: ", assistant_response["final_answer"])
        st.session_state.chat_history.append(AIMessage(content=str(assistant_response)))
        st.session_state.messages.append({"role": "assistant", "content": str(assistant_response)})

        with st.chat_message("assistant"):
            st.write(assistant_response)  # AI response

            # Copy button below the AI message
            if st.button("📋", key=f"copy_{len(st.session_state.messages) - 1}"):
                pyperclip.copy(assistant_response)
    
    if st.session_state.public_url is not None:
        st.sidebar.markdown(f"**Public URL:** {st.session_state.public_url}")


if __name__ == "__main__":
    main()
