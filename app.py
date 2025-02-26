# Solving pysqlite issue with streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pyperclip
import streamlit as st
from src.assistant.graph import researcher
from src.assistant.utils import get_report_structures, process_uploaded_files
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pyngrok import ngrok

load_dotenv()

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
                    value_response = {"final_answer": value["final_answer"]["response"]} if is_dict_instance else {"final_answer":value["final_answer"]}
                    value_reasoning = {"final_thinking": value["final_answer"]["reasoning"]} if is_dict_instance else {"final_thinking": value["final_answer"]}
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

    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
        help="Select the structure for the generated report."
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
    
    enable_web_search = st.sidebar.checkbox("Enable Web Search", value=False)

    # Upload file logic
    uploaded_files = st.sidebar.file_uploader(
        "Upload New Documents",
        type=["pdf", "txt", "csv", "md", "json"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # Check for new unprocessed files 
    # Display the "Process Files" button and status
    if uploaded_files:
        process_button_placeholder = st.sidebar.empty()
        current_files = {f.name for f in uploaded_files}
        unprocessed_files = current_files - st.session_state.processed_files
        # Show process button if there are unprocessed files
        if unprocessed_files:
            with process_button_placeholder.container():
                process_clicked = st.button("Process Files", use_container_width=True)

                if process_clicked:
                    with st.sidebar.status("Processing files...", expanded=False) as status:
                        if process_uploaded_files(uploaded_files, unprocessed_files):
                            # Add newly processed files to the set
                            st.session_state.processed_files.update(current_files)
                            st.session_state.file_status = status
                            status.update(label=f"Processed {len(unprocessed_files)} new files successfully!", state="complete", expanded=False)
        
        # Show status for all processed files
        if st.session_state.processed_files:
            if st.session_state.file_status:
                total_processed = len(st.session_state.processed_files)
                st.session_state.file_status.update(
                    label=f"Total files processed: {total_processed}", 
                    state="complete", 
                    expanded=False
                )

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
        st.session_state.chat_history.append(AIMessage(content=assistant_response["final_answer"]["response"]))
        st.session_state.messages.append({"role": "assistant", "content": assistant_response["final_answer"]["response"]})

        with st.chat_message("assistant"):
            st.write(assistant_response["final_answer"]["response"])  # AI response

            # Copy button below the AI message
            if st.button("📋", key=f"copy_{len(st.session_state.messages) - 1}"):
                pyperclip.copy(assistant_response["final_answer"])
    
    # Show display the ngrok link inside your Streamlit app UI.
    ngrok.set_auth_token("2tXPwDst3Zu00YOYMDzYjCnQZ96_Sz2BVfU7jCHmg5bBmffc")
    tunnel = ngrok.connect(8501, "http")
    public_url = tunnel.public_url
    st.sidebar.markdown(f"**Public URL:** {public_url}")

if __name__ == "__main__":
    main()