import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import (
    WebBaseLoader,
    JSONLoader,
    TextLoader,
    PDFPlumberLoader,
    CSVLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    loader = CSVLoader("/home/leihan-chen/Downloads/sp500_stock_data.csv")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    all_splits = text_splitter.split_documents(data)

    # Store splits
    vector_store = FAISS.from_documents(
        documents=all_splits, embedding=OllamaEmbeddings(model="nomic-embed-text")
    )
    llm = ChatOllama(model="deepseek-r1:7b")

    # chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    result = rag_chain.invoke({"input": "summarize AMD stock performance in last five years with accurate metric?"})
    answer = result["answer"]
    context = result["context"]

    # Print full response with sources
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    


if __name__ == "__main__":
    # Run the main function
    main()
