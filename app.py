import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv('AI_Key')

# Define function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define Streamlit app
def main():
    st.title("Gutenberg AI Search")
    st.write("This app retrieves text from a URL and allows you to ask questions about the loaded data using ChatGPT.")

    # Ask user for URL input
    st.write("Please enter a URL with a link to Gutenberg book:")
    url = st.text_input("Enter URL",value="https://littlesunnykitchen.com/marry-me-chicken/")

    # Fetch file button
    if st.button("Fetch File"):
        with st.spinner("Fetching data from the URL..."):
            # Load, chunk, and index blog content
            loader = WebBaseLoader(web_paths=(url,))
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()
            prompt = hub.pull("rlm/rag-prompt")
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            # Create RAG pipeline
            st.session_state['rag_chain'] = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success("Data fetched successfully!")

    # Allow user to ask questions
    st.write("Please ask questions about the book:")
    question = st.text_area("Enter question:")

    # Submit question button
    if st.button("Submit Question"):
        if 'rag_chain' in st.session_state:
            with st.spinner("Generating answer..."):
                answer = st.session_state['rag_chain'].invoke(question)
            st.success("Answer generated successfully!")
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()
