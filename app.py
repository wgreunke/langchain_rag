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

#os.environ["OPENAI_API_KEY"]=os.getenv('AI_Key')
os.environ["OPENAI_API_KEY"]=secrets["GPT_Key"]


# Define function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define Streamlit app
def main():
    st.title("Gutenberg AI Search")
    st.write("ChatGpt allows you to query many books that are in the public domain howerver there are some books in Project Gutemberg that were not included in the model.")
    st.write("This app uses the text from a book in the Gutenberg library and allows you query the book just like ChatGPT.")
    st.write("To start, go to the Gutenberg Library, find a book and copy the Read Online URL to a book you want to query.")
    st.link_button("Go to Gutenberg Library", "https://www.gutenberg.org")
    st.write("Sample Book")
    st.wriate("https://www.gutenberg.org/cache/epub/73170/pg73170-images.html")
    #st.write("image")
    # Ask user for URL input
    url = st.text_input("Enter URL")

    # Fetch file button
    if st.button("Fetch Book"):
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
    #st.write("Please ask a question about the book:")
    question = st.text_input("Enter question:")

    # Submit question button
    if st.button("Submit Question"):
        if 'rag_chain' in st.session_state:
            with st.spinner("Generating answer..."):
                answer = st.session_state['rag_chain'].invoke(question)
            st.success("Answer generated successfully!")
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()
