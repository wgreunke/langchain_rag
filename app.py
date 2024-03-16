#RAG from LangChain document
#https://python.langchain.com/docs/use_cases/question_answering/quickstart
import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
print("")
print("************ New Run  ******************")
#Set the key
#os.environ["OPENAI_API_KEY"] = getpass.getpass()
#dff
#Load the env file when working locally.  For streamlit, load secrets
load_dotenv()

#print(os.getenv('test_key'))
#print(os.getenv('AI_Key'))
os.environ["OPENAI_API_KEY"]=os.getenv('AI_Key')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if 'model_built' not in st.session_state:
    st.session_state['model_built']=False

st.title("Gutenberg AI Search")
st.write("Chat GPT has indexed many books in the open domain but some books were left out")
st.write("This app lets you past a link to a Project Gutenberg text and then query it with the help of Chat GPT")

st.write("How it Works - The app grabs the text from the URL then chunks the data using langchain. ")

source_url="https://www.gutenberg.org/cache/epub/1232/pg1232-images.html"
st.write(source_url)


st.write("Please enter a URL with a link to Gutenberg book")
st_url=st.text_input("Enter URL",value=source_url,key="st_url_key")


if st.button("Fetch File"):
    source_url=st_url
    st.write("The url is ", source_url)

    #Load, chunk and index blog content
    #source_url="https://lilianweng.github.io/posts/2023-06-23-agent/"
    #source_url="https://www.gutenberg.org/cache/epub/3176/pg3176-images.html"
    loader=WebBaseLoader(
        web_paths=(source_url,),
        bs_kwargs=dict(
            #parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header"))
        ),)


    #docs=bs4.BeautifulSoup(html_doc,'html.parser')

    docs = loader.load()
    #print(docs[0])

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    #Retrieve and generate using the relevant snippets of the blog
    retriever = vectorstore.as_retriever()
    prompt=hub.pull("rlm/rag-prompt")
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    st.session_state['model_built']=True
    st.write("Session state changed")
    rag_chain=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
        #If you get this far than the model has been created.  Now you can show the loop for the query
    st.write("Rag chain is created")
    st.write(rag_chain)

    query_counter=0
    while query_counter <=10:
        st_question=st.text_input("Please ask a question about the book")
        if st_question:
            rag_answer=rag_chain.invoke(st_question)
            st.write(rag_answer)




#query_count=0

#if st.session_state['model_built']==True:
    
