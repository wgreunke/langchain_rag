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
print("")
print("************ New Run  ******************")
#Set the key


#Load, chunk and index blog content
#source_url="https://lilianweng.github.io/posts/2023-06-23-agent/"
source_url="https://www.gutenberg.org/cache/epub/3176/pg3176-images.html"
loader=WebBaseLoader(
    web_paths=(source_url,),
    bs_kwargs=dict(
        #parse_only=bs4.SoupStrainer(class_=("post-content","post-title","post-header"))
    ),)


#docs=bs4.BeautifulSoup(html_doc,'html.parser')

docs = loader.load()
print(docs[0])

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

#Retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever()
prompt=hub.pull("rlm/rag-prompt")
llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#print(rag_chain.invoke("What is Task Decomposition"))
#print(rag_chain.invoke("What was the itenerary of the trip?"))

text_input=""
while text_input !="quit":
    text_input=input("Ask me a question\n")
    print(rag_chain.invoke(text_input))
