import os
from dotenv import load_dotenv
from langchain.globals import *
from langchain_openai import ChatOpenAI

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts.chat import SystemMessagePromptTemplate

# ===============================================================
# Pinecone Question-Answering System
# This script does a semantic search of a named Pinecone index
# to retrieve relevant documents based on a user's question.
# The retrieved documents are then used to generate a response
# using a language model.
# 
# You can load data into the Pinecone index using the
# `pdf_to_pinecone.py` script.
# ===============================================================

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "graph-powered-machine-learning"

llm = ChatOpenAI(model="gpt-4o")

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

prompt = SystemMessagePromptTemplate.from_template("You are an assistant for question-answering tasks. Use only the context provided to answer the question.  Do not make up your own answer. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nContext: {context} \nQuestion: {question}")

print("Querying Pinecone Index: ", PINECONE_INDEX_NAME)
print("Ready to answer questions. Type 'exit' to quit.")

question = ""
while(question != "exit"):
    question = input("Ask a question: ")
    if question == "exit":
        print("Bye!")
        break
    
    retrieved_docs = vector_store.similarity_search(question, k=3)
    print("Retrieved info from pages: ", [doc.metadata["page_label"] for doc in retrieved_docs])
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    messages = prompt.format_messages(question=question, context=docs_content)
    response = llm.invoke(messages)
    print("Answer: ", response.content)
