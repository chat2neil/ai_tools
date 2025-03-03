import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

PDF_FILE = "./data/Graph-Powered_Machine_Learning.pdf"
PINECONE_INDEX_NAME = os.path.splitext(os.path.basename(PDF_FILE))[0].lower().replace("_", "-").replace(" ", "-")

print(f"Loading PDF file: {PDF_FILE}")
pdf_loader = PyPDFLoader(PDF_FILE)
docs = []
for doc in pdf_loader.load():
    docs.append(doc)

pinecone = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating a new index: {PINECONE_INDEX_NAME}")
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pinecone.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pinecone.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

print("Loading data into Pinecone")
uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

print("Data loaded.  Now let's do some queries!")

results = vector_store.similarity_search_with_score(
    "what are embeddings?", k=2
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

print("Done!")
