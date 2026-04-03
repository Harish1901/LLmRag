# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:14:11 2025. I have successfully created my local branch

@author: haris
"""
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
# %% read files
def get_pdf_files(target_directory):
    path = Path(target_directory)
    
    # 1. Check if the directory exists and is actually a folder
    if path.is_dir():
        # 2. Find all files ending in .pdf
        # glob() returns a generator; we convert it to a list
        pdf_files = list(path.glob("*.pdf"))
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF(s) in {target_directory}")
            return pdf_files
        else:
            print("Folder exists, but no PDF files were found.")
            return []
    else:
        print(f"The folder '{target_directory}' does not exist.")
        return []

# Your specific path
folder_path = os.getcwd()
print("folderpath: ", folder_path)
pdfs = get_pdf_files(folder_path)

# Example: iterate through found PDFs
for pdf in pdfs:
    print(f"Loading: {pdf.name}")

# %% Load all documents
docs = []
for pdf in pdfs:
    # The PyPDFLoader cannot find the specified file because it's not uploaded or the path is incorrect.
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())  # Add pages from each PDF

print(f"Total pages loaded: {len(docs)}")

# Assuming you already loaded docs using PyPDFLoader
first_page_text = docs[0].page_content  # Get text from the first page
wordword_count = len(first_page_text.split())  # Split by whitespace and count
print(wordword_count)
# %%
#get word count from each page and store in dataframe
# Create a list of dictionaries with page info and word count
pageBypage_data = []
for doc in docs:
    text = doc.page_content
    word_count = len(text.split())
    pageBypage_data.append({
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", "unknown"),
        "word_count": word_count
    })

# Convert to DataFrame
pageData_df = pd.DataFrame(pageBypage_data)
# Save the DataFrame
pageData_df.to_csv('pageData_df.csv', index=False)

# %% Chunking
#https://medium.com/@anixlynch/7-chunking-strategies-for-langchain-b50dac194813
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(chunks)
#print("\n\n" in docs[0].page_content)
# docs is your list of 132 LangChain Document objects
double_newline_count = sum(1 for doc in docs if "\n\n" in doc.page_content)
print(f"Documents with double newlines: {double_newline_count} out of {len(docs)}")

# %% Chunking 
#https://docs.langchain.com/oss/javascript/integrations/splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(len(chunks))

# %% Chunking pip install -qU langchain-postgres( create PGVEC )
#https://www.youtube.com/watch?v=FDBnyJu_Ndg
import json
import os
# Load API key from JSON file
with open("credentials.json", "r") as f:
    config = json.load(f)
# Set environment variable
os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
# Initialize embeddings# Initialize embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
from langchain_postgres import PGVector
CONNECTION = "postgresql+psycopg://postgres:test@localhost:5432/vector_lgrag"  # Uses psycopg3!
COLLECTION_NAME = "lgvec"
vector_store = PGVector(embeddings = embeddings, collection_name=COLLECTION_NAME,connection =CONNECTION )
ids = vector_store.add_documents(documents=chunks)
# %% similarity search not cosine or L2 due to table size
results = vector_store.similarity_search(
    "How to install french door", k = 2
)

#print(results)

# %% tool 
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=1)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # Return both text and metadata explicitly
    return serialized, retrieved_docs

# %% Agent

import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

tools = [retrieve_context]

# Read prompt from file
with open("refrigerator_agent_prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()
agent = create_agent(model, tools, system_prompt=prompt)

# query = (
# "How do I replace the water filter in my LG French Door refrigerator?"
# )
# %% API
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    query: str
    
class AgentResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]

@app.post("/ask", response_model=AgentResponse)
async def ask_ai(request: QueryRequest):
    raw_response = agent.invoke({
        "messages": [{"role": "user", "content": request.query}]
    })

    # Find ToolMessage
    tool_messages = [
        msg for msg in raw_response.get("messages", [])
        if msg.__class__.__name__ == "ToolMessage"
    ]

    if tool_messages:
        tool_msg = tool_messages[-1]  # take the last ToolMessage
        content = tool_msg.content
        
        # Dynamically extract everything after "Content:"
        start_marker = "Content:"
        if start_marker in content:
            start_index = content.index(start_marker) + len(start_marker)
            content = content[start_index:].strip()
            content = content.replace("\n", " ")


        # Extract metadata from artifacts
        retrieved_metadata = []
        for doc in getattr(tool_msg, "artifact", []):
            retrieved_metadata.append(doc.metadata)
    else:
        content = "No tool output found."
        retrieved_metadata = []

    return AgentResponse(
        content=content,
        metadata={"retrieved_chunks": retrieved_metadata},
    )










