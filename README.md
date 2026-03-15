A retrieval-augmented generation (RAG) application that processes PDF documents, generates semantic embeddings, stores them in a PostgreSQL vector database, and uses LangChain agents with intelligent retrieval tools to answer user queries with context-aware responses powered by Google Generative AI.

Quick Start
Install dependencies: pip install langchain langchain-google-genai langchain-postgres pydantic fastapi pandas
Add Google API key to credentials.json
Place PDF files in the working directory
Run python lGRagApp.py
Tech Stack
LangChain for RAG orchestration and agents
Google Generative AI for embeddings and LLM reasoning
PostgreSQL + PGVector for vector similarity search
FastAPI for API backend (partial)
Features
End-to-end PDF → embeddings → retrieval → LLM reasoning pipeline
Semantic similarity search over documents
LangChain @tool functions for structured retrieval
Custom prompt engineering for domain-specific agents
Intelligent document chunking with metadata preservation
