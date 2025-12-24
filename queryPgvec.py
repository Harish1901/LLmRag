# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:15:55 2025

@author: haris
"""


import psycopg
import numpy as np

# Connection details
conn = psycopg.connect("postgresql://postgres:test@localhost:5432/vector_lgrag")

# Example query text converted to embedding (you can use your LangChain embedding model)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

query_text = "How to install french door"
query_vector = embeddings.embed_query(query_text)  # returns a list of floats

# Convert to PostgreSQL vector format
query_vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

# Raw SQL similarity search
sql = f"""
SELECT id, document,cmetadata, embedding <=> '{query_vector_str}' AS distance
FROM public.langchain_pg_embedding
ORDER BY embedding <=> '{query_vector_str}'
LIMIT 1;
"""

with conn.cursor() as cur:
    cur.execute(sql)
    results = cur.fetchall()

# Print results
for row in results:
    print(f"ID: {row[0]}, Distance: {row[3]}")
    print(f"Content: {row[1][:200]}...")  # show first 200 chars
    print(f"Metadata: {row[2]}")
    print("-" * 50)

