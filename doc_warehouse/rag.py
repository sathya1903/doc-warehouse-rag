import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB (in-memory FAISS for now)
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []  # stores text + metadata


def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def add_document(text: str, metadata: dict):
    for chunk in chunk_text(text):
        embedding = model.encode(chunk)
        index.add(np.array([embedding]).astype("float32"))
        documents.append({
            "text": chunk,
            "metadata": metadata
        })


def search(query: str, top_k=5):
    query_embedding = model.encode(query)
    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        top_k
    )

    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx])

    return results


def ask_llm(question: str, context_docs: list):
    context = "\n\n".join([doc["text"] for doc in context_docs])

    prompt = f"""
Use the following documents to answer the question.

Documents:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
