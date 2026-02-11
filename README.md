# doc-warehouse-rag

📄 Doc Warehouse – RAG-Based PDF Question Answering System
🚀 Overview
Doc Warehouse is a Retrieval-Augmented Generation (RAG) backend system that enables intelligent question answering over uploaded PDF documents.
Users can:
Upload PDF files
Convert documents into vector embeddings
Store them in a FAISS vector database
Ask natural language questions
Receive AI-generated answers grounded in the document content
This project demonstrates practical implementation of modern AI backend architecture.
🧠 How It Works
PDF Upload
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embedding Generation (Sentence Transformers)
   ↓
FAISS Vector Store
   ↓
Similarity Search
   ↓
Gemini 2.5 Flash (LLM)
   ↓
Final AI Answer
The system ensures responses are based only on relevant document context.
🛠️ Tech Stack
Framework: FastAPI
Vector Database: FAISS
Embedding Model: all-MiniLM-L6-v2
LLM: Gemini 2.5 Flash (Google AI)
Server: Uvicorn
Language: Python
📂 Project Structure
doc_warehouse/
├── app.py              # FastAPI app & endpoints
├── rag.py              # RAG pipeline (chunking, embeddings, search, LLM)
├── ocr.py              # PDF text extraction
├── requirements.txt    # Dependencies
├── data/uploads/       # Uploaded PDFs
└── vector_store/       # FAISS index storage
