from fastapi import FastAPI, UploadFile, File
from ocr import extract_text_from_pdf
from rag import add_document, search, ask_llm
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)

    metadata = {
        "filename": file.filename
    }

    add_document(text, metadata)

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename
    }


@app.post("/chat")
async def chat(question: str):
    relevant_docs = search(question)
    answer = ask_llm(question, relevant_docs)

    return {
        "question": question,
        "answer": answer
    }
