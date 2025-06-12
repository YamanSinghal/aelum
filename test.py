from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import uvicorn

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = "sample"

app = FastAPI()

def load_and_chunk_documents(file_paths: List[str]):
    documents = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print("chunking done")
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("created vector db")
    return vectordb


def get_documents():
    folder_path = r"C:\Users\yaman\rag_assignment_documents"
    all_items = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    return files

document_paths = get_documents()
print("paths ", document_paths)
chunks = load_and_chunk_documents(document_paths)
vectordb = create_vector_store(chunks)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_question(req: QueryRequest):
    try:
        result = qa_chain.invoke({"question": req.query})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        return JSONResponse(content={
            "response": answer,
            "sources": [s.metadata for s in sources]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)
