from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import uvicorn

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = "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"

app = FastAPI()

def load_and_chunk_documents(file_paths: List[str]):
    documents = []
    for path in file_paths:
        print(f"Loading document: {path}")
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            print(f"Skipping unsupported file: {path}")
            continue
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from: {path}")
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Chunking completed. Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    print("Creating vector store using HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("Vector store created.")
    return vectordb

def get_documents():
    folder_path = r"C:\Users\yaman\rag_assignment_documents"
    all_items = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Found {len(files)} files in document folder.")
    return files

document_paths = get_documents()
chunks = load_and_chunk_documents(document_paths)
vectordb = create_vector_store(chunks)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

print("Initializing ChatOpenAI model...")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
    temperature=0,
    model="openai/gpt-3.5-turbo"
)
print("Model initialized.")

print("Building ConversationalRetrievalChain...")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True
)
print("Chain created successfully.")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_question(req: QueryRequest):
    try:
        print(f"Received query: {req.query}")
        result = qa_chain.invoke({"question": req.query})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        print(f"Answer generated. Source documents retrieved: {len(sources)}")
        return JSONResponse(content={
            "response": answer,
            "sources": [s.metadata for s in sources]
        })
    except Exception as e:
        print(f"Error occurred while processing query: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("Starting FastAPI app with Uvicorn...")
    uvicorn.run("test:app", host="0.0.0.0", port=8000)
