import os
import sys
import time
import shutil
import logging
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SYSTEM_NODE_IO: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FAANG_RAG_CORE")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from guardrails import SecurityGuardrail
from memory_manager import SessionMemoryManager
from evaluator import RAGEvaluator
from feedback_logger import FeedbackLogger

os.environ["GROQ_API_KEY"] = "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"

app = FastAPI(title="FAANG Distributed Neural Retrieval Engine", version="5.0.0")
memory = SessionMemoryManager()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_cluster_user"

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str

@app.post("/api/v5/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v5/ingest")
def ingest_data():
    try:
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            return {"status": "empty", "total_chunks": 0}
        loader = PyPDFDirectoryLoader(DATA_DIR)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=450, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
        return {"status": "success", "total_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v5/query")
def query_rag(request: QueryRequest):
    start_time = time.time()
    if not SecurityGuardrail.verify_query_safety(request.question):
        return {
            "answer": "⚠️ TRANSACTION TERMINATED: Violation of Security Firewalls. Process intercepted.",
            "sources": [], "chart_data": [], "metrics": {"latency_sec": 0.001, "confidence_score": 0.0, "security_state": "BLOCKED"}
        }
    try:
        if not os.path.exists(DB_DIR):
            raise HTTPException(status_code=400, detail="Core vector space missing.")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        initial_docs = vector_db.similarity_search(request.question, k=10)
        if not initial_docs:
            return {"answer": "No overlapping vectors discovered.", "sources": [], "chart_data": [], "metrics": {}}
        pairs = [[request.question, doc.page_content] for doc in initial_docs]
        rerank_scores = reranker.predict(pairs)
        reranked_results = sorted(zip(initial_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        top_3 = reranked_results[:3]
        final_docs = [item[0] for item in top_3]
        confidence_score = float(top_3[0][1]) if top_3 else 0.0
        context = "\n\n".join([d.page_content for d in final_docs])
        sources = [f"Page {d.metadata.get('page', 0) + 1} ({os.path.basename(d.metadata.get('source', 'Doc'))})" for d in final_docs]
        chart_data = [{"Rank": idx + 1, "Source_Context": f"P. {doc.metadata.get('page', 0) + 1}", "Neural_Weight": round(float(score), 4)} for idx, (doc, score) in enumerate(reranked_results)]
        past_history = memory.fetch_history(request.session_id, limit=4)
        messages = [("system", f"You are a Principal AI System Engineer. Synthesize a professional response using this context:\n\n{context}")]
        for msg in past_history:
            messages.append((msg["role"], msg["content"]))
        messages.append(("user", request.question))
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        response = llm.invoke(messages)
        memory.save_message(request.session_id, "user", request.question)
        memory.save_message(request.session_id, "assistant", response.content)
        eval_metrics = RAGEvaluator.compute_precision_metrics(response.content, context)
        execution_time = round(time.time() - start_time, 3)
        return {
            "answer": response.content,
            "sources": list(set(sources)),
            "chart_data": chart_data,
            "metrics": {
                "latency_sec": execution_time,
                "confidence_score": round(confidence_score, 4),
                "chunks_scanned": len(initial_docs),
                "faithfulness": eval_metrics["faithfulness"],
                "context_recall": eval_metrics["context_recall"],
                "security_state": "SECURE"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v5/feedback")
def log_feedback(request: FeedbackRequest):
    FeedbackLogger.log_alignment_signal(request.query, request.answer, request.rating)
    return {"status": "logged"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
