import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from config import URLS

app = FastAPI()
load_dotenv()

class ArticleRequest(BaseModel):
    content: str

def load_namu_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("main") or soup
        content = content_div.get_text(separator="\n", strip=True)
        return Document(page_content=content, metadata={"source": url})
    except Exception as e:
        print(f"[ERROR] {url} 로딩 실패: {e}")
        return Document(page_content="", metadata={"source": url})

# 여러 배치를 합쳐서 FAISS VectorStore를 만드는 함수
def chunk_documents(docs, chunk_size=1000):
    for i in range(0, len(docs), chunk_size):
        yield docs[i : i + chunk_size]

def create_faiss_vectorstore(docs, embeddings, batch_size=500):
    vectorstore = None
    for chunked_docs in chunk_documents(docs, batch_size):
        partial_store = FAISS.from_documents(chunked_docs, embeddings)
        if vectorstore is None:
            vectorstore = partial_store
        else:
            vectorstore.merge_from(partial_store)
    return vectorstore

# 전역 변수
VECTORSTORE_PATH = "faiss_index"
vectorstore = None

@app.on_event("startup")
def startup_event():
    global vectorstore
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(VECTORSTORE_PATH):
        print("===== 기존 벡터스토어 로드 중 =====")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
    else:
        print("===== 벡터스토어 새로 생성 중 =====")
        # 1. 문서 로딩
        documents = [load_namu_page(url) for url in URLS]
        documents = [doc for doc in documents if doc.page_content.strip()]
        # 2. 텍스트 쪼개기
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        # 3. FAISS VectorStore 생성 (배치 Embedding)
        vectorstore = create_faiss_vectorstore(docs, embeddings, batch_size=500)
        # 4. 로컬에 저장
        vectorstore.save_local(VECTORSTORE_PATH)
        print("===== 벡터스토어 생성 및 저장 완료 =====")

@app.post("/rag")
async def generate_summary(request: ArticleRequest):
    user_input = request.content
    
    # 이미 로드된 vectorstore 사용
    global vectorstore
    if not vectorstore:
        return {"error": "Vectorstore not initialized."}
    
    # RAG 체인 구성
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
        retriever=retriever,
        return_source_documents=True,
    )

    # 쿼리 수행
    result = qa_chain.invoke({"query": user_input})
    answer = result["result"]
    sources = [
        {
            "source": doc.metadata.get("source", "N/A"),
            "content": doc.page_content[:300]
        }
        for doc in result["source_documents"]
    ]

    return {
        "summary": answer,
        "sources": sources
    }
