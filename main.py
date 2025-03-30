from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import URLS
import logging

app = FastAPI()
load_dotenv()

# 요청을 받을 데이터 모델
class ArticleRequest(BaseModel):
    content: str

@app.post("/rag")
async def generate_summary(request: ArticleRequest):
    user_input = request.content

    # 🔹 1. 관련 문서 웹에서 로딩 (예: 특정 뉴스 url)
    loader = WebBaseLoader(URLS)
    documents = loader.load()

    # 🔹 2. 문서 쪼개기
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # 🔹 3. 벡터 저장소 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 🔹 4. RAG 체인 구성
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        retriever=retriever,
        return_source_documents=False
    )

    # 🔹 5. 사용자 입력을 기반으로 RAG 수행
    result = qa_chain.invoke({"query": user_input})

    return {"summary": result["result"]}


