from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
import os
from langchain_community.document_loaders import WebBaseLoader
import bs4
app = FastAPI()
load_dotenv()

# 요청을 받을 데이터 모델
class ArticleRequest(BaseModel):
    content: str

@app.post("/rag")
async def generate_summary(request: ArticleRequest):
    content = request.content
    
    # 간단한 요약 예제 (실제 RAG 로직을 여기에 추가)
    summary = f"요약: {content[:100]}..."  # 앞 100자 사용
    return {"summary": summary}  # JSON 응답 반환


