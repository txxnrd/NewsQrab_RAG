"""
rag_server.py
==============
FastAPI RAG 서버 – 기사 전문 + 기존 대사(Q&A 스크립트)를 받아
벡터 검색 문맥과 함께 '개선된 스크립트'를 생성하여 반환합니다.
"""

import os
import tempfile
from typing import List
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
# (torchvision 연산자 누락 오류 방지를 위해) transformers 가 torchvision 을 import 하지 않도록 환경변수 설정
os.environ["DISABLE_TORCHVISION_IMPORTS"] = "1"
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import BaseModel
from config import URLS  # URL 목록만 담고 있는 모듈
import textwrap
import logging

# ---------------------------------------------------------------------------
# 로깅 설정 (모듈 상단 한 번만)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()  # OPENAI_API_KEY 등 환경 변수 읽기

# ---------------------------------------------------------------------------
# FastAPI 인스턴스
# ---------------------------------------------------------------------------

app = FastAPI()

# ---------------------------------------------------------------------------
# 벡터스토어 유틸
# ---------------------------------------------------------------------------

VECTORSTORE_PATH = "faiss_index"
vectorstore: FAISS | None = None  # 전역 벡터스토어 객체

def load_pdf_from_url(url: str) -> Document:
    """PDF URL에서 텍스트를 추출하여 Document로 반환 (디버깅 로그 포함)"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        try:
            doc = fitz.open(tmp_file_path)
            text_content = ""
            total_chars = 0

            for page in doc:
                raw_text = page.get_text().strip()
                cleaned_text = raw_text  # 필요하면 clean_pdf_text(raw_text)

                if cleaned_text:
                    # ────────────────────────────────────────────────
                    # 📌 ① 페이지별 길이·앞부분 미리보기 출력
                    # ────────────────────────────────────────────────
                    logger.info(
                        "Page %-3d | %5d chars | Preview: %s",
                        page.number + 1,
                        len(cleaned_text),
                        textwrap.shorten(cleaned_text, width=80, placeholder=" …"),
                    )

                    text_content += f"페이지 {page.number + 1}: {cleaned_text}\n\n"
                    total_chars += len(cleaned_text)

            

            # ────────────────────────────────────────────────────────
            # 📌 ② 전체 추출 결과 요약
            # ────────────────────────────────────────────────────────
            logger.info(
                "Finished extracting PDF (%s) → total %d chars across %d pages",
                url,
                total_chars,
                len(doc),
            )
            doc.close() 
            # arXiv 논문이면 출처 주석 추가
            if "arxiv.org" in url.lower():
                text_content = f"arXiv 논문 출처: {url}\n\n" + text_content

            return Document(page_content=text_content,
                            metadata={"source": url, "type": "pdf"})

        finally:
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.exception("[ERROR] PDF %s 로딩 실패: %s", url, e)
        return Document(page_content="",
                        metadata={"source": url, "type": "pdf"})

def load_namu_page(url: str) -> Document:
    """나무위키 페이지 본문을 Document 로 로드"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("main") or soup
        content = content_div.get_text(separator="\n", strip=True)
        return Document(page_content=content, metadata={"source": url, "type": "namu_wiki"})
    except Exception as e:
        print(f"[ERROR] {url} 로딩 실패: {e}")
        return Document(page_content="", metadata={"source": url, "type": "namu_wiki"})


def load_document_from_url(url: str) -> Document:
    """URL 타입에 따라 적절한 로더를 선택하여 Document 반환"""
    parsed_url = urlparse(url)
    
    # PDF 파일인지 확인
    if url.lower().endswith('.pdf') or 'arxiv.org/pdf/' in url.lower():
        return load_pdf_from_url(url)
    # 나무위키 페이지인지 확인
    elif 'namu.wiki' in parsed_url.netloc:
        return load_namu_page(url)
    else:
        # 기본적으로 웹 페이지로 처리
        return load_namu_page(url)


def chunk_documents(docs: List[Document], chunk_size: int = 1000):
    """문서 배열을 chunk_size 단위로 yield"""
    for i in range(0, len(docs), chunk_size):
        yield docs[i : i + chunk_size]


def create_faiss_vectorstore(
    docs: List[Document], embeddings, batch_size: int = 500
) -> FAISS:
    """배치 임베딩으로 대용량 문서를 FAISS 스토어로 인덱싱"""
    vector_store: FAISS | None = None
    doc_count = 0
    for chunked_docs in chunk_documents(docs, batch_size):
        # ────────────────────────────────────────────────────────
        # 📌 각 청크 임베딩 진행 상황 로깅
        # ────────────────────────────────────────────────────────
        logger.info(f"임베딩 중: 문서 {doc_count + 1} ~ {doc_count + len(chunked_docs)}")

        # PDF 문서가 포함되어 있는지, 있다면 몇 개인지 로깅
        pdf_docs_in_chunk = [doc for doc in chunked_docs if doc.metadata.get("type") == "pdf"]
        if pdf_docs_in_chunk:
            logger.info(f"  └ 이 청크에 PDF 문서 {len(pdf_docs_in_chunk)}개 포함:")
            for i, pdf_doc in enumerate(pdf_docs_in_chunk[:3]): # 처음 3개 PDF 소스만 로깅 (너무 많으면 생략)
                logger.info(f"    PDF [{i+1}] 소스: {pdf_doc.metadata.get('source', 'N/A')}, 내용 일부: {textwrap.shorten(pdf_doc.page_content, width=50, placeholder='...')}")
        
        partial = FAISS.from_documents(chunked_docs, embeddings)
        if vector_store is None:
            vector_store = partial
        else:
            vector_store.merge_from(partial)
        doc_count += len(chunked_docs)
    logger.info(f"총 {doc_count}개 문서 청크 임베딩 완료.")
    return vector_store


# ---------------------------------------------------------------------------
# FastAPI – 애플리케이션 시작 시 벡터스토어 준비
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    global vectorstore
    embeddings = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-multilingual-gemma2",
        model_kwargs={"device": "mps"},
    )

    if os.path.exists(VECTORSTORE_PATH):
        print("===== 기존 벡터스토어 로드 중 =====")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print("===== 벡터스토어 새로 생성 중 =====")
        # 1) URL → HTML → Document
        documents = [load_document_from_url(url) for url in URLS]
        documents = [doc for doc in documents if doc.page_content.strip()]
        # 2) Document → 작은 조각
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        docs = splitter.split_documents(documents)
        # 3) FAISS 인덱싱
        vectorstore = create_faiss_vectorstore(docs, embeddings)
        # 4) 디스크 저장
        vectorstore.save_local(VECTORSTORE_PATH)
        print("===== 벡터스토어 생성 및 저장 완료 =====")


# ---------------------------------------------------------------------------
# 요청/응답 모델
# ---------------------------------------------------------------------------

class RagRequest(BaseModel):
    content: str          # 기사 전문
    originalScript: str   # 기존 Q&A 스크립트


class RagResponse(BaseModel):
    script: str
    sources: List[dict]


# ---------------------------------------------------------------------------
# 엔드포인트: /rag
# ---------------------------------------------------------------------------

@app.post("/rag", response_model=RagResponse)
async def generate_rag_script(req: RagRequest):
    """
    ▸ req.content        : 뉴스 기사 본문
    ▸ req.originalScript : 기존 user1:, user2:… 형태 스크립트

    반환값:
        {
            "script": "user1: ...\nuser2: ...",
            "sources": [{ "source": "...", "content": "..."}, ...]
        }
    """
    global vectorstore
    if vectorstore is None:
        return {"error": "Vectorstore not initialized."}

    article_text = req.content
    original_script = req.originalScript

    # ----------------- 1) 문맥 검색 -----------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(article_text)
    context = "\n\n".join(d.page_content for d in docs)

    # ----------------- 2) 프롬프트 -----------------
    prompt = PromptTemplate(
    input_variables=["content", "originalScript","context"],
    template=(
        "아래 뉴스 기사 내용과 이전 대화 스크립트를 참고해서, "
        "두 캐릭터의 QnA 대사를 새로 생성해주세요.\n\n"
        "조건:\n"
        "- 질문자 (user1): 호기심 많고 직설적\n"
        "- 답변자 (user2): 친절하고 쉽게 설명\n"
        "- 총 3개의 QnA로 구성해주세요. (각 QnA는 질문 + 대답 세트)\n"
        "- 반드시 아래 형식을 정확히 지켜 작성:\n"
        "user1: [질문1]  \n"
        "user2: [답변1]  \n\n"
        "user1: [질문2]  \n"
        "user2: [답변2]  \n\n"
        "user1: [질문3]  \n"
        "user2: [답변3]  \n\n"
        "- 대화만 출력하고, 다른 설명이나 문장은 쓰지 마세요.\n"
        "- 모든 대사는 한국어로 작성해주세요.\n\n"
        "Content:\n"
        "{content}\n\n"
        "Original Script:\n"
        "{originalScript}"
    ),
)


    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    final_prompt = prompt.format(
    content=article_text,          # {content}
    originalScript=original_script # {originalScript}
)
    response = llm.invoke(final_prompt)

    # ----------------- 3) 응답 -----------------
    sources = [
        {
            "source": doc.metadata.get("source", "N/A"),
            "content": doc.page_content[:300],
        }
        for doc in docs
    ]
    print(sources)
    return RagResponse(
        script=response.content.strip(),
        sources=sources,
    )