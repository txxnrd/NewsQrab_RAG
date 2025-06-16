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
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
from config import URLS  # URL 목록만 담고 있는 모듈
import textwrap
import logging

# ---------------------------------------------------------------------------
# 캐릭터 스타일 정의
# ---------------------------------------------------------------------------
CHARACTER_STYLE = {
    "crab": {
        "name": "crab",
        "role": "답변자",
        "mbti": "ENFJ",
        "voice": "중고음, 따뜻한 감성, 신뢰감 있는",
        "style": "긍정적이고 에너지 넘치며 정보를 능숙하게 전달하는 리더형",
        "example": "괜찮아, 내가 정리해줄게! 이건 우리가 꼭 알아야 해.",
    },
    "octopus": {
        "name": "octopus",
        "role": "답변자",
        "mbti": "ISTJ",
        "voice": "중저음, 차분한, 현실적인",
        "style": "냉철하고 통찰력 있는 백과사전형 설명",
        "example": "예상된 결과야. 기술은 늘 앞서가.",
    },
    "bok": {
        "name": "bok",
        "role": "질문자",
        "mbti": "ISFP",
        "voice": "느릿하고 순한 어버버 스타일",
        "style": "공감을 잘하며 느릿한 말투를 가진 순한 캐릭터",
        "example": "잘은 모르지만… 재밌어 보여…",
    },
    "starfish": {
        "name": "starfish",
        "role": "질문자",
        "mbti": "ENTP",
        "voice": "빠르고 튀는 발랄한 목소리",
        "style": "자기애 강하고 솔직한 관종형, 말이 빠름",
        "example": "내가 왔다!!! 흥, 나 없었으면 어쩔 뻔~? 어머~ 이건 무조건 저장각!",
    },
}

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
cross_encoder: CrossEncoder | None = None  # 전역 Cross-Encoder 객체

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
    global vectorstore, cross_encoder
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    print("===== Cross-encoder 모델 로드 중 =====")
    cross_encoder = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )
    print("===== Cross-encoder 모델 로드 완료 =====")

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
    character1: str = "starfish"  # 질문자 캐릭터 (기본값: 큐스타)
    character2: str = "crab"      # 답변자 캐릭터 (기본값: 크랩이)


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
    ▸ req.character1     : 질문자 캐릭터 (starfish, bok)
    ▸ req.character2     : 답변자 캐릭터 (crab, octopus)

    반환값:
        {
            "script": "character1: ...\ncharacter2: ...",
            "sources": [{ "source": "...", "content": "..."}, ...]
        }
    """
    global vectorstore, cross_encoder
    if vectorstore is None or cross_encoder is None:
        return {"error": "Vectorstore or CrossEncoder not initialized."}

    article_text = req.content
    original_script = req.originalScript
    character1 = req.character1
    character2 = req.character2
    logger.info(f"Character A: {req.character1}")
    logger.info(f"Character B: {req.character2}")



    # 캐릭터 스타일 가져오기
    char1 = CHARACTER_STYLE[character1]
    char2 = CHARACTER_STYLE[character2]

    # ----------------- 1) 문맥 검색 (2-stage: Retriever + Reranker) -----------------
    # 1-1) 1차 검색 (Retriever): FAISS에서 Top-100 문서 가져오기
    logger.info("1단계: FAISS에서 문서 100개 검색")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    retrieved_docs = retriever.get_relevant_documents(article_text)
    logger.info(f"  -> {len(retrieved_docs)}개 문서 검색 완료.")

    # 1-2) 2차 검색 (Reranker): Cross-encoder로 관련성 높은 Top-4 문서 재선정
    logger.info("2단계: Cross-encoder로 재점수화하여 Top-4 선정")
    # 쿼리와 문서 내용으로 페어 생성
    pairs = [[article_text, doc.page_content] for doc in retrieved_docs]

    # Cross-encoder로 점수 계산
    scores = cross_encoder.predict(pairs, show_progress_bar=True)

    # 점수와 문서를 튜플로 묶어 점수 기준 내림차순 정렬
    scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)

    # 상위 4개 문서 선택
    docs = [doc for score, doc in scored_docs[:4]]
    logger.info("  -> 재점수화 후 Top-4 문서 선정 완료.")
    for i, doc in enumerate(docs):
        logger.info(f"    Top {i+1}: {doc.metadata.get('source', 'N/A')} (Score: {scored_docs[i][0]:.4f})")
    
    context = "\n\n".join(d.page_content for d in docs)

    # ----------------- 2) 프롬프트 -----------------
    prompt = PromptTemplate(
        input_variables=["content", "originalScript", "context", "character1", "character2", "char1", "char2"],
        template=(
            "아래 뉴스 기사 내용과 이전 대화 스크립트를 참고해서, "
            "두 캐릭터의 QnA 대사를 새로 생성해주세요.\n\n"
            "조건:\n"
            "- 질문자 ({char1[name]}): {char1[style]} (예: \"{char1[example]}\")\n"
            "- 답변자 ({char2[name]}): {char2[style]} (예: \"{char2[example]}\")\n"
            "- 총 3개의 QnA로 구성해주세요. (각 QnA는 질문 + 대답 세트)\n"
            "- 각 질문과 답변은 너무 길지 않게, 한두 문장 정도의 짧고 간결한 대사로 작성해주세요.\n"
            "- 대사가 너무 설명식이 되지 않도록, 실제 캐릭터가 말하듯 자연스럽고 짧게 표현해주세요.\n"
            "형식 예시:\n"
            "{character1}: [질문1],\n"
            "{character2}: [답변1],\n\n"
            "{character1}: [질문2],\n"
            "{character2}: [답변2],\n\n"
            "{character1}: [질문3],\n"
            "{character2}: [답변3],\n\n"
            "- 대화만 출력하고, 다른 설명이나 문장은 쓰지 마세요.\n"
            "- character 이름을 제외한 모든 대사는 한국어로 작성해주세요.\n\n"
            "Content:\n"
            "{content}\n\n"
            "Original Script:\n"
            "{originalScript}"
        ),
    )


    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    final_prompt = prompt.format(
        content=article_text,
        originalScript=original_script,
        context=context,
        character1=character1,
        character2=character2,
        char1=char1,
        char2=char2
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
    print(response.content.strip())
    return RagResponse(
        script=response.content,
        sources=sources,
    )