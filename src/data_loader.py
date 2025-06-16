import os
import tempfile
import logging
import textwrap
from typing import List
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

logger = logging.getLogger(__name__)


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


