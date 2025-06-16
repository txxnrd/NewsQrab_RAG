from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

def rerank_documents(
    query: str, retrieved_docs: List[Document], cross_encoder: CrossEncoder, top_k: int = 4
) -> List[Document]:
    """
    Cross-encoder를 사용하여 검색된 문서 목록의 순위를 재조정합니다.

    Args:
        query (str): 사용자 쿼리.
        retrieved_docs (List[Document]): 1차 검색된 문서 목록.
        cross_encoder (CrossEncoder): 재순위에 사용할 Cross-encoder 모델.
        top_k (int): 반환할 상위 문서 개수.

    Returns:
        List[Document]: 재순위화된 상위 k개 문서 목록.
    """
    logger.info(f"2단계: Cross-encoder로 {len(retrieved_docs)}개 문서 재점수화하여 Top-{top_k} 선정")
    
    # 쿼리와 문서 내용으로 페어 생성
    pairs = [[query, doc.page_content] for doc in retrieved_docs]

    # Cross-encoder로 점수 계산
    scores = cross_encoder.predict(pairs, show_progress_bar=True)

    # 점수와 문서를 튜플로 묶어 점수 기준 내림차순 정렬
    scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)

    # 상위 k개 문서 선택
    docs = [doc for score, doc in scored_docs[:top_k]]
    
    logger.info(f"  -> 재점수화 후 Top-{top_k} 문서 선정 완료.")
    for i, doc in enumerate(docs):
        logger.info(f"    Top {i+1}: {doc.metadata.get('source', 'N/A')} (Score: {scored_docs[i][0]:.4f})")
        
    return docs


