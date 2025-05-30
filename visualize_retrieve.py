"""
visualize_retrieve.py ── 현재 RAG 시스템의 FAISS similarity search 3-D PCA 시각화
──────────────────────────────────────────────────────────────────────────────────
• query_text     : 검색하고 싶은 질문
• k              : top-k 이웃 개수 (쿼리 포함하면 k+1 점)
• show_content   : True면 문서 내용 일부도 출력
결과 파일        : retrieve_visualization.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (필수 import)
from sklearn.decomposition import PCA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import matplotlib.font_manager as fm

# 환경 변수 로드
load_dotenv()

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = ['AppleGothic']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────── 1. 파라미터 ──────────────────────────────
query_text = "김정은과 북한의 핵 개발에 대해서"  # 검색 질문
k = 10  # top-k 이웃 (전체를 보고 싶으면 더 크게 설정)
show_content = True  # 문서 내용 미리보기 출력 여부
VECTORSTORE_PATH = "faiss_index"

def main():
    # ─────────────────────── 2. 기존 벡터스토어 로드 ───────────────────────
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ 벡터스토어 경로가 존재하지 않습니다: {VECTORSTORE_PATH}")
        print("먼저 main.py 서버를 실행하여 벡터스토어를 생성하세요.")
        return

    print("===== 기존 벡터스토어 로드 중 =====")
    embeddings = OpenAIEmbeddings()
    
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"✓ 벡터스토어 로드 완료 (총 {vectorstore.index.ntotal}개 문서)")
    except Exception as e:
        print(f"❌ 벡터스토어 로드 실패: {e}")
        return

    # ─────────────────────── 3. 쿼리 임베딩 & 검색 ────────────────────────
    print(f"\n===== 쿼리 검색: '{query_text}' =====")
    
    # 쿼리 임베딩
    query_embedding = np.array(embeddings.embed_query(query_text)).astype("float32")
    
    # FAISS 인덱스에서 직접 검색
    faiss_index = vectorstore.index
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    
    print(f"✓ Top-{k} 문서 검색 완료")
    
    # ─────────────────────── 4. 검색된 문서들의 임베딩 추출 ────────────────────
    # FAISS에서 모든 임베딩을 가져올 수 있는지 확인하고, 검색된 문서들만 추출
    topk_embeddings = []
    topk_documents = []
    
    # retriever를 통해 문서 내용도 가져오기
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query_text)
    
    # 검색된 인덱스의 임베딩들 추출
    for i, idx in enumerate(indices[0]):
        emb = faiss_index.reconstruct(int(idx))
        topk_embeddings.append(emb)
        
        # 문서 내용과 메타데이터
        if i < len(retrieved_docs):
            doc = retrieved_docs[i]
            topk_documents.append({
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata,
                'distance': distances[0][i],
                'index': idx
            })
    
    topk_embeddings = np.array(topk_embeddings)
    
    # 쿼리 + 문서 임베딩 결합
    all_embeddings = np.vstack([query_embedding, topk_embeddings])  # (k+1, embedding_dim)
    
    # ─────────────────────── 5. 검색 결과 출력 ────────────────────────
    if show_content:
        print(f"\n===== 검색된 Top-{k} 문서들 =====")
        for i, doc_info in enumerate(topk_documents):
            print(f"\n📄 문서 {i+1} (거리: {doc_info['distance']:.4f}, 인덱스: {doc_info['index']})")
            print(f"   메타데이터: {doc_info['metadata']}")
            print(f"   내용: {doc_info['content']}")
    
    # ───────────────────────────── 6. 3D PCA 시각화 ──────────────────────────────
    print(f"\n===== PCA 차원 축소 ({all_embeddings.shape[1]}D → 3D) =====")
    pca = PCA(n_components=3, random_state=42)
    xyz = pca.fit_transform(all_embeddings)
    
    # 설명된 분산 비율 출력
    explained_variance = pca.explained_variance_ratio_
    print(f"✓ PCA 완료 - 설명된 분산: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}, PC3={explained_variance[2]:.3f}")
    print(f"✓ 총 설명된 분산: {sum(explained_variance):.3f}")
    
    # ───────────────────────────── 7. 3D 시각화 ────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', facecolor="white")
    
    # 색상 설정
    colors = plt.cm.tab10(np.linspace(0, 1, k+1))
    
    # (a) 점과 텍스트 그리기
    for i, (x, y, z) in enumerate(xyz):
        is_query = (i == 0)
        
        if is_query:
            # 쿼리 점
            ax.scatter(x, y, z, s=150, color="red", marker="★", 
                      label="Query", edgecolors="black", linewidths=2)
            ax.text(x, y, z + 0.1, "Q", fontsize=12, weight="bold", color="red")
        else:
            # 문서 점
            doc_type = topk_documents[i-1]['metadata'].get('type', 'unknown')
            marker = "o" if doc_type == "namu_wiki" else "s"  # 원형(나무위키) vs 사각형(PDF)
            
            ax.scatter(x, y, z, s=80, color=colors[i], marker=marker,
                      label=f"Doc {i} ({doc_type})", alpha=0.8)
            ax.text(x, y, z + 0.05, str(i), fontsize=10, weight="bold")
    
    # (b) 쿼리에서 각 문서로 점선 연결
    query_point = xyz[0]
    for i in range(1, len(xyz)):
        doc_point = xyz[i]
        ax.plot([query_point[0], doc_point[0]],
                [query_point[1], doc_point[1]],
                [query_point[2], doc_point[2]],
                "--", color="gray", alpha=0.6, linewidth=1)
    
    # (c) 축 설정 및 제목
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
    ax.set_zlabel(f"PC3 ({explained_variance[2]:.1%})")
    
    ax.set_title(f"RAG Retrieve 시각화: '{query_text[:30]}...'\n"
                f"Top-{k} 문서와 쿼리의 3D 임베딩 공간", fontsize=14, pad=20)
    
    # 범례 설정 (너무 많으면 생략)
    if k <= 10:
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    
    plt.tight_layout()
    
    # ───────────────────────────── 8. 저장 및 출력 ────────────────────────────
    output_filename = "retrieve_visualization.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 시각화 결과 저장: {output_filename}")
    
    # 추가 통계 정보
    print(f"\n===== 요약 통계 =====")
    print(f"총 문서 수: {vectorstore.index.ntotal}")
    print(f"검색된 문서 수: {k}")
    print(f"임베딩 차원: {all_embeddings.shape[1]}")
    print(f"쿼리와 가장 가까운 문서 거리: {min(distances[0]):.4f}")
    print(f"쿼리와 가장 먼 문서 거리: {max(distances[0]):.4f}")
    
    # 문서 타입별 분포
    doc_types = [doc['metadata'].get('type', 'unknown') for doc in topk_documents]
    type_counts = {}
    for doc_type in doc_types:
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    print(f"문서 타입 분포: {type_counts}")

if __name__ == "__main__":
    main() 