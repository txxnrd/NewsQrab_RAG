"""
visualize_retrieve.py ── 현재 RAG 시스템의 FAISS similarity search 3-D PCA 시각화
──────────────────────────────────────────────────────────────────────────────────
• query_text     : 검색하고 싶은 질문
• k              : top-k 이웃 개수 (쿼리 포함하면 k+1 점)
• show_content   : True면 문서 내용 일부도 출력
결과 파일        : retrieve_visualization.png
"""

import os
# transformers 가 torchvision 을 import 하지 않도록 설정 (MPS/arm 오류 방지)
os.environ["DISABLE_TORCHVISION_IMPORTS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (필수 import)
from sklearn.decomposition import PCA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import matplotlib.font_manager as fm

# 환경 변수 로드
load_dotenv()

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = ['AppleGothic']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────── 1. 파라미터 ──────────────────────────────
query_text = """'사이언스' 게재…"감염원 분리, 분석, 약물 반응 평가 한꺼번에 수행"
동물에서 온 바이러스, 동물 장기로 막는다국내 연구진이 바이러스 감염 특성과 면역 반응을 분석할 수 있는 실험용 플랫폼을 개발했다. 신·변종 바이러스와 미래 팬데믹에 선제 대응이 가능해질 전망이다.

과학기술정보통신부는 기초과학연구원(IBS) 한국바이러스기초연구소와 유전체 교정 연구단이 한국에 서식하는 박쥐에서 유래한 장기 오가노이드를 구축했다고 16일 밝혔다.

연구결과는 국제 학술지 사이언스(5월16일자)에 게재됐다.

오가노이드는 성체 및 배아 줄기세포를 실험실 환경에서 분화한 3차원 장기유사체다. 유사 장기로 불린며, 손상 장기를 치료하거나 동물 실험 모델을 대체하는데 쓰인다.


IBS 박쥐 오가노이드 연구진. 왼쪽부터 최영기 신변종바이러스연구센터장, 구본경 유전체 교정연구단장, 김현준 선임연구원, 허서영 연구원.

이번에 구축한 박쥐는 사스코로나-2(SARS-Cov-2), 메르스코로나(MERS-CoV), 에볼라, 니파 등 고위험 인수공통바이러스 자연 숙주로 알려져 있다.

박쥐 유래 신·변종 바이러스가 고위험 전염병이나 팬데믹을 유발할 잠재적 위협이 되는 이유다.

IBS 연구진은 우리나라를 비롯해 동북아시아 및 유럽에 널리 서식하는 식충성 박쥐인 애기박쥐과(Vespertilionidae) 및 관박쥐과(Rhinolophidae) 박쥐 5종으로부터 기도, 폐, 신장, 소장의 다조직 오가노이드 생체 모델을 구축했다.

연구진은 이 같이 새로 구축한 박쥐 오가노이드를 활용해 코로나(SARS-Cov-2, MERS-CoV), 인플루엔자, 한타 등 박쥐 유래 인수공통바이러스 특이적 감염 양상과 증식 특성을 규명했다. 또 선천적 면역 반응도 정량적으로 확인했다.

연구진은 "이는 쥐 오가노이드가 바이러스-면역 상호작용을 규명할 수 있는 중요한 연구 플랫폼으로 활용될 수 있다는 것"이라며 "야생 박쥐 분변 샘플에서 두 종류의 변종 바이러스를 찾아내고, 이를 배양하고 리하는 데에도 성공했다"고 말했다.

연구진은 기존 3차원 박쥐 오가노이드를 2차원 배양 방식으로 개량, 고속 항바이러스제 스크리닝에 적합한 실험 플랫폼으로 확장했다.

3차원 오가노이드는 모양과 크기가 균일하지 않아 자동화된 실험이 어렵고, 분석과 평가에도 시간이 오래 걸리는 데 반해, 연구진이 개발한 2차원 플랫폼은 오가노이드 유래 세포를 평평한 배양판에 펼쳐 균일한 세포층을 형성해 실험이 용이하고 분석이 빠르다.


IBS가 구축한 다종-다조직 박쥐 오가노이드 플랫폼 설명도.(그림=IBS 및 바이로렌더.com)

연구진은 이 플랫폼을 활용해 분리한 박쥐 유래 변종 바이러스를 대상으로 렘데시비르(Remdesivir) 등 항바이러스제의 효과를 정량적으로 분석한 결과, 기존 세포주 시스템보다 감염 억제 효과를 더 민감하고 정확하게 반영하는 것을 확인했다.

연구를 주도한 김현준 선임연구원은 "박쥐 오가노이드가 신·변종 바이러스의 감염성 평가와 치료제 선별에 모두 활용 가능한 생리학적 모델로 기능할 수 있음을 실증했다"며 "이번 플랫폼을 통해 그동안 세포주 기반 모델로는 어려웠던 바이러스 분리, 감염 분석, 약물 반응 평가를 한 번에 수행할 수 있게 됐다”고 말했다.

구본경 단장은 “실제 박쥐 장기의 생물학적 환경을 실험실에서 구현해 낸 점에 주목할 필요가 있다”며, “특히 바이러스에 대한 박쥐 조직의 감염 반응을 정량적으로 추적할 수 있게 됨으로써, 인수공통감염병의 병리 메커니즘 연구에 중요한 전환점이 될 것”이라고 덧붙였다.

최영기 소장은 “글로벌 감염병 연구자들에게 표준화된 박쥐 모델을 제공하는 바이오뱅크(Biobank) 자원으로서 중요한 의미를 갖는다”며, “박쥐 유래 신·변종 바이러스 감시(surveillance) 및 팬데믹 대비(pandemic preparedness)에 기여할 수 있는 핵심 플랫폼이 될 것"으로 기대했다.


박쥐 오가노이드를 활용한 주요 연구 결과 설명.바이러스 특성연구부터 플랫폼까지 담았다."""  # 검색 질문
k = 2000000  # top-k 이웃 (초기 검색 시 넉넉하게 설정, 추후 필터링)
show_content = True  # 문서 내용 미리보기 출력 여부
VECTORSTORE_PATH = "faiss_index"

def main():
    # ─────────────────────── 2. 기존 벡터스토어 로드 ───────────────────────
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"❌ 벡터스토어 경로가 존재하지 않습니다: {VECTORSTORE_PATH}")
        print("먼저 main.py 서버를 실행하여 벡터스토어를 생성하세요.")
        return

    print("===== 기존 벡터스토어 로드 중 =====")
    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    
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
    print(f"\n===== 쿼리 검색: '{query_text[:50]}...' =====")
    
    # 쿼리 임베딩
    query_embedding = np.array(embeddings.embed_query(query_text)).astype("float32")
    
    # FAISS 인덱스에서 직접 검색
    faiss_index = vectorstore.index
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    
    print(f"✓ Top-{k} 문서 검색 완료 (필터링 전)")
    
    # ─────────────────────── 4. 검색된 문서들의 임베딩 추출 및 필터링 ────────────────────
    initial_topk_embeddings = []
    initial_topk_documents_with_details = [] # 거리, 인덱스 정보 포함
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k}) # 충분한 문서를 가져오도록 retriever의 k도 동일하게 설정
    retrieved_docs = retriever.get_relevant_documents(query_text)
    
    # 검색된 인덱스의 임베딩들 추출
    for i, idx in enumerate(indices[0]):
        emb = faiss_index.reconstruct(int(idx))
        initial_topk_embeddings.append(emb)
        
        if i < len(retrieved_docs):
            doc = retrieved_docs[i]
            initial_topk_documents_with_details.append({
                'content': doc.page_content, # 전체 내용 저장
                'metadata': doc.metadata,
                'distance': distances[0][i],
                'index': idx,
                'embedding': emb # 임베딩도 함께 저장
            })

    # 문서 타입별 필터링 (나무위키 5개, PDF 5개)
    namu_docs_selected = []
    pdf_docs_selected = []

    for doc_info in initial_topk_documents_with_details:
        doc_type = doc_info['metadata'].get('type', 'unknown')
        if doc_type == 'namu_wiki' and len(namu_docs_selected) < 5:
            namu_docs_selected.append(doc_info)
        elif doc_type == 'pdf' and len(pdf_docs_selected) < 5:
            pdf_docs_selected.append(doc_info)
        
        # 두 타입 모두 5개씩 채워졌으면 중단
        if len(namu_docs_selected) == 5 and len(pdf_docs_selected) == 5:
            break
            
    # 최종 선택된 문서들
    final_selected_documents_with_details = namu_docs_selected + pdf_docs_selected
    
    # 최종 문서 리스트와 임베딩 리스트 생성
    topk_documents = []
    topk_embeddings = []
    
    for doc_info in final_selected_documents_with_details:
        topk_documents.append({
            'content': doc_info['content'][:200] + "..." if len(doc_info['content']) > 200 else doc_info['content'],
            'metadata': doc_info['metadata'],
            'distance': doc_info['distance'],
            'index': doc_info['index']
        })
        topk_embeddings.append(doc_info['embedding'])

    topk_embeddings = np.array(topk_embeddings)
    
    # 실제 선택된 문서 수로 k 업데이트 (시각화 및 통계용)
    k_final_selected = len(topk_documents)
    
    if k_final_selected == 0:
        print("\n❌ 조건에 맞는 문서를 찾지 못했습니다. (나무위키 5개, PDF 5개)")
        print(f"   초기 검색된 문서 타입 분포:")
        initial_doc_types = [doc['metadata'].get('type', 'unknown') for doc in initial_topk_documents_with_details]
        initial_type_counts = {}
        for doc_type in initial_doc_types:
            initial_type_counts[doc_type] = initial_type_counts.get(doc_type, 0) + 1
        print(f"     {initial_type_counts}")
        return

    print(f"✓ 필터링 후 최종 선택된 문서 수: {k_final_selected}")
    
    # 쿼리 + 문서 임베딩 결합
    all_embeddings = np.vstack([query_embedding, topk_embeddings])  # (k+1, embedding_dim)
    
    # ─────────────────────── 5. 검색 결과 출력 ────────────────────────
    if show_content:
        print(f"\n===== 검색된 Top-{k_final_selected} 문서들 =====")
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
    colors = plt.cm.tab10(np.linspace(0, 1, k_final_selected + 1))
    
    # (a) 점과 텍스트 그리기
    for i, (x, y, z) in enumerate(xyz):
        is_query = (i == 0)
        
        if is_query:
            # 쿼리 점
            ax.scatter(x, y, z, s=150, color="red", 
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
    
    # ax.set_title(f"RAG Retrieve 시각화: '수학자 소파 문제...'\n"
    #             f"Top-{k_final_selected} 문서와 쿼리의 3D 임베딩 공간 (나무위키 최대 5, PDF 최대 5)", fontsize=14, pad=20)
    
    # 범례 설정 (너무 많으면 생략)
    if k_final_selected <= 10:
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    
    plt.tight_layout()
    
    # ───────────────────────────── 8. 저장 및 출력 ────────────────────────────
    output_filename = "retrieve_visualization.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 시각화 결과 저장: {output_filename}")
    
    # 추가 통계 정보
    print(f"\n===== 요약 통계 =====")
    print(f"총 문서 수: {vectorstore.index.ntotal}")
    print(f"검색된 문서 수 (필터링 후): {k_final_selected}")
    print(f"임베딩 차원: {all_embeddings.shape[1]}")
    if distances.size > 0 and len(topk_documents) > 0: # Check if distances and topk_documents are not empty
        # 필터링된 문서들의 거리만 사용
        filtered_distances = [doc['distance'] for doc in topk_documents]
        if filtered_distances: # Ensure filtered_distances is not empty
             print(f"쿼리와 가장 가까운 문서 거리 (필터링 후): {min(filtered_distances):.4f}")
             print(f"쿼리와 가장 먼 문서 거리 (필터링 후): {max(filtered_distances):.4f}")
    else:
        print("거리를 계산할 수 있는 문서가 없습니다.")
    
    # 문서 타입별 분포
    doc_types = [doc['metadata'].get('type', 'unknown') for doc in topk_documents]
    type_counts = {}
    for doc_type in doc_types:
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    print(f"문서 타입 분포: {type_counts}")

if __name__ == "__main__":
    main() 