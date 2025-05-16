# 🧠 RAG Summary Server (FastAPI + LangChain)

이 프로젝트는 FastAPI와 LangChain을 기반으로 한 **RAG(Retrieval-Augmented Generation)** 서버입니다.  
과학 기술 관련 나무위키 데이터셋을 기반으로 사용자가 과학/기술 관련 query를 질의할 시 retrieval을 통해 더욱 풍부한 응답을 반환합니다.
검색한 소스 문헌의 출처도 반환하여 신뢰도를 높이며, 환각 증상을 완화합니다.

## 🔧 사용 기술

- **FastAPI**: Python 기반의 고성능 웹 프레임워크
- **LangChain**: RAG 구현 및 문서 검색/요약 처리
- **OpenAI API**: GPT 기반 자연어 요약 생성
- **WebBaseLoader**: 웹 URL에서 정보 로딩 (예: 나무위키)
- **Pydantic**: 데이터 모델링 및 검증
- **dotenv**: 환경 변수 관리

## 🚀 기능

- `/rag` 엔드포인트 (`POST`)
  - 사용자의 `content`를 입력으로 받아,
  - 웹 문서 기반 검색 (Retriever),
  - GPT로 요약 생성
  - 결과를 JSON 형태로 반환

# 서버 실행

uvicorn main:app --host 0.0.0.0 --port 8000

환경 변수 설정(.env)
OPENAI_API_KEY=your_openai_api_key_here

**개발자** : [Taeyun Roh]
