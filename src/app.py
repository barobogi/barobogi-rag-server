"""
FastAPI 기반 API 서버.
Ollama와 벡터 저장소를 활용해 질의-응답을 처리합니다.

사용 방법:
1. pip install fastapi uvicorn
2. uvicorn src.app:app --reload
3. POST http://127.0.0.1:8000/api/query
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query_db import QueryDB


# ============================================================
# FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="RAG API Server",
    description="Ollama + 벡터 저장소 기반 질의-응답 API",
    version="1.0.0",
)


# ============================================================
# CORS 설정 - 외부 웹 UI, 카카오톡, 프론트엔드 등에서 자유롭게 호출 가능
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 모든 출처 허용
    allow_credentials=True,    # 쿠키 등 인증 정보 허용
    allow_methods=["*"],       # 모든 HTTP 메서드 허용
    allow_headers=["*"],       # 모든 헤더 허용
)


# ============================================================
# Pydantic 모델 정의
# ============================================================
class QueryRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ============================================================
# 쿼리 엔진 초기화 (쿼리DB 시작 시 벡터 저장소 + 임베딩 래퍼 로드)
# ============================================================
query_db = QueryDB()


# ============================================================
# 엔드포인트 정의
# ============================================================
@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "message": "FastAPI RAG 서버가 실행 중입니다.",
        "endpoints": {
            "query": "POST /api/query - 질문하기",
            "docs": "GET /docs - Swagger UI 문서",
            "redoc": "GET /redoc - ReDoc 문서",
        },
    }


@app.post("/api/query", response_model=AnswerResponse)
async def query(request: QueryRequest):
    """
    POST /api/query
    
    요청 예시:
        {"question": "Money Tree가 무엇인가요?"}
    
    응답 예시:
        {"answer": "AI의 답변 내용..."}
    
    내부 동작:
        1. 질문을 임베딩으로 변환
        2. 벡터 저장소에서 상위 3개 유사 문서 검색
        3. Ollama qwen2.5 모델에 프롬프트 전달
        4. 생성된 답변 반환
    """
    try:
        answer = query_db.run_query(request.question, top_k=3)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 직접 실행 시 uvicorn으로 서버 기동
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)