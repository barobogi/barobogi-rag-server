"""
FastAPI 기반 API 서버 (보안 인증 적용).
Ollama와 벡터 저장소를 활용해 질의-응답을 처리합니다.

사용 방법:
1. pip install fastapi uvicorn
2. uvicorn src.app:app --reload
3. POST http://127.0.0.1:8000/api/query
   Header: Authorization: Bearer <토큰>
"""

import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query_db import QueryDB


# ============================================================
# 보안 인증 설정
# ============================================================
# 환경 변수 SERVER_ACCESS_TOKEN에서 마스터 키를 읽음 (없으면 기본값)
MASTER_TOKEN = os.environ.get("SERVER_ACCESS_TOKEN", "barobogi-master-key")

# HTTPBearer 보안 스키마 (Authorization: Bearer <토큰>)
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Bearer 토큰 검증 Dependency.
    - 토큰이 없거나, 환경 변수의 값과 다르면 401 반환
    """
    if credentials.credentials != MASTER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="인증 실패: 유효하지 않은 액세스 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ============================================================
# FastAPI 앱 생성
# ============================================================
app = FastAPI(
    title="RAG API Server",
    description="Ollama + 벡터 저장소 기반 질의-응답 API (Bearer 인증 필요)",
    version="1.1.0",
)


# ============================================================
# CORS 설정
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic 모델
# ============================================================
class QueryRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ============================================================
# 쿼리 엔진 초기화
# ============================================================
query_db = QueryDB()


# ============================================================
# 엔드포인트 (루트는 인증 불필요)
# ============================================================
@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 반환"""
    return {
        "message": "FastAPI RAG 서버가 실행 중입니다.",
        "version": "1.1.0 (secured)",
        "endpoints": {
            "query": "POST /api/query - 질문하기 (Authorization: Bearer 필요)",
            "docs": "GET /docs - Swagger UI 문서",
            "redoc": "GET /redoc - ReDoc 문서",
        },
    }


@app.post("/api/query", response_model=AnswerResponse)
async def query(
    request: QueryRequest,
    token: str = Depends(verify_token),  # ← 인증 문지기
):
    """
    POST /api/query (인증 필수)
    
    요청 헤더:
        Authorization: Bearer <SERVER_ACCESS_TOKEN>
    
    요청 바디:
        {"question": "질문 내용"}
    
    응답:
        {"answer": "AI의 답변 내용..."}
    """
    try:
        answer = query_db.run_query(request.question, top_k=3)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)