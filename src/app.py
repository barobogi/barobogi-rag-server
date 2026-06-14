"""
FastAPI 기반 API 서버 (보안 인증 + 채팅 웹훅).
Ollama와 벡터 저장소를 활용해 질의-응답을 처리합니다.

사용 방법:
1. pip install fastapi uvicorn
2. uvicorn src.app:app --reload
3. POST /api/query (Authorization: Bearer 필요)
4. POST /api/kakao (카카오톡 챗봇)
5. POST /api/webhook (디스코드/범용 메신저)
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.query_db import QueryDB

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 보안 인증 설정
# ============================================================
MASTER_TOKEN = os.environ.get("SERVER_ACCESS_TOKEN", "barobogi-master-key")
security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(
            status_code=401, detail="인증 헤더가 없습니다."
        )
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
    description="하이브리드 RAG + 카카오톡/디스코드 웹훅",
    version="1.2.0",
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

# /api/query 용
class QueryRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# /api/kakao 용 (카카오 i 오픈빌더 스킬 요청)
class KakaoUserRequest(BaseModel):
    utterance: str = ""


class KakaoRequest(BaseModel):
    userRequest: KakaoUserRequest = KakaoUserRequest()


# /api/webhook 용
class WebhookRequest(BaseModel):
    text: str = ""
    message: str = ""
    content: str = ""
    question: str = ""


# ============================================================
# 쿼리 엔진 초기화
# ============================================================
query_db = QueryDB()


def run_rag(question: str) -> str:
    """RAG 엔진 공통 호출 함수"""
    try:
        return query_db.run_query(question, top_k=3)
    except Exception as e:
        logger.error(f"RAG 오류: {e}")
        return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# 헬퍼 함수: 입력 텍스트 추출
# ============================================================
def _extract_text(body: dict) -> str:
    """여러 형식의 요청에서 텍스트를 추출합니다."""
    for key in ["question", "text", "message", "content"]:
        val = body.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    # 카카오톡 형식
    user_req = body.get("userRequest") or {}
    utterance = user_req.get("utterance", "")
    if utterance:
        return utterance.strip()

    return ""


# ============================================================
# 엔드포인트
# ============================================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "FastAPI RAG 서버가 실행 중입니다.",
        "version": "1.2.0",
        "endpoints": {
            "query": "POST /api/query - REST API (Authorization: Bearer)",
            "kakao": "POST /api/kakao - 카카오톡 챗봇 스킬",
            "webhook": "POST /api/webhook - 디스코드/범용 웹훅",
            "docs": "GET /docs - Swagger UI",
        },
    }


# ── REST API (Bearer 인증 필수) ──

@app.post("/api/query", response_model=AnswerResponse)
async def query(
    request: QueryRequest,
    token: str = Depends(verify_token),
):
    """POST /api/query (Bearer 인증 필수)"""
    answer = run_rag(request.question)
    return AnswerResponse(answer=answer)


# ── 카카오톡 챗봇 웹훅 ──

@app.post("/api/kakao")
async def kakao_webhook(request: Request):
    """
    POST /api/kakao — 카카오 i 오픈빌더 스킬 서버 연동용.

    카카오톡 → 카카오 i 오픈빌더 → 우리 서버
    인증: 카카오 자체 토큰 검증 (별도 Bearer 불필요)
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="잘못된 JSON 요청입니다.")

    question = _extract_text(body)
    if not question:
        # 빈 질문 처리
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "무엇을 도와드릴까요? 질문을 입력해 주세요."
                        }
                    }
                ]
            },
        }

    logger.info(f"[Kakao] 질문: {question}")
    answer = run_rag(question)
    logger.info(f"[Kakao] 답변: {answer[:100]}...")

    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": answer}}
            ]
        },
    }


# ── 디스코드/범용 웹훅 ──

@app.post("/api/webhook")
async def webhook(request: Request):
    """
    POST /api/webhook — 디스코드, 슬랙, Telegram 등 범용 메신저 연동용.

    지원 형식:
    - {"text": "질문"}
    - {"message": "질문"}
    - {"content": "질문"}
    - {"question": "질문"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="잘못된 JSON 요청입니다.")

    question = _extract_text(body)
    if not question:
        return {"answer": "질문을 입력해 주세요."}

    logger.info(f"[Webhook] 질문: {question}")
    answer = run_rag(question)
    logger.info(f"[Webhook] 답변: {answer[:100]}...")

    return {"answer": answer}


# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)