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
import time
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query_db import QueryDB

# 로깅
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 보안 인증 설정
# ============================================================
MASTER_TOKEN = os.environ.get("SERVER_ACCESS_TOKEN", "barobogi-master-key")
security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="인증 헤더가 없습니다.")
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
    version="1.3.0",
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
# 헬퍼 함수
# ============================================================
KAKAO_ERROR_RESPONSE = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "죄송합니다. AI 엔진 통신 지연이 발생했습니다. 잠시 후 다시 시도해 주세요."
                }
            }
        ]
    },
}


def _extract_text(body: dict) -> str:
    """여러 형식의 요청에서 텍스트를 추출합니다."""
    for key in ["question", "text", "message", "content"]:
        val = body.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
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
        "version": "1.3.0",
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
    try:
        answer = query_db.run_query(request.question, top_k=3)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"/api/query 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 카카오톡 챗봇 웹훅 ──

@app.post("/api/kakao")
async def kakao_webhook(request: Request):
    """
    POST /api/kakao — 카카오 i 오픈빌더 스킬 서버 연동용.

    카카오톡 → 카카오 i 오픈빌더 → 우리 서버
    인증: 카카오 자체 토큰 검증 (별도 Bearer 불필요)
    """
    # ★ 거대한 try-except: 어떤 에러가 나도 500 안 냄
    try:
        start_time = time.time()

        # 1. 요청 파싱
        try:
            body = await request.json()
        except Exception:
            body = {}

        question = _extract_text(body)
        user_utterance = body.get("userRequest", {}).get("utterance", "")
        logger.info(f"[Kakao Request] utterance={user_utterance}")

        if not question:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": "무엇을 도와드릴까요? 질문을 입력해 주세요."}}
                    ]
                },
            }

        # 2. RAG 질의 (카카오 전용: max_tokens=120, timeout=5초)
        answer = query_db.run_kakao_query(question, top_k=3)

        # 3. 응답 구성
        elapsed = time.time() - start_time
        response = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer}}]
            },
        }
        logger.info(f"[Kakao Response] elapsed={elapsed:.2f}s | answer={answer[:80]}...")
        return response

    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"[Kakao Error] elapsed={elapsed:.2f}s | error={e}")
        # ★ 절대 500을 반환하지 않음. 항상 카카오 규격 에러 메시지 리턴
        return KAKAO_ERROR_RESPONSE


# ── 디스코드/범용 웹훅 ──

@app.post("/api/webhook")
async def webhook(request: Request):
    """
    POST /api/webhook — 디스코드, 슬랙, Telegram 등 범용 메신저 연동용.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="잘못된 JSON 요청입니다.")

    question = _extract_text(body)
    if not question:
        return {"answer": "질문을 입력해 주세요."}

    logger.info(f"[Webhook] 질문: {question}")
    try:
        answer = query_db.run_query(question, top_k=3)
    except Exception as e:
        logger.error(f"[Webhook] 오류: {e}")
        return {"answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다."}

    logger.info(f"[Webhook] 답변: {answer[:100]}...")
    return {"answer": answer}


# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)