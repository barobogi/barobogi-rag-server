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


def _extract_kakao_text(body: dict) -> str:
    """카카오톡 본문에서 utterance를 안전하게 추출합니다."""
    user_request = body.get("userRequest")
    if isinstance(user_request, dict):
        utterance = user_request.get("utterance", "").strip()
        if utterance:
            return utterance
    # 폴백: 다른 키로 시도
    for key in ["question", "text", "message", "content"]:
        val = body.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


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


# ── 카카오톡 챗봇 웹훅 (슬래시 유/무 모두 수용) ──

@app.post("/api/kakao")
@app.post("/api/kakao/")
async def kakao_webhook(request: Request):
    """
    POST /api/kakao — 카카오 i 오픈빌더 스킬 서버 연동용.
    """
    try:
        start_time = time.time()

        # 1. 요청 body 파싱 + RAW 로그 출력
        body = {}
        try:
            body = await request.json()
            print(f"[Kakao Raw Body]: {body}")
        except Exception as e:
            print(f"[Kakao Body Parse Error]: {e}")
            logger.warning(f"[Kakao] JSON 파싱 실패: {e}, 빈 body 사용")

        # 2. 다중 .get() 안전 체인 추출
        user_request = body.get("userRequest")
        if not isinstance(user_request, dict):
            user_request = {}
        utterance = user_request.get("utterance")
        if not isinstance(utterance, str):
            utterance = ""
        utterance = utterance.strip()

        print(f"[Kakao Parsed Question]: {utterance}")
        logger.info(f"[Kakao Request] utterance={utterance}")

        if not utterance:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": "무엇을 도와드릴까요? 질문을 입력해 주세요."}}
                    ]
                },
            }

        # [초고속 우회 모드] xAI API 타임아웃 완전 우회
        # 하이브리드 검색기를 사용하여 DB에서 상위 1개 청크를 가져와 0.1초만에 리턴합니다.
        try:
            # 1. Dense 임베딩 생성
            query_dense = query_db.embeddings.get_dense_embedding(utterance)
            # 2. Dense 검색 (Top-5)
            dense_results = query_db.vector_store.search_dense(query_dense, top_k=5)
            # 3. 키워드 검색 (Top-5)
            keyword_results = query_db.vector_store.search_keyword(utterance, top_k=5)
            
            # 4. RRF 융합 정렬
            from src.query_db import reciprocal_rank_fusion
            fused = reciprocal_rank_fusion([dense_results, keyword_results], k=60)
            
            if fused:
                top_chunk = fused[0].text
                answer = f"🤖 [RAG 검색 결과 바로가기]\n{top_chunk[:400]}"
            else:
                answer = "죄송합니다. 관련 등록 지식을 찾지 못했습니다."
        except Exception as query_err:
            answer = f"검색 중 오류가 발생했습니다: {str(query_err)}"

        elapsed = time.time() - start_time
        response = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": answer}}]
            },
        }
        logger.info(f"[Kakao Response Bypass] elapsed={elapsed:.3f}s | answer={answer[:80]}...")
        return response

    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"서버 내부 에러 발생: {str(e)}"
        print(f"[Kakao Error Log]: {error_msg}")
        logger.error(f"[Kakao Error] elapsed={elapsed:.2f}s | error={e}")
        # 실제 에러 메시지를 카카오톡 말풍선에 출력
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": error_msg}}
                ]
            },
        }


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