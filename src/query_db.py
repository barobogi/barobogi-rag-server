"""
하이브리드 RAG 질의 처리 파이프라인.

1. 질문을 Dense + 키워드 임베딩으로 변환
2. 벡터 저장소에서 Dense 검색(Top-10) + 키워드 검색(Top-10) 수행
3. RRF(Reciprocal Rank Fusion)로 최종 Top-3 재정렬
4. 외부 LLM API에 프롬프트 전달 → 답변 생성

환경 변수:
- LLM_API_URL: LLM API 엔드포인트 URL
- LLM_API_KEY: 외부 API 키 (있으면 OpenAI ChatCompletion 규격)
- LLM_MODEL: 사용할 모델 이름 (기본: qwen2.5:0.5b)
- KAKAO_BYPASS_LLM: "true"면 LLM 호출 없이 청크 원문 반환 (카카오톡 속도 테스트용)
"""

import os
import time
import requests
from typing import List, Optional

from .vector_store import VectorStore, RetrievedDocument
from .custom_embeddings import CustomEmbeddings


def reciprocal_rank_fusion(
    rankings: List[List[RetrievedDocument]],
    k: int = 60,
) -> List[RetrievedDocument]:
    """
    RRF(Reciprocal Rank Fusion) 알고리즘.
    score = Σ 1 / (k + rank_i)
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedDocument] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc.id] = doc_scores.get(doc.id, 0.0) + rrf_score
            doc_map[doc.id] = doc

    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        results.append(
            RetrievedDocument(
                id=doc.id,
                text=doc.text,
                score=doc_scores[doc.id],
                method="hybrid",
            )
        )
    return results


class QueryDB:
    """하이브리드 RAG 질의-응답 파이프라인."""

    DEFAULT_TOP_K: int = 3
    TIMEOUT: int = 120

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[CustomEmbeddings] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embeddings = embeddings or CustomEmbeddings()

        # 환경 변수
        self.llm_api_url = os.environ.get(
            "LLM_API_URL", "http://host.docker.internal:11434/api/generate"
        )
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.llm_model = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
        self.kakao_bypass_llm = os.environ.get("KAKAO_BYPASS_LLM", "").lower() == "true"

    def run_query(self, question: str, top_k: int = 3) -> str:
        """일반 REST API용 질의 (기본 설정)."""
        return self._run_internal(question, top_k, max_tokens=1024, timeout=self.TIMEOUT)

    def run_kakao_query(self, question: str, top_k: int = 3) -> str:
        """
        카카오톡 챗봇용 질의.
        - LLM 우회 모드: 청크 원문 반환 (초고속)
        - 정상 모드: max_tokens=100, timeout=5초
        """
        return self._run_internal(
            question,
            top_k,
            max_tokens=100,
            timeout=5,
            system_prompt="카카오톡 답변이므로 이모지를 적절히 섞어 2~3줄로 매우 짧고 친절하게 핵심만 요약해서 답변해 주세요.",
            bypass_llm=self.kakao_bypass_llm,
        )

    def _run_internal(
        self,
        question: str,
        top_k: int = 3,
        max_tokens: int = 1024,
        timeout: Optional[int] = None,
        system_prompt: Optional[str] = None,
        bypass_llm: bool = False,
    ) -> str:
        """내부 질의 실행 (공통 로직)."""
        t0 = time.time()

        # 1. Dense 검색
        query_dense = self.embeddings.get_dense_embedding(question)
        dense_results = self.vector_store.search_dense(query_dense, top_k=10)

        # 2. 키워드 검색
        keyword_results = self.vector_store.search_keyword(question, top_k=10)

        # 3. RRF 융합
        fused = reciprocal_rank_fusion([dense_results, keyword_results], k=60)
        top_docs = fused[:top_k]

        # 4. 컨텍스트 구성
        context = "\n\n".join(doc.text for doc in top_docs)

        # ★ LLM 우회: 검색된 청크 원문을 바로 반환 (초고속)
        if bypass_llm:
            elapsed = time.time() - t0
            print(f"[Bypass LLM] elapsed={elapsed:.3f}s | context={context[:100]}...")
            return context[:500] if context else "죄송합니다. 관련 정보를 찾지 못했습니다."

        # 5. 프롬프트 구성
        if system_prompt:
            prompt = f"""{system_prompt}

다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:"""
        else:
            prompt = self._build_default_prompt(context, question)

        # 6. LLM 호출
        answer = self._query_llm(prompt, max_tokens=max_tokens, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[Query] elapsed={elapsed:.2f}s | question={question[:50]}...")
        return answer

    def _build_default_prompt(self, context: str, question: str) -> str:
        if context:
            return f"""다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:"""
        else:
            return f"""다음 질문에 답변해주세요.

질문: {question}

답변:"""

    # ── LLM 호출 ──

    def _query_llm(
        self,
        prompt: str,
        max_tokens: int = 1024,
        timeout: Optional[int] = None,
    ) -> str:
        if timeout is None:
            timeout = self.TIMEOUT

        if self.llm_api_key:
            return self._query_openai_compatible(prompt, max_tokens, timeout)
        else:
            return self._query_basic(prompt, max_tokens, timeout)

    def _query_basic(
        self, prompt: str, max_tokens: int = 100, timeout: int = 5
    ) -> str:
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self.llm_api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API 호출 실패: {e}") from e
        return resp.json().get("response", "")

    def _query_openai_compatible(
        self, prompt: str, max_tokens: int = 100, timeout: int = 5
    ) -> str:
        system_content = "당신은 한국어로 답변하는 유용한 AI 어시스턴트입니다."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(
                self.llm_api_url, headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI 호환 API 호출 실패: {e}") from e
        result = resp.json()
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return result.get("response", "")

    # ── 문서 추가 ──

    def add_document(self, text: str) -> str:
        dense = self.embeddings.get_dense_embedding(text)
        return self.vector_store.add_document(text=text, dense_embedding=dense)

    def add_documents(self, texts: List[str]) -> List[str]:
        return [self.add_document(text) for text in texts]