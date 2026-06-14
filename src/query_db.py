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
"""

import os
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

    여러 검색 결과 리스트를 하나로 합쳐 최종 순위를 결정합니다.
    score = Σ 1 / (k + rank_i)  (각 결과물이 속한 순위 리스트의 합)

    Parameters
    ----------
    rankings:
        검색 결과 리스트들의 목록.
    k:
        RRF 상수 (기본: 60, 원래 논문 권장값).

    Returns
    -------
    List[RetrievedDocument]
        RRF 점수로 재정렬된 상위 문서 목록.
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedDocument] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            rrf_score = 1.0 / (k + rank)
            if doc.id in doc_scores:
                doc_scores[doc.id] += rrf_score
            else:
                doc_scores[doc.id] = rrf_score
            doc_map[doc.id] = doc

    # RRF 점수 기준 내림차순 정렬
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

    def run_query(self, question: str, top_k: int = 3) -> str:
        """
        질문에 대한 답변을 생성합니다.

        1) Dense 검색 Top-10
        2) 키워드 검색 Top-10
        3) RRF로 Top-3 재정렬
        4) LLM에 전달 → 답변
        """
        # 1. Dense 검색 (의미론적)
        query_dense = self.embeddings.get_dense_embedding(question)
        dense_results = self.vector_store.search_dense(query_dense, top_k=10)

        # 2. 키워드 검색 (BM25)
        keyword_results = self.vector_store.search_keyword(question, top_k=10)

        # 3. RRF 융합 → Top-3
        fused = reciprocal_rank_fusion([dense_results, keyword_results], k=60)
        top_docs = fused[:top_k]

        # 4. 프롬프트 구성
        context = "\n\n".join(doc.text for doc in top_docs)
        prompt = self._build_prompt(context, question)

        # 5. LLM 호출
        answer = self._query_llm(prompt)
        return answer

    def _build_prompt(self, context: str, question: str) -> str:
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

    def _query_llm(self, prompt: str) -> str:
        if self.llm_api_key:
            return self._query_openai_compatible(prompt)
        else:
            return self._query_basic(prompt)

    def _query_basic(self, prompt: str) -> str:
        payload = {"model": self.llm_model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(self.llm_api_url, json=payload, timeout=self.TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API 호출 실패: {e}") from e
        return resp.json().get("response", "")

    def _query_openai_compatible(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "당신은 한국어로 답변하는 유용한 AI 어시스턴트입니다."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(
                self.llm_api_url, headers=headers, json=payload, timeout=self.TIMEOUT
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