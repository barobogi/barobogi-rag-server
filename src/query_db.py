"""
질의 처리 파이프라인.

1. 질문을 임베딩으로 변환
2. 벡터 저장소에서 유사 문서 검색
3. 외부 LLM API에 프롬프트 전달
4. 생성된 답변 반환

환경 변수:
- LLM_API_URL: LLM API 엔드포인트 URL (기본: http://host.docker.internal:11434/api/generate)
- LLM_API_KEY: 외부 API 키 (있으면 OpenAI ChatCompletion 규격, 없으면 Ollama 규격)
- LLM_MODEL: 사용할 모델 이름 (기본: qwen2.5:0.5b)
"""

import os
import requests
from typing import List, Optional

from .vector_store import VectorStore
from .custom_embeddings import CustomEmbeddings


class QueryDB:
    """질의-응답 파이프라인을 처리하는 클래스."""

    DEFAULT_TOP_K: int = 3
    TIMEOUT: int = 120

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[CustomEmbeddings] = None,
    ) -> None:
        """
        Parameters
        ----------
        vector_store:
            VectorStore 인스턴스. None이면 파일 기반 DB를 사용합니다.
        embeddings:
            CustomEmbeddings 인스턴스. None이면 기본값으로 생성합니다.
        """
        self.vector_store = vector_store or VectorStore()
        self.embeddings = embeddings or CustomEmbeddings()

        # 환경 변수에서 설정 읽기
        self.llm_api_url = os.environ.get(
            "LLM_API_URL", "http://host.docker.internal:11434/api/generate"
        )
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.llm_model = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")

    def run_query(self, question: str, top_k: int = 3) -> str:
        """
        질문에 대한 답변을 생성합니다.

        Parameters
        ----------
        question:
            사용자 질문.
        top_k:
            검색할 문서 수.

        Returns
        -------
        str
            생성된 답변.
        """
        # 1. 질문 임베딩 생성
        query_embedding = self.embeddings.get_embedding(question)

        # 2. 유사 문서 검색
        similar_docs = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. 프롬프트 구성
        context = "\n\n".join(doc.text for doc in similar_docs)
        prompt = self._build_prompt(context, question)

        # 4. LLM API에 질의 (API 키 유무에 따라 분기)
        answer = self._query_llm(prompt)

        return answer

    def _build_prompt(self, context: str, question: str) -> str:
        """
        LLM에 전달할 프롬프트를 구성합니다.
        """
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

    def _query_llm(self, prompt: str) -> str:
        """
        LLM API를 호출하여 답변을 생성합니다.
        API 키가 있으면 OpenAI ChatCompletion 규격으로,
        없으면 Ollama 기본 규격으로 요청합니다.
        """
        if self.llm_api_key:
            return self._query_openai_compatible(prompt)
        else:
            return self._query_basic(prompt)

    def _query_basic(self, prompt: str) -> str:
        """
        API 키 없이 기본 POST 요청 (Ollama 등).
        페이로드: {"model": ..., "prompt": ..., "stream": false}
        """
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                self.llm_api_url, json=payload, timeout=self.TIMEOUT
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API 호출 실패: {e}") from e

        result = response.json()
        return result.get("response", "")

    def _query_openai_compatible(self, prompt: str) -> str:
        """
        API 키가 있을 때 OpenAI ChatCompletion 규격으로 요청.
        헤더: Authorization: Bearer <API_KEY>
        페이로드: {"model": ..., "messages": [...], "temperature": 0.7}
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }

        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 한국어로 답변하는 유용한 AI 어시스턴트입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                self.llm_api_url, headers=headers, json=payload, timeout=self.TIMEOUT
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI 호환 API 호출 실패: {e}") from e

        result = response.json()
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return result.get("response", "")

    def add_document(self, text: str) -> str:
        """
        문서를 추가합니다. (테스트/데모용)

        Parameters
        ----------
        text:
            추가할 문서 텍스트.

        Returns
        -------
        str
            저장된 문서 ID.
        """
        embedding = self.embeddings.get_embedding(text)
        return self.vector_store.add_document(text=text, embedding=embedding)

    def add_documents(self, texts: List[str]) -> List[str]:
        """
        여러 문서를 추가합니다. (테스트/데모용)

        Parameters
        ----------
        texts:
            추가할 문서 텍스트 목록.

        Returns
        -------
        List[str]
            저장된 문서 ID 목록.
        """
        return [self.add_document(text) for text in texts]