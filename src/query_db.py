"""
질의 처리 파이프라인.

1. 질문을 임베딩으로 변환
2. 벡터 저장소에서 유사 문서 검색
3. 외부 LLM API(Ollama, OpenRouter, Groq 등)에 프롬프트 전달
4. 생성된 답변 반환

환경 변수:
- OLLAMA_URL: Ollama 서버 URL (기본: http://host.docker.internal:11434)
- LLM_MODEL: 사용할 모델 이름 (기본: qwen2.5:0.5b)
- LLM_API_KEY: 외부 API 키 (선택사항, OpenRouter/Groq 등 사용 시)
- LLM_PROVIDER: LLM 제공자 (ollama, openrouter, groq, openai 등)
"""

import os
import requests
from typing import List, Optional

from .vector_store import VectorStore
from .custom_embeddings import CustomEmbeddings


class QueryDB:
    """질의-응답 파이프라인을 처리하는 클래스."""

    # 환경 변수에서 읽어오는 기본값들
    DEFAULT_MODEL: str = "qwen2.5:0.5b"
    DEFAULT_TOP_K: int = 3
    TIMEOUT: int = 120

    # LLM 제공자별 API 엔드포인트 매핑
    PROVIDER_ENDPOINTS = {
        "ollama": "{base_url}/api/generate",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
    }

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
        self.llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
        self.llm_model = os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.ollama_base_url = os.environ.get(
            "OLLAMA_URL", "http://host.docker.internal:11434"
        )

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

        # 4. LLM API에 질의
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
        설정된 LLM 제공자에 따라 API를 호출하여 답변을 생성합니다.
        """
        if self.llm_provider == "ollama":
            return self._query_ollama(prompt)
        else:
            return self._query_external_api(prompt)

    def _query_ollama(self, prompt: str) -> str:
        """
        Ollama generate API를 호출하여 답변을 생성합니다.
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API 호출 실패: {e}") from e

        result = response.json()
        return result.get("response", "")

    def _query_external_api(self, prompt: str) -> str:
        """
        외부 LLM API(OpenRouter, Groq, OpenAI 등)를 호출합니다.
        OpenAI 호환 API 형식을 사용합니다.
        """
        url = self.PROVIDER_ENDPOINTS.get(self.llm_provider)
        if not url:
            raise ValueError(
                f"지원하지 않는 LLM 제공자입니다: {self.llm_provider}\n"
                f"지원 제공자: {list(self.PROVIDER_ENDPOINTS.keys())}"
            )

        headers = {
            "Content-Type": "application/json",
        }
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

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
                url, headers=headers, json=payload, timeout=self.TIMEOUT
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"{self.llm_provider} API 호출 실패: {e}"
            ) from e

        result = response.json()
        # OpenAI 호환 형식에서 응답 추출
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