"""
질의 처리 파이프라인.

1. 질문을 임베딩으로 변환
2. 벡터 저장소에서 유사 문서 검색
3. Ollama qwen2.5 모델에 프롬프트 전달
4. 생성된 답변 반환
"""

import requests
from typing import List, Optional

from .vector_store import VectorStore
from .custom_embeddings import CustomEmbeddings


class QueryDB:
    """질의-응답 파이프라인을 처리하는 클래스."""

    OLLAMA_GENERATE_URL: str = "http://localhost:11434/api/generate"
    DEFAULT_MODEL: str = "qwen2.5:0.5b"
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
            VectorStore 인스턴스. None이면 메모리 DB를 사용합니다.
        embeddings:
            CustomEmbeddings 인스턴스. None이면 기본값으로 생성합니다.
        """
        self.vector_store = vector_store or VectorStore()
        self.embeddings = embeddings or CustomEmbeddings()

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

        # 4. Ollama qwen2.5에 질의
        answer = self._query_ollama(prompt)

        return answer

    def _build_prompt(self, context: str, question: str) -> str:
        """
        Ollama 모델에 전달할 프롬프트를 구성합니다.
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

    def _query_ollama(self, prompt: str) -> str:
        """
        Ollama generate API를 호출하여 답변을 생성합니다.
        """
        payload = {
            "model": self.DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                self.OLLAMA_GENERATE_URL,
                json=payload,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API 호출 실패: {e}") from e

        result = response.json()
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