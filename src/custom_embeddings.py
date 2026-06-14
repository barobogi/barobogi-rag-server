"""
하이브리드 임베딩 생성 모듈 (Pure Python + NumPy).

1. TF-IDF 기반 키워드 임베딩
2. Dense 임베딩: 외부 임베딩 API(OpenAI 호환) 또는 NumPy 기반 경량 해시 임베딩

의존성: numpy (선택), 표준 라이브러리만으로도 동작
"""

import re
import math
import os
import hashlib
from collections import Counter
from typing import List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CustomEmbeddings:
    """하이브리드 임베딩 생성 클래스 (키워드 + Dense)."""

    DENSE_DIM: int = 256  # Dense 임베딩 차원 (경량 해시 기반)

    def __init__(self):
        self._vocab = {}
        self._vocab_idx = 0
        self._dense_api_url = os.environ.get("EMBEDDING_API_URL", "")
        self._dense_api_key = os.environ.get("EMBEDDING_API_KEY", "")
        self._dense_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── 토큰화 ──

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())

    # ── TF-IDF 키워드 임베딩 ──

    def _build_vocab(self, texts: List[str]):
        for text in texts:
            for token in self._tokenize(text):
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_idx
                    self._vocab_idx += 1

    def _compute_tf(self, text: str) -> dict:
        tokens = self._tokenize(text)
        token_count = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {token: count / total for token, count in token_count.items()}

    def _compute_idf(self, documents: List[str]) -> dict:
        n_docs = len(documents)
        doc_freq = {}
        for doc in documents:
            for token in set(self._tokenize(doc)):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        return {t: math.log(n_docs / (1 + f)) + 1 for t, f in doc_freq.items()}

    def get_tfidf_embedding(self, text: str, idf: Optional[dict] = None) -> List[float]:
        if not self._vocab:
            self._build_vocab([text])
        if idf is None:
            idf = self._compute_idf([text])
        tf = self._compute_tf(text)
        tfidf = [tf.get(t, 0.0) * idf.get(t, 0.0) for t in sorted(self._vocab.keys())]
        norm = math.sqrt(sum(x * x for x in tfidf))
        if norm > 0:
            tfidf = [x / norm for x in tfidf]
        return tfidf

    # ── Dense 임베딩 ──

    def _hash_embedding(self, text: str) -> List[float]:
        """
        NumPy 기반 경량 해시 임베딩.
        텍스트의 n-gram을 해시하여 고정 차원 벡터를 생성합니다.
        """
        ngram_range = (2, 4)
        dim = self.DENSE_DIM

        if HAS_NUMPY:
            vec = np.zeros(dim, dtype=np.float64)
        else:
            vec = [0.0] * dim

        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h // dim) % 2 == 0 else -1.0
                if HAS_NUMPY:
                    vec[idx] += sign
                else:
                    vec[idx] += sign

        # L2 정규화
        if HAS_NUMPY:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        else:
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            return vec

    def _api_embedding(self, text: str) -> Optional[List[float]]:
        """
        외부 임베딩 API(OpenAI 호환)를 호출합니다.
        EMBEDDING_API_URL 환경 변수가 설정되어 있을 때만 동작합니다.
        """
        if not self._dense_api_url:
            return None

        import requests as _req

        headers = {"Content-Type": "application/json"}
        if self._dense_api_key:
            headers["Authorization"] = f"Bearer {self._dense_api_key}"

        payload = {"model": self._dense_model, "input": text}

        try:
            resp = _req.post(
                self._dense_api_url, headers=headers, json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"[WARN] 외부 임베딩 API 호출 실패: {e}")
            return None

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Dense 임베딩 벡터를 생성합니다.
        1순위: 외부 임베딩 API (EMBEDDING_API_URL 설정 시)
        2순위: NumPy 기반 경량 해시 임베딩 (로컬, 무설치)
        """
        # 외부 API 시도
        api_result = self._api_embedding(text)
        if api_result is not None:
            return api_result

        # 로컬 해시 임베딩
        return self._hash_embedding(text)

    # ── 호환 인터페이스 ──

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        return self.get_dense_embedding(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.get_dense_embedding(t) for t in texts]