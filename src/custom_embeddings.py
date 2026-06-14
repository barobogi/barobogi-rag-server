"""
텍스트 임베딩 생성 모듈.

Ollama 임베딩이 지원되지 않는 환경에서는 TF-IDF 기반
단순 텍스트 유사도를 사용하여 임베딩을 생성합니다.
"""

import re
import math
from collections import Counter
from typing import List, Optional


class CustomEmbeddings:
    """텍스트 임베딩 생성 클래스."""

    def __init__(self):
        """임베딩 클래스 초기화."""
        self._idf_cache = {}
        self._vocab = {}
        self._vocab_idx = 0

    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리합니다."""
        # 한국어, 영어, 숫자를 토큰으로 분리
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
        return tokens

    def _build_vocab(self, texts: List[str]):
        """어휘 사전을 구축합니다."""
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_idx
                    self._vocab_idx += 1

    def _compute_tf(self, text: str) -> dict:
        """TF(Term Frequency)를 계산합니다."""
        tokens = self._tokenize(text)
        token_count = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {token: count / total for token, count in token_count.items()}

    def _compute_idf(self, documents: List[str]) -> dict:
        """IDF(Inverse Document Frequency)를 계산합니다."""
        n_docs = len(documents)
        idf = {}
        
        # 각 토큰이 문서에 나타나는 횟수 계산
        doc_freq = {}
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # IDF 계산
        for token, freq in doc_freq.items():
            idf[token] = math.log(n_docs / (1 + freq)) + 1
        
        return idf

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 일괄 생성합니다.

        Parameters
        ----------
        texts:
            임베딩을 생성할 텍스트 목록.

        Returns
        -------
        List[List[float]]
            생성된 임베딩 벡터 목록.
        """
        if not texts:
            return []

        # 어휘 사전 구축
        self._build_vocab(texts)
        
        # IDF 계산
        idf = self._compute_idf(texts)
        
        # TF-IDF 벡터 생성
        embeddings = []
        for text in texts:
            tf = self._compute_tf(text)
            tfidf = []
            
            # 어휘 사전 순서대로 TF-IDF 값 계산
            for token in sorted(self._vocab.keys()):
                tf_val = tf.get(token, 0.0)
                idf_val = idf.get(token, 0.0)
                tfidf.append(tf_val * idf_val)
            
            # 정규화
            norm = math.sqrt(sum(x * x for x in tfidf))
            if norm > 0:
                tfidf = [x / norm for x in tfidf]
            
            embeddings.append(tfidf)
        
        return embeddings

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        단일 텍스트의 임베딩을 생성합니다.

        Parameters
        ----------
        text:
            임베딩을 생성할 텍스트.
        model:
            사용할 모델 이름 (사용되지 않음, 호환성을 위해 유지).

        Returns
        -------
        List[float]
            생성된 임베딩 벡터.
        """
        # 단일 텍스트의 경우 더미 문서와 함께 처리
        dummy_docs = [text, "임베딩 생성을 위한 더미 텍스트"]
        embeddings = self.get_embeddings_batch(dummy_docs)
        return embeddings[0] if embeddings else []
