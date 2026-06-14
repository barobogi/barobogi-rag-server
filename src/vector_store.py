"""
하이브리드 벡터 저장소.

SQLite를 사용해 문서 텍스트, Dense 임베딩, 키워드 임베딩을 저장하고,
의미론적 검색 + 키워드 검색을 지원합니다.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    text: str
    score: float
    method: str = ""  # "dense", "keyword", "hybrid"


class VectorStore:
    """SQLite 기반 하이브리드 벡터 저장소."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            self.db_path = str(data_dir / "vectors.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    dense_embedding TEXT,
                    keyword_embedding TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)")
            conn.commit()

    def add_document(
        self,
        text: str,
        dense_embedding: Optional[List[float]] = None,
        keyword_embedding: Optional[List[float]] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """문서 하나를 저장합니다."""
        document_id = document_id or str(uuid.uuid4())
        dense_json = json.dumps(dense_embedding or [], ensure_ascii=False)
        keyword_json = json.dumps(keyword_embedding or [], ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, text, dense_embedding, keyword_embedding)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, text, dense_json, keyword_json),
            )
            conn.commit()
        return document_id

    def add_documents(
        self,
        documents: List[dict],
    ) -> List[str]:
        """
        여러 문서를 저장합니다.
        documents: [{"text": ..., "dense_embedding": [...], "keyword_embedding": [...]}, ...]
        """
        saved_ids = []
        for doc in documents:
            saved_ids.append(
                self.add_document(
                    text=doc["text"],
                    dense_embedding=doc.get("dense_embedding"),
                    keyword_embedding=doc.get("keyword_embedding"),
                )
            )
        return saved_ids

    # ── Dense 검색 (의미론적 유사도) ──

    def search_dense(self, query_dense: List[float], top_k: int = 10) -> List[RetrievedDocument]:
        """Dense 임베딩 기반 유사 문서 검색."""
        if top_k <= 0 or not query_dense:
            return []

        with self._connect() as conn:
            rows = conn.execute("SELECT id, text, dense_embedding FROM documents").fetchall()

        results = []
        for row in rows:
            stored = json.loads(row["dense_embedding"]) if row["dense_embedding"] else []
            if not stored:
                continue
            score = cosine_similarity(query_dense, stored)
            if score is not None:
                results.append(RetrievedDocument(id=row["id"], text=row["text"], score=score, method="dense"))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ── 키워드 검색 (BM25 스타일) ──

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())

    def search_keyword(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """키워드 기반 BM25 검색."""
        if top_k <= 0 or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        with self._connect() as conn:
            rows = conn.execute("SELECT id, text FROM documents").fetchall()

        # IDF 계산
        n_docs = len(rows)
        doc_freq = Counter()
        doc_tokens_list = []
        for row in rows:
            tokens = set(self._tokenize(row["text"]))
            doc_tokens_list.append(tokens)
            for t in tokens:
                doc_freq[t] += 1

        # BM25 점수 계산
        k1, b = 1.5, 0.75
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(n_docs, 1)

        results = []
        for i, row in enumerate(rows):
            doc_tokens = self._tokenize(row["text"])
            doc_len = len(doc_tokens)
            tf_counter = Counter(doc_tokens)

            score = 0.0
            for qt in query_tokens:
                if qt not in doc_freq:
                    continue
                idf = math.log((n_docs - doc_freq[qt] + 0.5) / (doc_freq[qt] + 0.5) + 1)
                tf = tf_counter.get(qt, 0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1)))
                score += idf * tf_norm

            if score > 0:
                results.append(RetrievedDocument(id=row["id"], text=row["text"], score=score, method="keyword"))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ── 유틸리티 ──

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> Optional[float]:
    """코사인 유사도 계산."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)