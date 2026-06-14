"""
간단한 벡터 저장소 구현.

SQLite를 사용해 문서 텍스트와 임베딩 벡터를 저장하고,
코사인 유사도 기반으로 상위 k개 문서를 검색합니다.
의존성을 최소화하기 위해 NumPy 없이 Python 표준 라이브러리로 구현했습니다.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    text: str
    score: float


class VectorStore:
    """SQLite 기반의 간단한 벡터 저장소."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        db_path:
            SQLite DB 파일 경로. None이면 기본 파일 DB를 사용합니다.
        """
        if db_path is None:
            # 프로젝트 루트에 data 폴더 생성 후 파일 DB 사용
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
                    embedding TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id)")
            conn.commit()

    def add_document(self, text: str, embedding: Iterable[float], document_id: Optional[str] = None) -> str:
        """
        문서 하나를 저장합니다.

        Parameters
        ----------
        text:
            저장할 문서 텍스트.
        embedding:
            문서의 임베딩 벡터.
        document_id:
            문서 ID. 지정하지 않으면 UUID를 생성합니다.

        Returns
        -------
        str
            저장된 문서 ID.
        """
        document_id = document_id or str(uuid.uuid4())
        embedding_list = [float(value) for value in embedding]
        embedding_json = json.dumps(embedding_list, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, text, embedding)
                VALUES (?, ?, ?)
                """,
                (document_id, text, embedding_json),
            )
            conn.commit()

        return document_id

    def add_documents(
        self,
        documents: Iterable[tuple[str, Iterable[float]]],
    ) -> list[str]:
        """
        여러 문서를 저장합니다.

        Parameters
        ----------
        documents:
            (text, embedding) 형태의 이터러블.

        Returns
        -------
        list[str]
            저장된 문서 ID 목록.
        """
        saved_ids: list[str] = []
        for text, embedding in documents:
            saved_ids.append(self.add_document(text=text, embedding=embedding))
        return saved_ids

    def search(self, query_embedding: Iterable[float], top_k: int = 3) -> list[RetrievedDocument]:
        """
        질의 임베딩과 가장 유사한 문서를 반환합니다.

        Parameters
        ----------
        query_embedding:
            질의 임베딩 벡터.
        top_k:
            반환할 문서 수.

        Returns
        -------
        list[RetrievedDocument]
            유사도 점수와 함께 정렬된 문서 목록.
        """
        if top_k <= 0:
            return []

        query_vector = [float(value) for value in query_embedding]
        if not query_vector:
            return []

        with self._connect() as conn:
            rows = conn.execute("SELECT id, text, embedding FROM documents").fetchall()

        results: list[RetrievedDocument] = []
        for row in rows:
            stored_embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_vector, stored_embedding)
            if score is not None:
                results.append(
                    RetrievedDocument(
                        id=row["id"],
                        text=row["text"],
                        score=score,
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """저장된 문서 수를 반환합니다."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])

    def clear(self) -> None:
        """저장된 문서를 모두 삭제합니다."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> Optional[float]:
    """
    두 벡터의 코사인 유사도를 계산합니다.

    영벡터가 입력되면 None을 반환합니다.
    """
    vector_a = list(vec_a)
    vector_b = list(vec_b)

    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return None

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        dot_product += value_a * value_b
        norm_a += value_a * value_a
        norm_b += value_b * value_b

    if norm_a == 0.0 or norm_b == 0.0:
        return None

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


def ensure_db_directory(db_path: str) -> str:
    """
    SQLite DB 파일의 디렉터리가 존재하도록 생성합니다.

    메모리 DB(':memory:')는 그대로 반환합니다.
    """
    if db_path == ":memory:":
        return db_path

    path = Path(db_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    return db_path