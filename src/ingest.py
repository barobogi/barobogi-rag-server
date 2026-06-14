"""
문서 수집 스크립트 (Ingest).

텍스트 파일이나 PDF를 청크 단위로 분할하여
하이브리드 벡터 저장소에 저장합니다.

사용법:
    python -m src.ingest <파일경로>
    python -m src.ingest data/sample.txt
    python -m src.ingest data/sample.pdf
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.custom_embeddings import CustomEmbeddings
from src.vector_store import VectorStore


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    텍스트를 청크 단위로 분할합니다.

    Parameters
    ----------
    text:
        분할할 텍스트.
    chunk_size:
        청크 최대 크기 (문자 수).
    overlap:
        청크 간 겹침 크기.

    Returns
    -------
    list[str]
        분할된 청크 목록.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise RuntimeError("PDF 처리를 위해 pypdf 또는 PyMuPDF를 설치하세요.")


def ingest_file(file_path: str, chunk_size: int = 500, overlap: int = 100):
    """
    파일을 수집하여 벡터 저장소에 저장합니다.

    Parameters
    ----------
    file_path:
        입력 파일 경로.
    chunk_size:
        청크 최대 크기.
    overlap:
        청크 간 겹침.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"[INFO] 파일 로드: {file_path}")

    # 텍스트 추출
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(str(path))
    else:
        text = path.read_text(encoding="utf-8")

    if not text.strip():
        print("[WARN] 파일이 비어 있습니다.")
        return

    # 청크 분할
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[INFO] 총 {len(chunks)}개 청크 생성 (chunk_size={chunk_size}, overlap={overlap})")

    # 임베딩 생성
    embeddings = CustomEmbeddings()
    vector_store = VectorStore()

    print("[INFO] Dense 임베딩 생성 중...")
    dense_embeddings = embeddings.get_embeddings_batch(chunks)

    print("[INFO] 벡터 저장소에 저장 중...")
    docs = []
    for i, (chunk, dense) in enumerate(zip(chunks, dense_embeddings)):
        docs.append({
            "text": chunk,
            "dense_embedding": dense,
        })

    saved_ids = vector_store.add_documents(docs)
    print(f"[INFO] {len(saved_ids)}개 청크 저장 완료!")
    print(f"[INFO] 저장소 총 문서 수: {vector_store.count()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m src.ingest <파일경로>")
        print("예시: python -m src.ingest data/sample.txt")
        sys.exit(1)

    file_path = sys.argv[1]
    ingest_file(file_path)