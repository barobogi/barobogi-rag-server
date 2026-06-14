# ============================================================
# FastAPI RAG 서버 Docker 이미지
# ============================================================
# 사용법:
#   빌드: docker build -t rag-api-server .
#   실행: docker run -p 8000:8000 rag-api-server
# ============================================================

# 베이스 이미지: 가볍고 안정적인 Python 3.11 슬림
FROM python:3.11-slim

# 메타데이터
LABEL maintainer="RAG API Server"
LABEL description="FastAPI + Ollama RAG 서버"

# 작업 디렉토리 생성
WORKDIR /app

# 시스템 의존성 설치 (SQLite 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사하여 Docker 레이어 캐싱 활용
COPY requirements.txt .

# Python 의존성 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 및 데이터 복사
COPY src/ ./src/
COPY index.html .
COPY data/ ./data/

# 데이터 디렉토리 생성 (없을 경우)
RUN mkdir -p /app/data

# 포트 개방
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# LLM 설정 (환경 변수로 오버라이드 가능)
ENV LLM_PROVIDER=ollama
ENV LLM_MODEL=qwen2.5:0.5b
ENV OLLAMA_URL=http://host.docker.internal:11434

# 서버 실행 명령
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
