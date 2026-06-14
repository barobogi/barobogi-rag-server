"""API 테스트 스크립트"""

import requests

# 1. 루트 엔드포인트 테스트
print("1. 루트 엔드포인트 테스트:")
try:
    r = requests.get("http://127.0.0.1:8000/", timeout=10)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"에러: {e}")

# 2. Ollama 임베딩 테스트
print("\n2. Ollama 임베딩 테스트:")
try:
    r = requests.post("http://localhost:11434/api/embeddings", 
                       json={"model": "qwen2.5", "prompt": "test"}, 
                       timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        embedding = r.json().get("embedding", [])
        print(f"임베딩 길이: {len(embedding)}")
    else:
        print(f"에러: {r.text[:200]}")
except Exception as e:
    print(f"에러: {e}")

# 3. FastAPI 쿼리 테스트
print("\n3. FastAPI /api/query 테스트:")
try:
    r = requests.post("http://127.0.0.1:8000/api/query",
                       json={"question": "Money Tree가 무엇인가요?"},
                       timeout=120)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"에러: {e}")