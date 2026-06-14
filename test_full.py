"""완전한 테스트 스크립트"""

import requests
import sys
import traceback

print("=" * 50)
print("1. 루트 엔드포인트 테스트")
print("=" * 50)
try:
    r = requests.get("http://127.0.0.1:8000/", timeout=10)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"에러: {e}")

print("\n" + "=" * 50)
print("2. Ollama 연결 테스트")
print("=" * 50)
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=10)
    models = r.json().get("models", [])
    print(f"Status: {r.status_code}")
    print(f"사용 가능한 모델: {[m['name'] for m in models]}")
except Exception as e:
    print(f"에러: {e}")

print("\n" + "=" * 50)
print("3. Ollama 임베딩 API 테스트")
print("=" * 50)
try:
    r = requests.post("http://localhost:11434/api/embeddings", 
                       json={"model": "qwen2.5", "prompt": "테스트"}, 
                       timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        embedding = r.json().get("embedding", [])
        print(f"임베딩 길이: {len(embedding)}")
    else:
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"에러: {e}")

print("\n" + "=" * 50)
print("4. Ollama generate 테스트")
print("=" * 50)
try:
    r = requests.post("http://localhost:11434/api/generate",
                       json={"model": "qwen2.5", "prompt": "안녕하세요", "stream": False},
                       timeout=60)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(f"Response: {r.json().get('response', '')[:200]}")
    else:
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"에러: {e}")

print("\n" + "=" * 50)
print("5. FastAPI /api/query 테스트")
print("=" * 50)
try:
    r = requests.post("http://127.0.0.1:8000/api/query",
                       json={"question": "Money Tree가 무엇인가요?"},
                       timeout=120)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except requests.exceptions.ConnectionError:
    print("서버에 연결할 수 없습니다.")
except Exception as e:
    print(f"에러: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("테스트 완료!")
print("=" * 50)