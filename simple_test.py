import requests

print("서버 연결 테스트...")
try:
    r = requests.get("http://127.0.0.1:8000/", timeout=5)
    print(f"루트: {r.status_code}")
except Exception as e:
    print(f"루트 에러: {e}")

print("\n쿼리 테스트 (60초 타임아웃)...")
try:
    r = requests.post("http://127.0.0.1:8000/api/query", 
                       json={"question": "안녕하세요"},
                       timeout=60)
    print(f"쿼리 상태: {r.status_code}")
    print(f"쿼리 응답: {r.text[:300]}")
except requests.exceptions.Timeout:
    print("쿼리 타임아웃 (60초)")
except Exception as e:
    print(f"쿼리 에러: {e}")