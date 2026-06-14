"""보안 인증 테스트 스크립트"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
TOKEN = "barobogi-master-key"  # 기본 토큰 (환경 변수 SERVER_ACCESS_TOKEN과 일치)
WRONG_TOKEN = "wrong-token-123"
QUESTION = {"question": "안녕하세요"}

print("=" * 60)
print("🔒 RAG API 서버 보안 인증 테스트")
print("=" * 60)

# 1. 루트 엔드포인트 (인증 불필요)
print("\n1️⃣  루트 엔드포인트 (인증 불필요)")
r = requests.get(f"{BASE_URL}/", timeout=10)
print(f"   Status: {r.status_code}")
print(f"   응답: {r.json().get('message', '')}")

# 2. 인증 없이 /api/query 호출 (토큰 없음)
print("\n2️⃣  인증 없이 /api/query 호출 (토큰 없음 → 실패)")
r = requests.post(f"{BASE_URL}/api/query", json=QUESTION, timeout=10)
print(f"   Status: {r.status_code}")
print(f"   응답: {r.text[:100]}")

# 3. 잘못된 토큰으로 호출
print("\n3️⃣  잘못된 토큰으로 호출 (401 예상)")
r = requests.post(
    f"{BASE_URL}/api/query",
    headers={"Authorization": f"Bearer {WRONG_TOKEN}"},
    json=QUESTION,
    timeout=10,
)
print(f"   Status: {r.status_code}")
print(f"   응답: {r.text[:100]}")

# 4. 올바른 토큰으로 호출
print("\n4️⃣  올바른 토큰으로 호출 (200 예상)")
r = requests.post(
    f"{BASE_URL}/api/query",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=QUESTION,
    timeout=120,
)
print(f"   Status: {r.status_code}")
if r.status_code == 200:
    print(f"   응답: {r.json().get('answer', '')[:150]}")
else:
    print(f"   응답: {r.text[:200]}")

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
print(f"\n📌 Render 배포 시 환경 변수 설정:")
print(f"   SERVER_ACCESS_TOKEN: your-secret-key-here")
print(f"\n📌 curl 테스트:")
print(f'   curl -X POST {BASE_URL}/api/query \\')
print(f'     -H "Content-Type: application/json" \\')
print(f'     -H "Authorization: Bearer {TOKEN}" \\')
print(f'     -d \'{{"question": "Money Tree가 무엇인가요?"}}\'')