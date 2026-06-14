"""테스트 스크립트: /api/query 엔드포인트 테스트"""

import requests
import json

url = "http://127.0.0.1:8000/api/query"
payload = {"question": "Money Tree가 무엇인가요?"}

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except requests.exceptions.ConnectionError:
    print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
except Exception as e:
    print(f"에러 발생: {e}")