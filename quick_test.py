import requests
import traceback

# 테스트 결과를 파일로 저장
output_lines = []

try:
    output_lines.append("=== 테스트 시작 ===")
    
    # 1. 루트 테스트
    r = requests.get("http://127.0.0.1:8000/", timeout=5)
    output_lines.append(f"1. 루트: {r.status_code}")
    output_lines.append(f"   응답: {r.text[:200]}")
    
    # 2. 쿼리 테스트 (타임아웃 120초로 설정)
    output_lines.append("\n2. /api/query 테스트:")
    try:
        r = requests.post("http://127.0.0.1:8000/api/query", 
                           json={"question": "Money Tree가 무엇인가요?"},
                           timeout=120)
        output_lines.append(f"   Status: {r.status_code}")
        output_lines.append(f"   응답: {r.text[:500]}")
    except requests.exceptions.Timeout:
        output_lines.append("   타임아웃: Ollama 응답이 120초 이상 걸렸습니다.")
    except Exception as e:
        output_lines.append(f"   에러: {e}")
    
except Exception as e:
    output_lines.append(f"에러: {e}")
    output_lines.append(traceback.format_exc())

output_lines.append("\n=== 테스트 완료 ===")

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("결과를 test_result.txt에 저장했습니다.")