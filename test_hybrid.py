"""하이브리드 검색 엔진 테스트"""
import os, sys
os.environ["LLM_API_URL"] = "http://127.0.0.1:11434/api/generate"
os.environ["LLM_MODEL"] = "qwen2.5:0.5b"

sys.path.insert(0, os.path.dirname(__file__))
from src.custom_embeddings import CustomEmbeddings
from src.vector_store import VectorStore
from src.query_db import QueryDB, reciprocal_rank_fusion

results = []

# 1. 키워드 검색
print("=" * 60)
print("1. 키워드 검색 (BM25)")
print("=" * 60)
vs = VectorStore()
kw = vs.search_keyword("은퇴 준비용 돈줄기", top_k=5)
for r in kw:
    print(f"[KW] score={r.score:.4f}: {r.text[:80]}")
    results.append(r)

# 2. Dense 검색
print("\n" + "=" * 60)
print("2. Dense 검색 (해시 임베딩)")
print("=" * 60)
emb = CustomEmbeddings()
q = emb.get_dense_embedding("은퇴 준비용 돈줄기")
ds = vs.search_dense(q, top_k=5)
for r in ds:
    print(f"[DS] score={r.score:.4f}: {r.text[:80]}")
    results.append(r)

# 3. RRF 융합
print("\n" + "=" * 60)
print("3. RRF 융합 결과")
print("=" * 60)
fused = reciprocal_rank_fusion([kw, ds], k=60)
for i, r in enumerate(fused[:3], 1):
    print(f"RRF #{i} (score={r.score:.4f}): {r.text[:80]}")

# 4. 전체 파이프라인
print("\n" + "=" * 60)
print("4. 전체 RAG 파이프라인 (RRF → LLM)")
print("=" * 60)
try:
    qdb = QueryDB()
    ans = qdb.run_query("은퇴 준비용 돈줄기가 뭐야?")
    print(f"최종 답변:\n{ans[:300]}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n테스트 완료!")