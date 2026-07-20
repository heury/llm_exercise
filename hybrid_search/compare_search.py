# compare_search.py - BM25와 Vector Search의 검색 결과를 나란히 비교한다
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "BM25 vs Vector Search 비교 실험"
#
# 쿼리에 "반려동물", "먹이"라는 단어를 쓰지만, 코퍼스에는 이 단어들이
# 그대로 등장하지 않는다. BM25는 표면적 단어 일치가 없어 점수가 낮게
# 나오고, Vector Search는 의미가 비슷한 문서를 그대로 찾아낸다.
from rank_bm25 import BM25Okapi

from documents import RAW_DOCUMENTS
from vector_search import build_vector_store

QUERY = "반려동물 먹이"


def main() -> None:
    vector_db = build_vector_store()
    vector_results = vector_db.similarity_search(QUERY, k=3)

    tokenized_docs = [doc.split() for doc in RAW_DOCUMENTS]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(QUERY.split())

    print(f"쿼리: '{QUERY}'\n")

    print("=== Vector Search 결과 (의미 기반) ===")
    for result in vector_results:
        print(f"📄 {result.page_content}")

    print("\n=== BM25 Search 결과 (키워드 기반, 점수 포함) ===")
    for doc, score in sorted(zip(RAW_DOCUMENTS, bm25_scores), key=lambda x: -x[1]):
        print(f"📄 {doc}  (score={score:.4f})")
    print(
        "\n-> '반려동물', '먹이'라는 단어가 코퍼스에 그대로 등장하지 않아 "
        "BM25 점수가 모두 0에 가깝다. 표면적 키워드 일치가 없으면 "
        "BM25는 관련 문서를 구분해내지 못한다."
    )


if __name__ == "__main__":
    main()
