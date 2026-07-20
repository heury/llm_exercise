# bm25_search.py - rank-bm25 라이브러리와 LangChain BM25Retriever 사용법
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "Python으로 BM25 구현하기"
from rank_bm25 import BM25Okapi

from documents import RAW_DOCUMENTS, get_langchain_documents


def demo_raw_bm25(query: str = "강아지 음식") -> None:
    """rank-bm25 라이브러리를 직접 사용하는 가장 기본적인 방법."""
    tokenized_docs = [doc.split() for doc in RAW_DOCUMENTS]
    print("토크나이징 결과:", tokenized_docs)

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    print("BM25 점수들:", scores)

    best_docs = bm25.get_top_n(tokenized_query, RAW_DOCUMENTS, n=2)
    print("상위 2개 문서:", best_docs)


def demo_langchain_bm25(query: str = "강아지 음식"):
    """LangChain의 BM25Retriever로 동일한 검색을 수행한다."""
    from langchain_community.retrievers import BM25Retriever

    docs = get_langchain_documents()

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    results = bm25_retriever.invoke(query)
    for result in results:
        print(f"문서: {result.page_content}")
        print(f"소스: {result.metadata['source']}")
        print("---")
    return results


if __name__ == "__main__":
    print("=== rank-bm25 직접 사용 ===")
    demo_raw_bm25()

    print("\n=== LangChain BM25Retriever ===")
    demo_langchain_bm25()
