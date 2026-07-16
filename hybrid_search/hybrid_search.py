# hybrid_search.py - Vector Search와 BM25를 EnsembleRetriever로 결합한다
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "하이브리드 검색: 최고의 성능"
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from documents import get_langchain_documents
from vector_search import build_vector_store

QUERY = "강아지 음식"


def build_hybrid_retriever(weights: tuple[float, float] = (0.7, 0.3)) -> EnsembleRetriever:
    docs = get_langchain_documents()

    vector_db = build_vector_store()
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # Vector 70% + BM25 30% 가중치로 두 검색기의 순위를 결합(RRF)한다.
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=list(weights))


def main() -> None:
    hybrid_retriever = build_hybrid_retriever()
    results = hybrid_retriever.invoke(QUERY)

    print(f"쿼리: '{QUERY}'")
    print("하이브리드 검색 결과:")
    for result in results:
        print(f"📄 {result.page_content}")


if __name__ == "__main__":
    main()
