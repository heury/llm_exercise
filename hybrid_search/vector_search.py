# vector_search.py - FAISS + 문장 임베딩을 이용한 의미 기반 검색
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "Vector Search (FAISS, ChromaDB)"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from documents import get_langchain_documents

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_vector_store() -> FAISS:
    docs = get_langchain_documents()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)


def demo_vector_search(query: str = "강아지가 좋아하는 음식", k: int = 3):
    """쿼리와 정확히 같은 단어가 문서에 없어도 의미가 비슷하면 검색되는지 확인한다."""
    vector_db = build_vector_store()
    results = vector_db.similarity_search(query, k=k)
    for result in results:
        print(f"📄 {result.page_content}  (source={result.metadata['source']})")
    return results


if __name__ == "__main__":
    print(f"쿼리: '강아지가 좋아하는 음식'")
    demo_vector_search()
