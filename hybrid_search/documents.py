# documents.py - 모든 예제 스크립트가 공유하는 샘플 문서 코퍼스
# 출처: wikidocs.net "BM25 알고리즘 완전 정복"의 예제 데이터를 그대로 사용한다.

RAW_DOCUMENTS = [
    "강아지가 좋아하는 음식은 사료다",
    "고양이도 음식을 좋아한다",
    "강아지 사료는 영양가가 높다",
    "음식은 생명의 근원이다",
]

METADATA_SOURCES = ["pet_guide", "animal_info", "nutrition", "philosophy"]


def get_langchain_documents():
    """LangChain Document 객체 리스트로 변환한 코퍼스를 반환한다."""
    from langchain.schema import Document

    return [
        Document(page_content=text, metadata={"source": source})
        for text, source in zip(RAW_DOCUMENTS, METADATA_SOURCES)
    ]
