# utils.py - 한국어 토크나이징과 전처리 유틸리티
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "실무 활용 팁"
import re

STOPWORDS = ["그리고", "하지만", "그런데"]


def korean_tokenizer(text: str) -> list[str]:
    """konlpy Okt 형태소 분석기로 한국어 문장을 형태소 단위로 분리한다.

    konlpy는 JVM(JPype)에 의존하므로 설치되어 있지 않으면 공백 기준
    분리로 대체한다.
    """
    try:
        from konlpy.tag import Okt

        okt = Okt()
        return okt.morphs(text)
    except Exception:
        return text.split()


def preprocess_text(text: str) -> list[str]:
    """소문자 변환, 특수문자 제거, 불용어 제거 후 단어 리스트를 반환한다."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words
