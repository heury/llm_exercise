# hybrid_search

BM25(키워드 기반 검색)와 Vector Search(의미 기반 검색)를 비교하고,
LangChain `EnsembleRetriever`로 두 검색 방식을 결합하는 하이브리드 검색 예제.

출처: wikidocs.net "BM25 알고리즘 완전 정복" (책: *LLM부터 Agent까지*)

## 설치

```bash
uv pip install -r requirements.txt
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `documents.py` | 모든 예제가 공유하는 샘플 문서 코퍼스 |
| `formula_walkthrough.py` | BM25 공식(TF·IDF·문서 길이 정규화)을 라이브러리 없이 손으로 계산 |
| `bm25_search.py` | `rank-bm25` 직접 사용 + LangChain `BM25Retriever` |
| `vector_search.py` | FAISS + 다국어 문장 임베딩으로 의미 기반 검색 |
| `compare_search.py` | 동일 쿼리에 대해 BM25 vs Vector Search 결과·점수 비교 |
| `hybrid_search.py` | `EnsembleRetriever`로 Vector 70% + BM25 30% 결합 |
| `utils.py` | 한국어 형태소 토크나이저(`konlpy`)와 전처리 유틸리티 |

## 실행

```bash
python formula_walkthrough.py   # 공식 손계산
python bm25_search.py           # BM25 기본 사용법
python vector_search.py         # 의미 기반 검색 (최초 실행 시 임베딩 모델 다운로드)
python compare_search.py        # BM25 vs Vector Search 비교
python hybrid_search.py         # 하이브리드(앙상블) 검색
```

## 핵심 요약

- **BM25**: 정확한 키워드 매칭에 강하고 빠르지만, 동의어·의미 유사어를 놓친다.
- **Vector Search**: 의미가 비슷하면 표면적 단어가 달라도 찾아내지만, 고유명사·전문 용어처럼
  정확한 일치가 중요한 경우 오히려 약할 수 있다.
- **하이브리드 검색**: 두 방식의 순위를 가중 결합해 정밀도(BM25)와 재현율(Vector)을 모두 확보한다.
