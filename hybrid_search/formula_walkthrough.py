# formula_walkthrough.py - BM25 공식을 라이브러리 없이 손으로 계산해 보는 예제
# 출처: wikidocs.net "BM25 알고리즘 완전 정복" > "BM25 계산 예시"
#
#   BM25(D, Q) = sum_i IDF(qi) * (tf(qi,D) * (k1+1))
#                        / (tf(qi,D) + k1 * (1 - b + b * |D| / avgdl))
import math

# 문서별 형태소 분석 기준 길이 (원문 예제 값을 그대로 사용)
DOC_LENGTHS = {1: 7, 2: 5, 3: 5, 4: 4}
AVGDL = sum(DOC_LENGTHS.values()) / len(DOC_LENGTHS)  # 5.25

# 쿼리 "강아지 음식"의 문서별 TF (원문 예제 값을 그대로 사용)
TF = {
    "강아지": {1: 1, 2: 0, 3: 1, 4: 0},
    "음식": {1: 1, 2: 1, 3: 0, 4: 1},
}
N = len(DOC_LENGTHS)  # 전체 문서 수
DF = {"강아지": 2, "음식": 3}  # 해당 단어가 포함된 문서 수

K1 = 1.2
B = 0.75


def idf(term: str) -> float:
    df = DF[term]
    return math.log((N - df + 0.5) / (df + 0.5))


def term_score(term: str, doc_id: int) -> float:
    tf = TF[term][doc_id]
    doc_len = DOC_LENGTHS[doc_id]
    numerator = tf * (K1 + 1)
    denominator = tf + K1 * (1 - B + B * doc_len / AVGDL)
    return idf(term) * numerator / denominator if denominator else 0.0


def bm25_score(doc_id: int, query_terms: list[str]) -> float:
    return sum(term_score(term, doc_id) for term in query_terms)


if __name__ == "__main__":
    query_terms = ["강아지", "음식"]

    print(f"전체 문서 수 N = {N}, 평균 문서 길이 avgdl = {AVGDL}")
    for term in query_terms:
        print(f"IDF({term}) = log(({N}-{DF[term]}+0.5)/({DF[term]}+0.5)) = {idf(term):.4f}")

    print("\n문서별 BM25 점수 (쿼리: '강아지 음식')")
    scores = {}
    for doc_id in DOC_LENGTHS:
        breakdown = ", ".join(f"{t}={term_score(t, doc_id):.4f}" for t in query_terms)
        scores[doc_id] = bm25_score(doc_id, query_terms)
        print(f"  문서{doc_id} (|D|={DOC_LENGTHS[doc_id]}): {breakdown} -> 합계={scores[doc_id]:.4f}")

    best = max(scores, key=scores.get)
    print(f"\n가장 관련성 높은 문서: 문서{best} (점수={scores[best]:.4f})")
    print(
        "\n참고: IDF('강아지')가 0인 이유는 df=2가 전체 문서 절반(N/2)과 같아 "
        "log(1)=0이 되기 때문이다. 실제 서비스에서는 흔한 단어의 IDF가 "
        "음수가 되는 것을 막기 위해 max(idf, 0) 등으로 클리핑하기도 한다."
    )
