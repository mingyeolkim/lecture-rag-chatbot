import re
import time
import pandas as pd

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def em_metric(pred: str, golds):
    return int(any(normalize(pred) == normalize(g) for g in golds))


def f1_metric(pred: str, golds):
    pred_tokens = set(normalize(pred).split())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for g in golds:
        gold_tokens = set(normalize(g).split())
        if not gold_tokens:
            continue
        inter = pred_tokens & gold_tokens
        if not inter:
            continue
        precision = len(inter) / len(pred_tokens)
        recall = len(inter) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def source_hit_metric(answer_raw: str, gold_pages):
    """
    원본 답변에서 p.xx 패턴을 찾아 gold_pages와 일치 여부 확인
    gold_pages: "3,4" 이런 식으로 저장했다고 가정
    """
    found = re.findall(r"p\.(\d+)", answer_raw)
    if not found:
        return 0
    gold = set(str(p).strip() for p in str(gold_pages).split(",") if p)
    return int(any(p in gold for p in found))


def hallucination_metric(em, f1, sh):
    """
    매우 단순한 환각률 정의:
    - EM=0, F1<0.3, 출처도 없음(SH=0) 이면 '환각'이라고 간주 (1)
    - 그 외는 0
    """
    if em == 0 and f1 < 0.3 and sh == 0:
        return 1
    return 0


def run_eval(show_details: bool = True):
    """
    data/qa.csv 형식 예시
    question,answers,gold_pages
    "역전파란 무엇인가?","가중치의 기울기를 계산해 파라미터를 업데이트하는 알고리즘이다.|오차를 줄이기 위한 기울기 계산 과정이다.","12"
    """
    # Defer loading the embedding, reranker and local LLM until evaluation runs.
    from rag import answer

    df = pd.read_csv("data/qa.csv")
    ems, f1s, shs, hls, lats = [], [], [], [], []

    for _, row in df.iterrows():
        q = row["question"]
        golds = [g.strip() for g in str(row["answers"]).split("|")]
        gold_pages = row.get("gold_pages", "")

        t0 = time.time()
        ans_clean, ans_raw, _ = answer(q, return_raw=True)
        lat = time.time() - t0

        em = em_metric(ans_clean, golds)
        f1 = f1_metric(ans_clean, golds)
        sh = source_hit_metric(ans_raw, gold_pages)
        hl = hallucination_metric(em, f1, sh)

        ems.append(em)
        f1s.append(f1)
        shs.append(sh)
        hls.append(hl)
        lats.append(lat)

        if show_details:
            print(f"[Q] {q}")
            print(f"[A] {ans_clean[:120]}...")
            print(f"EM={em} F1={f1:.3f} SH={sh} HALLU={hl} Lat={lat:.2f}s")
            print("-" * 60)

    summary = {
        "EM": sum(ems) / len(ems),
        "F1": sum(f1s) / len(f1s),
        "SourceHit": sum(shs) / len(shs),
        "HallucinationRate": sum(hls) / len(hls),
        "Latency": sum(lats) / len(lats),
    }

    if show_details:
        print("\n=== Summary ===")
        print(f"EM               : {summary['EM']:.3f}")
        print(f"F1               : {summary['F1']:.3f}")
        print(f"Source Hit       : {summary['SourceHit']:.3f}")
        print(f"HallucinationRate: {summary['HallucinationRate']:.3f}")
        print(f"Latency(s)       : {summary['Latency']:.2f}")

    return summary


def main():
    run_eval(show_details=True)


if __name__ == "__main__":
    main()
