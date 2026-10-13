
import warnings
from transformers.utils import logging as hf_logging

from rag import answer
from eval import run_eval


warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()


def main():
    print("======================================")
    print("  Lecture RAG CLI Chatbot")
    print("  질문을 입력해 주세요. (종료: exit / quit, 평가: metric)")
    print("======================================")

    while True:
        try:
            q = input("\nQ: ")
        except EOFError:
            print("\n[종료]")
            break

        cmd = q.strip().lower()

        if cmd in ("exit", "quit"):
            print("\n[종료합니다]")
            break

        if cmd == "metric":
            print("\n[평가지표 계산 중...]\n")
            summary = run_eval(show_details=True)
            print("\n[요약 결과]")
            print(summary)
            continue

        if not q.strip():
            continue


        try:
            ans, _ = answer(q)
        except Exception as e:
            print(f"\n[에러 발생] {e}")
            continue

        print("\n[A]")
        print(ans)


if __name__ == "__main__":
    main()
