import re
import tiktoken

def clean_text(s: str) -> str:
    """여러 줄 공백 등을 정리합니다."""
    s = s.replace('\x00', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def chunk_text(text: str, min_toks=300, max_toks=700, overlap=0.15):
    """
    토큰 수 기준으로 청킹합니다.
    - max_toks: 한 청크 최대 토큰 수
    - overlap: 청크 간 겹치는 비율
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if not tokens:
        return []

    step = int(max_toks * (1 - overlap))
    if step <= 0:
        step = max_toks

    chunks = []
    for i in range(0, len(tokens), step):
        window = tokens[i:i+max_toks]
        if len(window) < min_toks:
            break
        chunk_text = enc.decode(window)
        chunks.append(chunk_text)

    return chunks

def guardrail_prompt() -> str:
    """LLM 시스템 지침."""
    return (
        "당신은 '강의자료 전용 학습 챗봇'입니다.\n"
        "다음 규칙을 따르세요.\n"
        "1) 제공된 컨텍스트 범위 안에서만 답변합니다.\n"
        "2) 컨텍스트에 있는 문장을 요약·재서술하는 수준으로만 설명하고, 새로운 내용이나 가정을 만들지 않습니다.\n"
        "3) 컨텍스트에 전혀 없는 내용은 추측하지 말고, 정보가 없다고만 말합니다.\n"
        "4) 답변은 자연스러운 한국어 문장만 사용하고, 코드 예시나 장문의 영어 튜토리얼은 포함하지 않습니다.\n"
        "5) [답변], <SYS> 같은 메타 태그는 출력하지 않고, 순수한 설명 문장만 작성합니다.\n"
    )
