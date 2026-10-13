import yaml
import numpy as np
import pandas as pd
from typing import List, Dict

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import chromadb
from chromadb.config import Settings

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import guardrail_prompt
import re

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

CHUNK_FILE = CFG["paths"]["chunk_file"]
CHROMA_DIR = CFG["paths"]["chroma_dir"]


print("[rag] loading embedding model...")

EMB = SentenceTransformer(CFG["embedding"]["model"], device="cpu")

print("[rag] loading chroma collection...")
CLIENT = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())
COL = CLIENT.get_collection("lectures")

print("[rag] loading chunks for BM25...")
DF_CHUNKS = pd.read_parquet(CHUNK_FILE).reset_index(drop=True)
CORPUS_TEXTS = DF_CHUNKS["text"].tolist()
TOKENIZED_CORPUS = [t.split() for t in CORPUS_TEXTS]
BM25 = BM25Okapi(TOKENIZED_CORPUS)


RERANKER_MODEL_ID = CFG["reranker"]["model"]
print(f"[rag] loading reranker: {RERANKER_MODEL_ID} ...")

RERANKER = CrossEncoder(RERANKER_MODEL_ID, device="cpu")


print("[rag] loading local LLaMA model...")

LLAMA_MODEL_ID = CFG["generation"]["llama_model"]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
LLAMA_MODEL.to(DEVICE)
LLAMA_MODEL.eval()




def hybrid_search(query: str) -> List[Dict]:
    """dense + BM25 하이브리드 검색"""
    topk_initial = CFG["retrieval"]["topk_initial"]


    qv = EMB.encode([query], normalize_embeddings=CFG["embedding"]["normalize"])
    dense_res = COL.query(
        query_embeddings=qv,
        n_results=topk_initial * 2,
        include=["documents", "metadatas", "distances"],
    )

    d_ids = dense_res["ids"][0]
    d_docs = dense_res["documents"][0]
    d_metas = dense_res["metadatas"][0]
    d_dists = np.array(dense_res["distances"][0])


    d_sims = 1 / (1 + d_dists)
    if len(d_sims) > 0:
        d_sims = (d_sims - d_sims.min()) / (d_sims.max() - d_sims.min() + 1e-8)


    bm_scores = BM25.get_scores(query.split())
    n_bm = topk_initial * 2
    bm_idx = np.argsort(bm_scores)[::-1][:n_bm]
    bm_sims = bm_scores[bm_idx]
    if len(bm_sims) > 0:
        bm_sims = (bm_sims - bm_sims.min()) / (bm_sims.max() - bm_sims.min() + 1e-8)

    alpha = CFG["retrieval"]["hybrid_weight_dense"]
    pool: Dict[str, Dict] = {}


    for cid, txt, meta, s in zip(d_ids, d_docs, d_metas, d_sims):
        pool[cid] = {
            "id": cid,
            "text": txt,
            "meta": meta,
            "score_dense": float(s),
            "score_bm25": 0.0,
        }


    for idx, s in zip(bm_idx, bm_sims):
        row = DF_CHUNKS.iloc[idx]
        cid = row["chunk_id"]
        meta = {"doc": row["doc"], "page": int(row["page"])}
        if cid in pool:
            pool[cid]["score_bm25"] = float(s)
        else:
            pool[cid] = {
                "id": cid,
                "text": row["text"],
                "meta": meta,
                "score_dense": 0.0,
                "score_bm25": float(s),
            }


    for cid, item in pool.items():
        item["score"] = alpha * item["score_dense"] + (1 - alpha) * item["score_bm25"]

    items = sorted(pool.values(), key=lambda x: x["score"], reverse=True)
    return items[:topk_initial]


def rerank(query: str, cands: List[Dict]) -> List[Dict]:
    """
    bge-reranker-large 사용한 실제 reranking
    - query와 각 candidate text를 CrossEncoder에 넣어서 재점수
    """
    if not cands:
        return []

    texts = [c["text"] for c in cands]
    pairs = [[query, t] for t in texts]
    scores = RERANKER.predict(pairs)  # numpy array

    for c, s in zip(cands, scores):
        c["rerank_score"] = float(s)

    items = sorted(
        cands,
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
        reverse=True,
    )
    return items[:CFG["retrieval"]["topk_rerank"]]




def clean_context(text: str) -> str:
    """
    RAG 컨텍스트에서 슬라이드 본문에 들어있는
    [ddpm p.2], [문서명 ...] 참고. 같은 태그를 제거.
    """

    text = re.sub(r"\[[^\[\]]* p\.\d+\]", "", text)


    text = re.sub(r"\[문서명[^\]]*\]\s*참고\.?", "", text)


    text = re.sub(r"\[문서명[^\]]*\]", "", text)

    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_context(snippets: List[Dict]) -> str:
    """
    컨텍스트 텍스트 구성
    - 동일 chunk_id / 동일 텍스트 중복 제거
    - 각 스니펫 끝에 [doc p.xx] 형태의 출처 태그 1개씩 붙임
    """
    budget = CFG["generation"]["max_context_chars"]
    used = []
    total = 0

    seen_ids = set()
    seen_texts = set()

    for s in snippets:
        cid = s["id"]
        if cid in seen_ids:
            continue

        cleaned_block = clean_context(s["text"].strip())
        if not cleaned_block:
            continue


        if cleaned_block in seen_texts:
            continue

        seen_ids.add(cid)
        seen_texts.add(cleaned_block)

        tag = f"[{s['meta']['doc']} p.{s['meta']['page']}]"
        piece = cleaned_block + "\n" + tag

        if total + len(piece) + 4 > budget:
            break

        used.append(piece)
        total += len(piece) + 4

    context = "\n\n---\n\n".join(used)


    if CFG["generation"].get("summarize_context", False):
        max_chars_for_summary = CFG["generation"].get("summary_chars", 4000)
        if len(context) > max_chars_for_summary:
            context = summarize_context(context, max_chars_for_summary)

    return context


def _strip_code_and_english(text: str) -> str:
    """
    답변에서 튀어나온 코드/영어 꼬리 잘라내기:
    - ``` 코드블록
    - 'python', 'import ', 'class ', 'def ' 등
    - 영어만 잔뜩 있는 라인 제거
    """

    cut_markers = [
        "```",
        "python",
        "import ",
        "class ",
        "def ",
        "torch.",
        "nn.",
        "from ",
        "Dataset(",
        "```python",
    ]
    cut_idx = len(text)
    for m in cut_markers:
        idx = text.find(m)
        if idx != -1 and idx < cut_idx:
            cut_idx = idx
    text = text[:cut_idx].strip()


    lines = text.splitlines()
    cleaned_lines = []
    for ln in lines:
        alpha = len(re.findall(r"[A-Za-z]", ln))
        hangul = len(re.findall(r"[가-힣]", ln))
        if alpha > 0 and hangul == 0 and alpha / max(len(ln), 1) > 0.4:

            continue
        cleaned_lines.append(ln)
    text = "\n".join(cleaned_lines)

    return text


def _truncate_sentences(text: str, max_sent: int = 5) -> str:
    """
    문장이 너무 길어질 때 앞쪽 N문장만 남기기.
    한국어/영어를 가리지 않고 . ! ? 로 문장 경계를 본다.
    """

    text = re.sub(r"end of response.*$", "", text, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[\.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sent:
        return text.strip()
    return " ".join(sentences[:max_sent]).strip()


def clean_answer(text: str) -> str:
    """
    최종 LLM 답변 후처리:
    - 컨텍스트용 태그 제거
    - 코드/영어 튜토리얼 꼬리 제거
    - '---' 이후(메타 코멘트 구간) 제거
    - 영어 메타 코멘트 제거
    - 불필요한 특수문자 정리
    - 문장 수를 5문장 이내로 제한
    """

    text = clean_context(text)


    text = _strip_code_and_english(text)


    dash_idx = text.find('---')
    if dash_idx != -1:
        text = text[:dash_idx].strip()


    patterns = [
        r"Please let me know if you want me to explain anything else!?",
        r"Please provide an answer in Korean only\.?",
        r"Note:\s.*$",
        r"If you find any mistakes, please feel free to correct me!?",
        r"However, if you need further clarification.*$",
        r"response:\s*$",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)


    text = re.sub(r"[^\w\s\.\,\-\(\)\[\]\{\}가-힣ㄱ-ㅎㅏ-ㅣ/:%]", "", text)


    text = re.sub(r"\s{2,}", " ", text).strip()


    text = _truncate_sentences(text, max_sent=5)


    if not re.search(r"[가-힣]", text):
        return "문서에 정보가 없습니다."
    return text.strip()




def summarize_context(context: str, max_chars: int = 4000) -> str:
    """
    컨텍스트가 너무 길 경우, LLaMA를 사용해 요약.
    - 요약도 한국어로, 핵심 개념 위주로만.
    """
    prompt = (
        "다음은 강의자료에서 추출한 여러 문단입니다.\n"
        "질문에 답하기 위한 핵심 내용만 한국어로 간결하게 요약해 주세요.\n"
        "코드 예시나 영어 튜토리얼은 포함하지 마세요.\n\n"
        "[원본 컨텍스트]\n"
        f"{context[:max_chars*2]}\n\n"
        "요약:"
    )

    inputs = LLAMA_TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = LLAMA_MODEL.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=LLAMA_TOKENIZER.eos_token_id,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    summary = LLAMA_TOKENIZER.decode(gen_tokens, skip_special_tokens=True)
    summary = clean_answer(summary)
    return summary




def build_prompt(query: str, context: str) -> str:
    system_text = guardrail_prompt()

    prompt = f"""{system_text}

아래의 강의자료 컨텍스트만 참고해서 질문에 한국어로만 답변해 주세요.

- 답변은 3~5문장 정도로, 핵심 개념만 간단명료하게 설명합니다.
- 슬라이드에 적힌 표현을 위주로 재서술하고, 새로운 이론이나 가정을 만들어내지 마세요.
- 파이썬/코드 예시(예: python, import, class, def 등)는 절대 쓰지 마세요.
- 영어 튜토리얼이나 메타 코멘트(예: Please let me know~, Note:)를 쓰지 마세요.
- 컨텍스트에 없는 수식, 학습 과정(최적화, 옵티마이저, KL divergence 등)에 대한 추측성 설명은 만들지 마세요.
- 질문이 따로 요청하지 않는다면, 학습 objective나 최적화 방법은 설명하지 마세요.
- 출처나 문서명, p.xx 등은 답변에 쓰지 마세요.
- 컨텍스트에 정보가 충분하지 않으면 '문서에 정보가 없습니다.'라고만 답하세요.

[컨텍스트]
{context}

질문: {query}

답변(한국어 문장만, 코드/영어 튜토리얼 금지):"""
    return prompt.strip()


def generate_raw_answer(query: str, context: str) -> str:
    """LLaMA를 호출해서 '원본' 답변 텍스트를 반환 (클린 전)."""
    prompt = build_prompt(query, context)

    inputs = LLAMA_TOKENIZER(
        prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = LLAMA_MODEL.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            eos_token_id=LLAMA_TOKENIZER.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=8,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer_text = LLAMA_TOKENIZER.decode(
        gen_tokens,
        skip_special_tokens=True,
    )
    return answer_text.strip()


def answer(query: str, return_raw: bool = False):
    """
    최종 질의응답 엔드포인트
    - 기본: (clean_answer, used_snippets)
    - return_raw=True: (clean_answer, raw_answer, used_snippets)
    """
    cands = hybrid_search(query)
    rer = rerank(query, cands)
    context = build_context(rer)

    raw_ans = generate_raw_answer(query, context)
    clean_ans = clean_answer(raw_ans)

    if return_raw:
        return clean_ans, raw_ans, rer
    else:
        return clean_ans, rer
