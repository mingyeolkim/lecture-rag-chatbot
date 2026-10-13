import os
import yaml
import fitz
import pandas as pd
import numpy as np
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from utils import clean_text, chunk_text

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PDF_DIR = CFG["paths"]["pdf_dir"]
CHUNK_FILE = CFG["paths"]["chunk_file"]
CHROMA_DIR = CFG["paths"]["chroma_dir"]


def extract_pdf_pages(pdf_path: str, docname: str):
    """PDF 한 파일에서 페이지별 텍스트 추출"""
    doc = fitz.open(pdf_path)
    records = []
    for p in range(len(doc)):
        page = doc.load_page(p)
        txt = page.get_text("text")
        txt = clean_text(txt)
        if not txt:
            continue
        records.append(
            {
                "doc": docname,
                "page": p + 1,
                "raw_text": txt,
            }
        )
    return records


def build_chunks():
    """PDF 전체 → 청킹 결과 parquet 저장"""
    rows = []
    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)

    for fn in os.listdir(PDF_DIR):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, fn)
        docname = os.path.splitext(fn)[0]
        print(f"[build_chunks] processing {fn}")

        page_records = extract_pdf_pages(path, docname)
        for rec in page_records:
            chunks = chunk_text(
                rec["raw_text"],
                min_toks=CFG["chunking"]["min_tokens"],
                max_toks=CFG["chunking"]["max_tokens"],
                overlap=CFG["chunking"]["overlap"],
            )
            for idx, ch in enumerate(chunks):
                chunk_id = f"{docname}-{rec['page']}-{idx}"
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "doc": rec["doc"],
                        "page": rec["page"],
                        "text": ch,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(CHUNK_FILE, index=False)
    print(f"[build_chunks] {len(df)} chunks saved to {CHUNK_FILE}")


def index_chroma():
    """청킹된 텍스트를 임베딩 후 ChromaDB에 색인"""
    df = pd.read_parquet(CHUNK_FILE)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings())


    try:
        client.delete_collection(name="lectures")
    except Exception:
        pass

    col = client.create_collection(name="lectures")

    emb_model = SentenceTransformer(CFG["embedding"]["model"])
    normalize = CFG["embedding"]["normalize"]

    texts = df["text"].tolist()
    print(f"[index_chroma] embedding {len(texts)} chunks...")
    embs = emb_model.encode(texts, normalize_embeddings=normalize)
    embs = np.asarray(embs, dtype=np.float32)

    col.add(
        ids=df["chunk_id"].tolist(),
        documents=texts,
        embeddings=embs,
        metadatas=df[["doc", "page"]].to_dict(orient="records"),
    )

    print(f"[index_chroma] indexed {len(texts)} chunks into ChromaDB ({CHROMA_DIR})")


if __name__ == "__main__":
    build_chunks()
    index_chroma()
