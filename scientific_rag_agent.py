"""
literature_agent_crossref.py
-------------------------------------------------------------------------------
Retrieval‑Augmented Generation (RAG) literature‑review agent that queries
Crossref’s REST API.

Major fixes v0.2
────────────────
* Build‑index now removes empty abstracts and splits requests into manageable
  batches so OpenAI Embeddings endpoint never receives an invalid payload.
* JATS → plain‑text cleaner falls back to `html.parser` if `lxml` is absent.
* Additional defensive checks with clear error messages.

Usage
─────
$ export OPENAI_API_KEY=sk‑...   # or use .env + python‑dotenv
$ export CROSSREF_MAILTO=you@example.com
$ python literature_agent_crossref.py

The agent will prompt for a topic and create `literature_review.md` in the
current directory.
"""

from __future__ import annotations

import os
import re
import time
import json
import math
import textwrap
from pathlib import Path
from typing import List, Tuple
import numpy as np

import requests
from bs4 import BeautifulSoup, FeatureNotFound
import faiss  # cpu build
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment ---------------------------------------------------------
load_dotenv()
client = OpenAI()
MAILTO = "emirhan.soylu@databeeg.com"

# ── Constants ----------------------------------------------------------------
CROSSREF_URL = "https://api.crossref.org/works"
HEADERS = {"User-Agent": f"literature-agent/0.1 (+{MAILTO})"}
EMBED_MODEL = "text-embedding-3-small"
MAX_INPUT_TOKENS = 8192  # per item token limit for the embedding model
BATCH_SIZE = 100         # number of texts per embedding call

# ── Helpers ------------------------------------------------------------------

def _clean_html_abstract(jats: str) -> str:
    """Strip JATS/HTML tags → plain text."""
    try:
        soup = BeautifulSoup(jats, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(jats, "html.parser")
    txt = soup.get_text(separator=" ", strip=True)
    # collapse excessive whitespace
    return re.sub(r"\s+", " ", txt)

def _truncate(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Rough truncation by words (≈ tokens) to stay under model limit."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])

# ── Crossref Search ----------------------------------------------------------

def search_papers(query: str, k: int = 20) -> List[dict]:
    params = {
        "query": query,
        "rows": k,
        "select": "DOI,title,abstract,author,issued",
    }
    if MAILTO:
        params["mailto"] = MAILTO
    r = requests.get(CROSSREF_URL, params=params, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()["message"]["items"]

# ── Text Extraction ----------------------------------------------------------

def extract_texts(items: List[dict]) -> List[str]:
    texts: List[str] = []
    for it in items:
        abst = it.get("abstract")
        if abst:
            clean = _clean_html_abstract(abst)
        else:
            title = it.get("title", [""])[0]
            journal = it.get("container-title", [""])[0]
            clean = f"{title}. ({journal})"
        clean = _truncate(clean)
        if clean.strip():
            texts.append(clean)
    return texts

# ── Embedding & FAISS --------------------------------------------------------

def embed_batch(batch: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]


def build_index(texts: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    nonempty = [t for t in texts if t and t.strip()]
    if not nonempty:
        raise RuntimeError("No non‑empty abstracts or titles were retrieved from Crossref.")

    print(f"📐  Embedding {len(nonempty)} texts …")
    vectors: List[List[float]] = []
    for i in range(0, len(nonempty), BATCH_SIZE):
        chunk = nonempty[i : i + BATCH_SIZE]
        vectors.extend(embed_batch(chunk))
        time.sleep(0.5)  # courtesy pause

    vec_array = np.asarray(vectors, dtype="float32")
    dim = vec_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vec_array)
    return index, nonempty

# ── Retrieval + GPT Synthesis ------------------------------------------------

def knn_search(index: faiss.IndexFlatL2, query: str, docs: List[str], k: int = 5) -> str:
    qemb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    D, I = index.search(np.asarray([qemb], dtype="float32"), k)
    return "\n\n".join(docs[i] for i in I[0])


def generate_review(topic: str) -> str:
    items = search_papers(topic)
    texts = extract_texts(items)
    index, corpus = build_index(texts)
    context = knn_search(index, topic, corpus, k=5)

    system = (
        "You are an expert academic writer. Using the CONTEXT, write a concise "
        "literature review (~800 words) with subheadings, highlight knowledge "
        "gaps, and provide BibTeX‑like inline citations (citekey=FirstAuthorYear)."
    )
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nTASK: Write the review."},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.3)
    return resp.choices[0].message.content

# ── CLI ----------------------------------------------------------------------

def main() -> None:
    topic = "Retrieval-Augmented Generation (RAG) mimarilerinin biyomedikal soru-yanıt alanındaki güvenilirlik ve hataya dayanıklılık (robustness) üzerindeki etkisi"
    out = generate_review(topic)
    Path("literature_review.md").write_text(out, encoding="utf-8")
    print("✅ literature_review.md created.")

if __name__ == "__main__":
    main()
