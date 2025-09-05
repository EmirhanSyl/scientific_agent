import os
import asyncio
import json
from typing import List, Dict, Any, Set, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

from utils.citations import CitationFormatter
from .retrievers import CrossrefRetriever, ScopusRetriever, WosRetriever


# Kullanılacak retriever'lar
RETRIEVERS = [CrossrefRetriever()]
try:
    RETRIEVERS.append(ScopusRetriever())
except Exception:
    print("Warning: Scopus retriever not available")
try:
    RETRIEVERS.append(WosRetriever())
except Exception:
    print("Warning: WOS retriever not available")


# ─────────────────────────────────────────────────────────────────────────────
# Regex yerine structured output şemaları
class CitationItem(BaseModel):
    citekey: str = Field(..., description="Örn: Smith2021")
    doi: Optional[str] = Field(None, description="Varsa DOI")

class ReviewExtraction(BaseModel):
    review: str = Field(..., description="Nihai metin (References bölümü temizlenmiş)")
    used_citations: List[CitationItem] = Field(default_factory=list, description="Gerçekte kullanılan atıflar")
# ─────────────────────────────────────────────────────────────────────────────


async def gather_metadata(topic: str, k_per: int = 20) -> List[Dict[str, Any]]:
    all_meta: List[Dict[str, Any]] = []
    for ret in RETRIEVERS:
        try:
            data = ret.fetch_metadata(topic, k=k_per)
            print(data)
            all_meta.extend(data)
        except Exception as e:
            print(f"[WARN] Retriever {ret.__class__.__name__} failed: {e}")

    # DOI öncelikli tekilleştirme (doi yoksa title ile)
    seen: Set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for m in all_meta:
        key = (m.get("doi") or m.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(m)
    return dedup


async def generate_review(
    topic: str,
    citation_format: str = "raw",
    language: str = "English",
    max_items: int = 60
) -> Dict[str, Any]:
    meta_list = await gather_metadata(topic, k_per=max(1, max_items // max(1, len(RETRIEVERS))))
    if not meta_list:
        return {"query": topic, "result": "No metadata retrieved.", "resources": [], "citations": []}

    # Kaynakları LangChain Document'lerine çevir
    docs: List[Document] = []
    for m in meta_list:
        text = m.get("abstract") or m.get("title")
        if not text:
            continue
        header = f"[{m.get('citekey', '')}]"
        if m.get("doi"):
            header += f"[{m['doi']}] "
        else:
            header += " "
        docs.append(Document(page_content=header + text, metadata=m))

    # Böl, vektörleştir, retriever hazırla
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(
        split_docs,
        OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # LLM (Ollama - gpt-oss)
    llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss"), temperature=0.2)

    REVIEW_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "CONTEXT:\n{context}\n-----\n"
            "Write a ~600-word literature review **in {language}** on **{question}**.\n\n"
            "Rules:\n"
            "1) Use ONLY info in CONTEXT. Do NOT invent sources.\n"
            "2) Cite each factual sentence using given metadata (citekey/DOI).\n"
            "3) Use citekeys/DOIs as they appear; do NOT invent new ones.\n\n"
            "Begin:"
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": REVIEW_PROMPT.partial(language=language)},
    )

    out = chain.invoke({"query": topic})
    review_text: str = out["result"]
    retrieved_docs: List[Document] = out["source_documents"]

    # Aday kaynak listesi
    candidates: List[Dict[str, Any]] = []
    for d in retrieved_docs:
        md = d.metadata
        candidates.append({
            "citekey": md.get("citekey"),
            "doi": md.get("doi"),
            "title": md.get("title"),
            "year": md.get("year"),
            "authors": md.get("authors"),
            "source": md.get("source"),
        })

    # Structured output ile (regex yerine) atıf/temiz metin çıkarımı
    extractor_llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss"), temperature=0)
    structured = extractor_llm.with_structured_output(ReviewExtraction)
    extraction_prompt = (
        "You will receive a REVIEW and a set of CANDIDATE references.\n"
        "Return the review text without any trailing 'References' section, "
        "and the list of actually used citations. Match by citekey; use DOI to disambiguate. "
        "Do not hallucinate.\n\n"
        f"REVIEW:\n{review_text}\n\n"
        f"CANDIDATES(JSON):\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    extracted: ReviewExtraction = structured.invoke(extraction_prompt)

    cleaned_text = extracted.review.strip()
    used_keys: Set[str] = {c.citekey for c in extracted.used_citations if c.citekey}

    # Kullanılan kaynakları derle (eşsiz tut)
    resources: List[Dict[str, Any]] = []
    seen_pairs: Set[tuple] = set()
    for d in retrieved_docs:
        md = d.metadata
        ck = md.get("citekey")
        if used_keys and ck not in used_keys:
            continue
        doi = md.get("doi")
        key = (ck, doi)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        resources.append({
            "citekey": ck,
            "doi": doi,
            "title": md.get("title"),
            "year": md.get("year"),
            "authors": md.get("authors"),
            "source": md.get("source"),
        })

    # Hiç eşleşme çıkmazsa makul geri dönüş
    if not resources:
        for d in retrieved_docs[:10]:
            md = d.metadata
            resources.append({
                "citekey": md.get("citekey"),
                "doi": md.get("doi"),
                "title": md.get("title"),
                "year": md.get("year"),
                "authors": md.get("authors"),
                "source": md.get("source"),
            })

    formatter = CitationFormatter()
    formatted = formatter.format(resources, style=citation_format)

    return {
        "query": topic,
        "result": cleaned_text,
        "resources": resources,
        "citations": formatted
    }
