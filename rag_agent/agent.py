import os, asyncio
import re
from typing import List, Dict, Any, Set

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from utils.citations import CitationFormatter
from .retrievers import CrossrefRetriever, ScopusRetriever, WosRetriever


CITE_INLINE  = re.compile(r"\((?P<ck>[A-Z][a-zA-Z]+[0-9]{4})\)\[")
CITE_BIBTEX  = re.compile(r"@\w+{(?P<ck>[A-Z][a-zA-Z]+[0-9]{4})", re.I)
# instantiate retrievers
RETRIEVERS = [
    CrossrefRetriever(),
]

# opt-in to others if keys present
try:
    RETRIEVERS.append(ScopusRetriever())
except Exception:
    print("Warning: Scopus retriever not available")
    pass
try:
    RETRIEVERS.append(WosRetriever())
except Exception:
    print("Warning: WOS retriever not available")
    pass

async def gather_metadata(topic: str, k_per=20) -> List[Dict[str, Any]]:
    all_meta: List[Dict[str, Any]] = []
    for ret in RETRIEVERS:
        try:
            data = ret.fetch_metadata(topic, k=k_per)
            all_meta.extend(data)
        except Exception as e:
            print(f"[WARN] Retriever {ret.__class__.__name__} failed: {e}")
    # deduplicate by DOI
    seen = set()
    dedup = []
    for m in all_meta:
        doi = m.get("doi") or m.get("title")
        if doi and doi not in seen:
            seen.add(doi)
            dedup.append(m)
    return dedup

async def generate_review(topic: str, citation_format: str = "raw", max_items: int = 60) -> Dict[str, Any]:
    meta_list = await gather_metadata(topic, k_per=max_items // len(RETRIEVERS))

    docs = []
    for m in meta_list:
        text = m.get("abstract") or m.get("title")
        if not text:
            continue
        header = f"[{m['citekey']}][{m['doi']}] "
        docs.append(
            Document(
                page_content=header + text,
                metadata=m,
            )
        )

    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(
        split_docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=131096, max_completion_tokens=8096)
    REVIEW_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "CONTEXT:\n{context}\n-----\n"
            "Write a 600-word literature review on **{question}**.\n\n"
            "Rules:\n"
            "1. Use ONLY info in CONTEXT. Do NOT invent sources.\n"
            "2. Cite each factual sentence with the CITEKEY and DOI exactly as "
            "in metadata, like (Pi2024)[10.1016/j.patrec.2024.123456].\n"
            "3. End with “References:” listing each CITEKEY in BibTeX format.\n"
            "4. Use citekeys/DOIs as they appear; do NOT invent new ones.\n\n"
            "Begin:"
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": REVIEW_PROMPT},
    )

    out = chain.invoke({"query": topic})
    print(out)
    review_text: str = out["result"]
    retrieved_docs: List[Document] = out["source_documents"]

    cited: Set[str] = set(CITE_INLINE.findall(review_text)) | set(
        CITE_BIBTEX.findall(review_text)
    )

    resources: List[Dict[str, Any]] = []
    seen_pairs = set()
    for d in retrieved_docs:
        ck, doi = d.metadata["citekey"], d.metadata["doi"]
        if ck in cited and (ck, doi) not in seen_pairs:
            seen_pairs.add((ck, doi))
            resources.append(
                {
                    "citekey": ck,
                    "doi": doi,
                    "title": d.metadata["title"],
                    "year": d.metadata["year"],
                    "source": d.metadata["source"],
                }
            )

    formatter = CitationFormatter()
    formatted = formatter.format(resources, style=citation_format)
    return {"query": topic, "result": review_text, "resources": resources, "citations": formatted}