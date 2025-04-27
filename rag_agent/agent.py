import os, asyncio
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .retrievers import CrossrefRetriever, ScopusRetriever, WosRetriever

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

async def generate_review(topic: str, max_items: int = 60) -> str:
    metadata = await gather_metadata(topic, k_per=max_items//len(RETRIEVERS))
    print(metadata)

    docs = []
    for m in metadata:
        text = m.get("abstract") or m.get("title")
        if not text:
            continue

        header = f"[{m['citekey']}][{m['doi']}] "
        docs.append(
            Document(
                page_content=header + text,
                metadata={
                    "citekey": m["citekey"],
                    "doi": m["doi"],
                    "year": m["year"],
                    "title": m["title"],
                    "source": m["source"],
                },
            )
        )

    # docs = [m.get("abstract") or m.get("title") for m in metadata if (m.get("abstract") or m.get("title"))]
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=0)
    texts = splitter.split_documents(docs)

    print(texts)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(texts, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":6})

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=131096, max_completion_tokens=8096)

    REVIEW_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            """CONTEXT:
{context}
-----
Write a 600-word literature review on **{question}**.

Rules:
1. Use ONLY info in CONTEXT. Do NOT invent sources.
2. Cite each factual sentence with the CITEKEY and DOI exactly as in metadata, like (Pi2024)[10.1016/j.patrec.2024.123456].
3. End with “References:” listing each CITEKEY – DOI pair.
4. Use citekeys and DOIs as they appear at the start of each context chunk. 
5. Do NOT invent new ones.


Begin:"""
        )
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": REVIEW_PROMPT},
        return_source_documents=False,
    )

    return chain.invoke(topic)
