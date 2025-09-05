from __future__ import annotations

from typing import List, Dict, Any
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from rag_agent.retrievers import CrossrefRetriever, ScopusRetriever
from utils.citations import CitationFormatter


# ── Pydantic Models for Structured Output (internal only) ───────────────────

class LiteratureSection(BaseModel):
    heading: str = Field(..., description="Section heading")
    body: str = Field(..., description="Paragraphs with inline (CITEKEY) citations")


class LiteratureDraft(BaseModel):
    """LLM-structured output of the review; will be rendered to markdown string."""
    title: str
    summary: str
    sections: List[LiteratureSection]
    limitations: str
    references: List[str] = Field(
        default_factory=list,
        description="List of citekeys referenced in the text"
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _dedupe_citekeys(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure citekeys are unique across merged sources by adding a, b, c..."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        ck = r.get("citekey") or "AnonNd"
        buckets.setdefault(ck, []).append(r)

    out: List[Dict[str, Any]] = []
    for ck, group in buckets.items():
        if len(group) == 1:
            out.extend(group)
        else:
            for idx, g in enumerate(group):
                g2 = dict(g)
                g2["citekey"] = f"{ck}{chr(ord('a') + idx)}"
                out.append(g2)
    return out


def _normalize_for_vector_text(r: Dict[str, Any]) -> str:
    title = r.get("title") or ""
    abstract = r.get("abstract") or ""
    return f"{title}\n\n{abstract}".strip()


def _select_top_records(topic: str, records: List[Dict[str, Any]], k: int = 12) -> List[Dict[str, Any]]:
    """Embed with OpenAI and pick top-k by vector similarity using FAISS."""
    texts = [_normalize_for_vector_text(r) for r in records]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=records)
    sims = store.similarity_search(topic, k=k)
    return [d.metadata for d in sims]


def _records_minimal_json(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = ("citekey", "title", "abstract", "year", "authors", "venue", "doi", "url", "source")
    return [{k: r.get(k) for k in keep} for r in records]


def _gather_citekeys_from_text(sections: List[LiteratureSection]) -> List[str]:
    """Find patterns like (Smith2020) or (Smith2020a; Lee2022)."""
    citekeys: List[str] = []
    patt = re.compile(r"\(([^)]+)\)")
    for sec in sections:
        for match in patt.finditer(sec.body):
            inside = match.group(1)
            for token in [t.strip() for t in inside.split(";")]:
                if re.match(r"^[A-Za-z][A-Za-z]+[0-9]{3,4}[a-z]?$", token):
                    citekeys.append(token)
    # unique in order
    seen, uniq = set(), []
    for ck in citekeys:
        if ck not in seen:
            uniq.append(ck)
            seen.add(ck)
    return uniq


def _render_markdown(draft: LiteratureDraft, rendered_refs: List[str]) -> str:
    lines = [f"# {draft.title}", ""]
    if draft.summary:
        lines += ["## Executive Summary", draft.summary, ""]
    for s in draft.sections:
        lines += [f"## {s.heading}", s.body, ""]
    if draft.limitations:
        lines += ["## Limitations", draft.limitations, ""]
    if rendered_refs:
        lines += ["## References"]
        lines += [f"- {r}" for r in rendered_refs]
        lines.append("")
    return "\n".join(lines)


# ── Prompts (escaped braces) ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic writer.
Return ONLY a valid JSON object matching the LiteratureDraft schema:
- title: string
- summary: string (150–250 words)
- sections: array of {{heading, body}}
- limitations: string
- references: array of citekeys used in text

Rules:
- Ground claims strictly on the provided records (titles/abstracts).
- Use inline citations like (Smith2021) or (Lee2022a; Kim2023).
- Be precise, technical, and concise; avoid speculation.
- Do not invent sources or citekeys not in the records.
"""

HUMAN_PROMPT = """Topic: {topic}
Output language: {language}

You are given normalized records (JSON):
{records_json}

Write a literature review (~800–1200 words) with subheadings derived from the material.
Return a JSON object conforming to LiteratureDraft. No extra commentary.
"""


# ── Public API (keeps the response TYPE = string) ───────────────────────────

def generate_review(topic: str, citation_format: str = "raw", language: str = "English") -> str:
    """
    Returns a SINGLE STRING (markdown). This preserves the existing API response type.
    Internally uses LangChain structured output (Pydantic) and then renders markdown.
    """
    # 1) retrieval
    cr = CrossrefRetriever()
    sc = ScopusRetriever()
    rec_cr = cr.fetch_metadata(topic, k=50)
    try:
        rec_sc = sc.fetch_metadata(topic, k=50)
    except Exception:
        rec_sc = []

    merged = rec_cr + rec_sc
    if not merged:
        return f"# Literature review on: {topic}\n\n" \
               f"## Limitations\nNo records were retrieved from Crossref/Scopus for this query.\n"

    # 2) de-duplication and selection
    merged = _dedupe_citekeys(merged)
    top_records = _select_top_records(topic, merged, k=12)
    records_json = _records_minimal_json(top_records)

    # 3) LLM drafting with structured output (internal)
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    draft_llm = chat.with_structured_output(LiteratureDraft)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    chain = prompt | draft_llm
    draft: LiteratureDraft = chain.invoke(
        {"topic": topic, "language": language, "records_json": records_json}
    )

    # 4) references
    citekeys = draft.references or _gather_citekeys_from_text(draft.sections)
    by_ck = {r["citekey"]: r for r in merged}
    selected_records = [by_ck[ck] for ck in citekeys if ck in by_ck]
    formatter = CitationFormatter()
    rendered_refs = formatter.format(selected_records, style=citation_format)

    # 5) markdown render (STRING response)
    return _render_markdown(draft, rendered_refs)
