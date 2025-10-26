from __future__ import annotations

from typing import List, Dict, Any
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from rag_agent.retrievers import CrossrefRetriever, ScopusRetriever, WosRetriever
from utils.citations import CitationFormatter
from utils.search_terms import generate_search_terms
from utils.llm import build_llm, with_structured_output


# ── Pydantic Models for Structured Output ───────────────────────────────────

class LiteratureSection(BaseModel):
    heading: str = Field(..., description="Section heading")
    body: str = Field(..., description="Paragraphs with inline (CITEKEY) citations")


class LiteratureDraft(BaseModel):
    """LLM-structured output of the review."""
    title: str
    summary: str
    sections: List[LiteratureSection]
    limitations: str
    references: List[str] = Field(default_factory=list, description="Citekeys referenced")


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
    venue = r.get("venue") or ""
    return f"{title}\n{venue}\n\n{abstract}".strip()


def _select_top_records(topic: str, records: List[Dict[str, Any]], k: int = 12) -> List[Dict[str, Any]]:
    """Embed with OpenAI and pick top-k by vector similarity using FAISS."""
    texts = [_normalize_for_vector_text(r) for r in records]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=records)
    sims = store.similarity_search(topic, k=k)
    return [d.metadata for d in sims]


def _records_minimal_json(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = (
        "title", "doi", "abstract", "year", "citekey", "authors",
        "venue", "publisher", "volume", "issue", "pages", "source"
    )
    return [{k: r.get(k) for k in keep} for r in records]


def _gather_citekeys_from_text(sections: List[LiteratureSection]) -> List[str]:
    """Find (Smith2020) or (Smith2020a; Lee2022) patterns."""
    citekeys: List[str] = []
    patt = re.compile(r"\(([^)]+)\)")
    for sec in sections:
        for match in patt.finditer(sec.body or ""):
            inside = match.group(1)
            for token in [t.strip() for t in inside.split(";")]:
                if re.match(r"^[A-Za-z][A-Za-z]+[0-9]{3,4}[a-z]?$", token):
                    citekeys.append(token)
    seen, uniq = set(), []
    for ck in citekeys:
        if ck not in seen:
            uniq.append(ck)
            seen.add(ck)
    return uniq


def _paren_to_bracket_citations(text: str, doi_map: Dict[str, str]) -> str:
    """
    Replace parentheses citations with bracket style:
      "(Smith2020; Lee2021a)" → "[Smith2020][DOI1] [Lee2021a][DOI2]"
    If DOI missing → just "[Smith2020]".
    """
    patt = re.compile(r"\(([^)]+)\)")

    def repl(m: re.Match) -> str:
        tokens = [t.strip() for t in m.group(1).split(";")]
        out_tokens: List[str] = []
        for tok in tokens:
            tok_clean = re.sub(r"[.,;:\s]+$", "", tok)
            if not tok_clean:
                continue
            doi = doi_map.get(tok_clean)
            if doi:
                out_tokens.append(f"[{tok_clean}][{doi}]")
            else:
                out_tokens.append(f"[{tok_clean}]")
        return " ".join(out_tokens)

    return patt.sub(repl, text or "")


def _draft_to_result_text(draft: LiteratureDraft, doi_map: Dict[str, str]) -> str:
    blocks: List[str] = []
    if draft.summary:
        blocks.append(_paren_to_bracket_citations(draft.summary, doi_map))
    for s in draft.sections or []:
        body = _paren_to_bracket_citations(s.body, doi_map)
        if s.heading:
            blocks.append(f"**{s.heading}**\n\n{body}")
        else:
            blocks.append(body)
    if draft.limitations:
        blocks.append(_paren_to_bracket_citations(draft.limitations, doi_map))
    return "\n\n".join([b for b in blocks if (b or "").strip()])


def _render_citations_list(selected_records: List[Dict[str, Any]], citation_format: str) -> List[str]:
    formatter = CitationFormatter()
    return formatter.format(selected_records, style=citation_format)


# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic writer.
Return ONLY a valid JSON object matching the LiteratureDraft schema:
- title: string
- summary: string (150–250 words)
- sections: array of {heading, body}
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


# ── Public API (returns dict) ───────────────────────────────────────────────

def _retrieve_all(topic: str, language: str, k_each: int = 50) -> List[Dict[str, Any]]:
    """Robust retrieval with search-term generation fallback."""
    cr = CrossrefRetriever()
    sc = None
    wos = None
    try:
        sc = ScopusRetriever()
    except Exception:
        sc = None
    try:
        wos = WosRetriever()
    except Exception:
        wos = None

    records: List[Dict[str, Any]] = []

    # 1) try original topic
    try:
        records.extend(cr.fetch_metadata(topic, k=k_each))
    except Exception:
        pass
    if sc:
        try:
            records.extend(sc.fetch_metadata(topic, k=k_each))
        except Exception:
            pass
    if wos:
        try:
            records.extend(wos.fetch_metadata(topic, k=k_each))
        except Exception:
            pass

    # 2) If empty OR topic looks very long/verbose → derive compact search terms
    if (not records) or (len(topic) > 160 or len(topic.split()) > 25):
        try:
            terms = generate_search_terms(topic, language, max_terms=3)
        except Exception:
            terms = []
        for t in terms:
            try:
                records.extend(cr.fetch_metadata(t, k=k_each))
            except Exception:
                pass
            if sc:
                try:
                    records.extend(sc.fetch_metadata(t, k=k_each))
                except Exception:
                    pass
            if wos:
                try:
                    records.extend(wos.fetch_metadata(t, k=k_each))
                except Exception:
                    pass

    return records


def generate_review(topic: str, citation_format: str = "raw", language: str = "English") -> Dict[str, Any]:
    """
    End-to-end pipeline that returns:
    {
      "query": <topic>,
      "result": <single string body with [CITEKEY][DOI] style>,
      "resources": <List[Dict] minimal records>,
      "citations": <List[Dict] structured metadata>,
      "references_formatted": {"style": <style>, "entries": [str, ...]}
    }
    """
    # 1) retrieval (with fallback search terms)
    merged = _retrieve_all(topic, language, k_each=50)

    if not merged:
        return {
            "query": topic,
            "result": "No records were retrieved from Crossref/Scopus/WoS for this query.",
            "resources": [],
            "citations": [],
            "references_formatted": {"style": citation_format, "entries": []},
        }

    # 2) de-duplication and selection
    merged = _dedupe_citekeys(merged)
    top_records = _select_top_records(topic, merged, k=12)
    records_min = _records_minimal_json(top_records)

    # 3) LLM drafting with structured output (local → OpenAI fallback)
    llm = build_llm()
    draft_llm = with_structured_output(llm, LiteratureDraft)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    draft: LiteratureDraft = (prompt | draft_llm).invoke(
        {"topic": topic, "language": language, "records_json": records_min}
    )

    # 4) Map citekeys -> DOI and select actually cited resources
    citekeys = draft.references or _gather_citekeys_from_text(draft.sections)
    by_ck: Dict[str, Dict[str, Any]] = {r["citekey"]: r for r in merged}
    selected_records = [by_ck[ck] for ck in citekeys if ck in by_ck]
    doi_map = {r["citekey"]: r["doi"] for r in selected_records if r.get("doi")}

    # 5) Build result text with bracket citations
    result_text = _draft_to_result_text(draft, doi_map)

    # 6) Render citations (structured + optionally formatted strings)
    formatter = CitationFormatter()
    citations_struct = formatter.to_structured(selected_records)
    citations_formatted = formatter.format(selected_records, style=citation_format)

    # 7) API object
    return {
        "query": topic,
        "result": result_text,
        "resources": _records_minimal_json(selected_records),
        "citations": citations_struct,
        "references_formatted": {"style": citation_format, "entries": citations_formatted},
    }