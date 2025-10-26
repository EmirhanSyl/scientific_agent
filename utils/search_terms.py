from __future__ import annotations

from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .llm import build_llm, with_structured_output


class SearchTerms(BaseModel):
    language: str = Field(..., description="Language of the terms")
    terms: List[str] = Field(
        ..., min_items=1, max_items=5,
        description="Concise database-friendly queries (use quotes, AND/OR)",
    )


SYSTEM = (
    "You are an expert search strategist for scientific databases.\n"
    "Return only JSON that matches the provided schema."
)

HUMAN = (
    "Research topic (may be long or verbose):\n{topic}\n\n"
    "Output language: {language}.\n"
    "Craft 1–3 concise, effective search queries for Crossref/Scopus/WoS.\n"
    "Guidelines: keep queries short; prefer TITLE-ABS-KEY style phrases; use quotes for exact phrases; avoid special characters not supported by the engines.\n"
)


def generate_search_terms(topic: str, language: str, max_terms: int = 3) -> List[str]:
    llm = build_llm()
    parser = with_structured_output(llm, SearchTerms)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    result: SearchTerms = (prompt | parser).invoke({"topic": topic, "language": language})
    terms = [t for t in (result.terms or []) if t and t.strip()]
    return terms[: max_terms]