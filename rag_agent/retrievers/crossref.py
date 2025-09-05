import os
import requests
import html
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup, FeatureNotFound
from .base import BaseRetriever

CROSSREF_API = "https://api.crossref.org/works"


class CrossrefRetriever(BaseRetriever):
    """Retriever for Crossref Works API with robust parsing & fallbacks."""

    @staticmethod
    def _strip_html(raw: str) -> str:
        if not raw:
            return ""
        try:
            txt = BeautifulSoup(raw, "lxml").get_text(" ")
        except FeatureNotFound:
            txt = BeautifulSoup(raw, "html.parser").get_text(" ")
        txt = html.unescape(txt)
        return re.sub(r"\s+", " ", txt).strip()

    @staticmethod
    def _citekey(item: Dict[str, Any]) -> str:
        year = (
            item.get("issued", {}).get("date-parts", [[None]])[0][0]
            or item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            or "n.d."
        )
        authors = item.get("author") or []
        surname = authors[0].get("family", "Anon") if authors else "Anon"
        # keep a slightly more collision-resistant key but same field name
        title_words = re.sub(
            r"[^A-Za-z0-9 ]+", " ", ((item.get("title") or [""])[0])
        ).split()
        first_word = next((w for w in title_words if len(w) > 2), "Work")
        return f"{surname}{year}{first_word.capitalize()}"

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        mailto = os.getenv("CROSSREF_MAILTO", "example@example.com")
        params = {
            "query.bibliographic": query,
            "rows": k,
            "select": "title,author,issued,DOI,abstract",
            "mailto": mailto,
        }
        headers = {"User-Agent": f"rag-agent/0.2 (mailto:{mailto})"}

        resp = requests.get(CROSSREF_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])

        if not items:
            params.pop("query.bibliographic", None)
            params["query"] = query
            resp = requests.get(CROSSREF_API, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])

        results: List[Dict[str, Any]] = []
        for it in items:
            title = (it.get("title") or [""])[0]
            doi = it.get("DOI") or None
            abstract_raw = it.get("abstract") or title
            clean_abs = self._strip_html(abstract_raw)
            _year = (
                it.get("issued", {}).get("date-parts", [[None]])[0][0]
                or it.get("published-print", {}).get("date-parts", [[None]])[0][0]
            )
            year = str(_year) if _year else None

            results.append(
                {
                    "title": title,
                    "doi": doi,
                    "abstract": clean_abs,
                    "year": year,
                    "citekey": self._citekey(it),
                    "authors": [
                        ("{} {}".format(a.get("given", ""), a.get("family", ""))).strip()
                        for a in (it.get("author") or [])
                    ],
                    "source": "crossref",
                }
            )

        return results
