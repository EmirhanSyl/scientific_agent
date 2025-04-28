import os, requests, html, re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from .base import BaseRetriever

CROSSREF_API = "https://api.crossref.org/works"


class CrossrefRetriever(BaseRetriever):

    # ── Yardımcı --------------------------------------------------------------
    @staticmethod
    def _strip_html(raw: str) -> str:
        if not raw:
            return ""
        txt = BeautifulSoup(raw, "lxml").get_text(" ")
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
        return f"{surname}{year}"

    # ── Ana işlev -------------------------------------------------------------
    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        params = {
            "query": query,
            "rows": k,
            "select": "title,author,issued,DOI,abstract",
        }
        headers = {
            "User-Agent": (
                f"rag-agent/0.2 "
                f"(mailto:{os.getenv('CROSSREF_MAILTO', 'example@example.com')})"
            )
        }

        resp = requests.get(CROSSREF_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])

        results: List[Dict[str, Any]] = []
        for it in items:
            title = (it.get("title") or [""])[0]
            doi = it.get("DOI")
            abstract_raw = it.get("abstract") or title
            clean_abs = self._strip_html(abstract_raw)
            citekey = self._citekey(it)
            year = citekey[-4:] if citekey[-4:].isdigit() else None

            # Gürültülü / anlamsız kayıtları atla
            # if len(clean_abs.split()) < 30:
            #    continue
            if not doi:
                continue

            results.append(
                {
                    "title": title,
                    "doi": doi,
                    "year": year,
                    "citekey": citekey,
                    "abstract": clean_abs,
                    "authors": ["{} {}".format(a.get("given", ""), a.get("family", "")).strip()
                                for a in (it.get("author") or [])],
                    "source": "crossref",
                }
            )

        return results
