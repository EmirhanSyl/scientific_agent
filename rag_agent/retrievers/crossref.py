import os
import requests
import html
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup, FeatureNotFound
from tenacity import retry, wait_exponential, stop_after_attempt
from .base import BaseRetriever

CROSSREF_API = "https://api.crossref.org/works"


class CrossrefRetriever(BaseRetriever):
    """Retriever for Crossref Works API with robust parsing & fallbacks.

    Key behaviors:
    - Uses `query.bibliographic` as recommended by Crossref.
    - Adds `mailto` in query string AND includes it in User-Agent for polite pool.
    - Pulls additional bibliographic fields (container-title, publisher, volume, issue, page).
    - Strips JATS/HTML from abstracts; returns title if no abstract is deposited.
    - Caps `rows` to 2000 (Crossref max). If nothing is found, retries with `query`.
    """

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
            or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
            or "n.d."
        )
        authors = item.get("author") or []
        surname = authors[0].get("family", "Anon") if authors else "Anon"
        title_words = re.sub(
            r"[^A-Za-z0-9 ]+", " ", ((item.get("title") or [""])[0])
        ).split()
        first_word = next((w for w in title_words if len(w) > 2), "Work")
        return f"{surname}{year}{first_word.capitalize()}"

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _request(self, params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        resp = requests.get(CROSSREF_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json() or {}

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        mailto = os.getenv("CROSSREF_MAILTO", "example@example.com")
        rows = max(1, min(int(k or 20), 2000))  # Crossref hard max
        params = {
            "query.bibliographic": query,
            "rows": rows,
            # lean response but keep essentials
            "select": "title,author,issued,DOI,abstract,container-title,publisher,volume,issue,page",
            "mailto": mailto,
        }
        headers = {
            "User-Agent": f"rag-agent/0.3 (+mailto:{mailto})"
        }

        data = self._request(params, headers)
        items = (data.get("message") or {}).get("items", [])

        if not items:  # fallback to broad `query`
            params.pop("query.bibliographic", None)
            params["query"] = query
            data = self._request(params, headers)
            items = (data.get("message") or {}).get("items", [])

        results: List[Dict[str, Any]] = []
        for it in items:
            title = (it.get("title") or [""])[0]
            doi = it.get("DOI") or None
            abstract_raw = it.get("abstract") or title
            clean_abs = self._strip_html(abstract_raw)
            _year = (
                it.get("issued", {}).get("date-parts", [[None]])[0][0]
                or it.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or it.get("published-online", {}).get("date-parts", [[None]])[0][0]
            )
            year = str(_year) if _year else None
            container = (it.get("container-title") or [""])
            venue = container[0] if container else ""

            results.append(
                {
                    "title": title,
                    "doi": doi,
                    "abstract": clean_abs,
                    "year": year,
                    "citekey": self._citekey(it),
                    "authors": [
                        ("{} {}".format(a.get("given", ""), a.get("family", "")).strip())
                        for a in (it.get("author") or [])
                    ],
                    "venue": venue,
                    "publisher": it.get("publisher"),
                    "volume": it.get("volume"),
                    "issue": it.get("issue"),
                    "pages": it.get("page"),
                    "source": "crossref",
                }
            )

        return results