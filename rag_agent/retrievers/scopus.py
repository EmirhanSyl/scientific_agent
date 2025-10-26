import os
import requests
import html
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt

from .base import BaseRetriever

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusRetriever(BaseRetriever):
    """Retriever for Elsevier Scopus Search API with safer defaults & pagination.

    Notes:
    - Requires `ELSEVIER_API_KEY`. Optional: `ELSEVIER_INST_TOKEN`.
    - Uses TITLE-ABS-KEY() wrapper when a plain natural string is provided.
    - Honors Scopus page size (count) defaults; paginates until `k` docs collected.
    - Extracts venue/volume/issue/pages when present in COMPLETE view.
    """

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _request(self, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or {}

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        api_key = os.getenv("ELSEVIER_API_KEY")
        inst_token = os.getenv("ELSEVIER_INST_TOKEN")
        if not api_key:
            raise EnvironmentError("ELSEVIER_API_KEY not set")

        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json",
        }
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token

        q = query.strip()
        if "TITLE-ABS-KEY" not in q.upper() and any(ch.isalpha() for ch in q):
            q = f'TITLE-ABS-KEY("{q}")'

        page_size = max(1, min(int(k or 20), 25))  # Scopus typical max per page ~25
        params = {
            "query": q,
            "count": page_size,
            "view": "COMPLETE",
            "field": (
                "dc:title,prism:doi,dc:description,prism:coverDate,dc:creator,"
                "prism:publicationName,prism:volume,prism:issueIdentifier,prism:pageRange"
            ),
        }

        results: List[Dict[str, Any]] = []
        start = 0
        while len(results) < k:
            params["start"] = start
            data = self._request(headers, params)
            sr = data.get("search-results", {})
            entries = sr.get("entry", []) or []
            if not entries:
                break

            for e in entries:
                title = e.get("dc:title")
                doi = e.get("prism:doi")
                abstract = e.get("dc:description") or ""
                if abstract:
                    abstract = html.unescape(abstract)

                cover_date = e.get("prism:coverDate")  # "YYYY-MM-DD"
                year = cover_date[:4] if cover_date and len(cover_date) >= 4 else None

                creator = e.get("dc:creator")  # "Surname, Given"
                if creator:
                    surname = creator.split(",")[0].strip()
                    authors = [creator]
                else:
                    surname = "Anon"
                    authors = []

                citekey = f"{surname}{year or 'n.d.'}"

                results.append(
                    {
                        "title": title,
                        "doi": doi,
                        "abstract": abstract,
                        "year": year,
                        "citekey": citekey,
                        "authors": authors,
                        "venue": e.get("prism:publicationName"),
                        "volume": e.get("prism:volume"),
                        "issue": e.get("prism:issueIdentifier"),
                        "pages": e.get("prism:pageRange"),
                        "publisher": None,
                        "source": "scopus",
                    }
                )

                if len(results) >= k:
                    break

            # prepare next page
            start += page_size

        return results