import os
import requests
import html
from typing import List, Dict, Any

from .base import BaseRetriever

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusRetriever(BaseRetriever):
    """Retriever for Elsevier Scopus Search API with safer defaults."""

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        # Read environment at call-time (not import-time)
        api_key = os.getenv("ELSEVIER_API_KEY")
        inst_token = os.getenv("ELSEVIER_INST_TOKEN")
        if not api_key:
            raise EnvironmentError("ELSEVIER_API_KEY not set")

        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json",
            "httpAccept": "application/json",
        }
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token

        q = query.strip()
        # If the query is plain text, wrap it in TITLE-ABS-KEY(...)
        if "TITLE-ABS-KEY" not in q.upper() and any(ch.isalpha() for ch in q):
            q = f'TITLE-ABS-KEY("{q}")'

        params = {
            "query": q,
            "count": k,
            "view": "COMPLETE",
            "field": "dc:title,prism:doi,dc:description,prism:coverDate,dc:creator",
        }

        resp = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        sr = resp.json().get("search-results", {})
        entries = sr.get("entry", []) or []

        results: List[Dict[str, Any]] = []
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
                    "venue": None,
                    "url": None,
                    "source": "scopus",
                }
            )

        return results
