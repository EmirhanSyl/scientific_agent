import os
import requests
import html
from typing import List, Dict, Any

from .base import BaseRetriever

API_KEY = os.getenv("ELSEVIER_API_KEY")
INST_TOKEN = os.getenv("ELSEVIER_INST_TOKEN")
SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusRetriever(BaseRetriever):
    """Retriever for Elsevier Scopus."""

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not API_KEY:
            raise EnvironmentError("ELSEVIER_API_KEY not set")

        headers = {
            "X-ELS-APIKey": API_KEY,
            "Accept": "application/json",
        }
        if INST_TOKEN:
            headers["X-ELS-Insttoken"] = INST_TOKEN

        params = {"query": query, "count": k}
        resp = requests.get(SCOPUS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        entries = resp.json().get("search-results", {}).get("entry", [])
        results: List[Dict[str, Any]] = []

        for e in entries:
            title = e.get("dc:title")
            doi = e.get("prism:doi")
            abstract = html.unescape(e.get("dc:description", "")) if e.get("dc:description") else ""

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

            results.append({
                "title": title,
                "doi": doi,
                "abstract": abstract,
                "year": year,
                "citekey": citekey,
                "authors": authors,
                "source": "scopus",
            })

        return results
