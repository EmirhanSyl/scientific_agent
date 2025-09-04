import os
import requests
import html
from typing import List, Dict, Any

from .base import BaseRetriever

WOS_KEY = os.getenv("WOS_API_KEY")
WOS_URL = "https://api.clarivate.com/api/wos/v1"


class WosRetriever(BaseRetriever):
    """Retriever for Clarivate Web of Science."""

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not WOS_KEY:
            raise EnvironmentError("WOS_API_KEY not set")

        headers = {"X-ApiKey": WOS_KEY}
        params = {"databaseId": "WOK", "usrQuery": query, "count": k}
        resp = requests.get(WOS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        recs = (
            resp.json()
            .get("Data", {})
            .get("Records", {})
            .get("records", {})
            .get("REC", [])
        )

        results: List[Dict[str, Any]] = []
        for r in recs:
            static = r.get("static_data", {}) or {}
            summary = static.get("summary", {}) or {}

            # Title
            titles = (summary.get("titles", {}) or {}).get("title", []) or [{}]
            title = ""
            for t in titles:
                if isinstance(t, dict) and t.get("content"):
                    title = t["content"]
                    break

            # DOI
            dyn = r.get("dynamic_data", {}) or {}
            idents = (
                (dyn.get("cluster_related", {}) or {})
                .get("identifiers", {}) or {}
            ).get("identifier", []) or []
            doi = None
            for idobj in idents:
                t = (idobj.get("@type") or "").lower()
                if t == "doi" or "doi" in t:
                    doi = idobj.get("@value")
                    break

            # Abstract
            abstract = (
                (((summary.get("abstracts", {}) or {}).get("abstract", {}) or {}).get("p"))
                or ""
            )
            abstract = html.unescape(abstract) if abstract else ""

            # Year
            pub_info = summary.get("pub_info", {}) or {}
            year = pub_info.get("pubyear") or pub_info.get("@pubyear")

            # Authors
            names = (summary.get("names", {}) or {}).get("name", []) or []
            authors: List[str] = []
            surname = "Anon"
            if isinstance(names, list) and names:
                first = names[0]
                last_name = first.get("last_name") or ""
                first_name = first.get("first_name") or ""
                surname = last_name or "Anon"
                for n in names:
                    a_last = n.get("last_name") or ""
                    a_first = n.get("first_name") or ""
                    nm = f"{a_first} {a_last}".strip()
                    if nm:
                        authors.append(nm)

            citekey = f"{surname}{year or 'n.d.'}"

            results.append({
                "title": title,
                "doi": doi,
                "abstract": abstract,
                "year": year,
                "citekey": citekey,
                "authors": authors,
                "source": "wos",
            })

        return results
