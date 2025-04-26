import os, requests, html
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
        recs = resp.json().get("Data", {}).get("Records", {}).get("records", {}).get("REC", [])
        results = []
        for r in recs:
            title = r.get("static_data", {}).get("summary", {}).get("titles", {}).get("title", [{}])[0].get("content","")
            doi = r.get("dynamic_data", {}).get("cluster_related", {}).get("identifiers", {}).get("identifier", [{}])[0].get("@value")
            abstract = r.get("static_data", {}).get("summary", {}).get("abstracts", {}).get("abstract", {}).get("p", "")
            results.append({
                "title": title,
                "doi": doi,
                "abstract": html.unescape(abstract) if abstract else "",
                "source": "wos"
            })
        return results
