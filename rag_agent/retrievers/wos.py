import os
import requests
import html
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt

from .base import BaseRetriever

WOS_KEY = os.getenv("WOS_API_KEY")
WOS_URL = "https://api.clarivate.com/api/wos"  # Expanded API base


class WosRetriever(BaseRetriever):
    """Retriever for Clarivate Web of Science Expanded API.

    Notes:
    - Requires `WOS_API_KEY` entitlement and subscription.
    - Uses `databaseId=WOS` (per recent docs) and `usrQuery`.
    - Parses common fields; keeps code robust to minor response schema variations.
    """

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _request(self, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.get(WOS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or {}

    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not WOS_KEY:
            raise EnvironmentError("WOS_API_KEY not set")

        headers = {"X-ApiKey": WOS_KEY}
        page_size = max(1, min(int(k or 20), 50))
        params = {"databaseId": "WOS", "usrQuery": query, "count": page_size}

        results: List[Dict[str, Any]] = []
        first_record = 1
        while len(results) < k:
            params["firstRecord"] = first_record
            data = self._request(headers, params)

            # Try Expanded API JSON shape
            recs = (
                (data.get("Data") or {})
                .get("Records", {})
                .get("records", {})
                .get("REC", [])
            )

            if not recs and isinstance(data.get("hits"), list):  # Starter-like
                recs = data.get("hits", [])

            if not recs:
                break

            for r in recs:
                static = (r.get("static_data") or {}) if isinstance(r, dict) else {}
                summary = (static.get("summary") or {}) if isinstance(static, dict) else {}

                # Title
                titles = (summary.get("titles") or {}).get("title", [])
                title = ""
                if isinstance(titles, list):
                    for t in titles:
                        if isinstance(t, dict) and t.get("content"):
                            title = t["content"]
                            break

                # DOI
                dyn = (r.get("dynamic_data") or {}) if isinstance(r, dict) else {}
                idents = ((dyn.get("cluster_related") or {}).get("identifiers") or {}).get("identifier", [])
                doi = None
                for idobj in idents:
                    t = (idobj.get("@type") or "").lower()
                    if t == "doi" or "doi" in t:
                        doi = idobj.get("@value")
                        break

                # Abstract
                abstract = (
                    (((summary.get("abstracts") or {}).get("abstract") or {}).get("p"))
                    if isinstance(summary, dict)
                    else ""
                )
                abstract = html.unescape(abstract) if abstract else ""

                # Year / venue
                pub_info = summary.get("pub_info") or {}
                year = pub_info.get("pubyear") or pub_info.get("@pubyear")
                venue = (summary.get("pub_info") or {}).get("@pubtype") or None
                # Some responses include journal name in titles with type="source"
                if not venue and isinstance(titles, list):
                    for t in titles:
                        if t.get("@type") == "source" and t.get("content"):
                            venue = t["content"]
                            break

                # Authors
                names = (summary.get("names") or {}).get("name", [])
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
                    "venue": venue,
                    "publisher": None,
                    "volume": (pub_info.get("vol") or pub_info.get("@vol")),
                    "issue": (pub_info.get("issue") or pub_info.get("@issue")),
                    "pages": (pub_info.get("page") or pub_info.get("@page")),
                    "source": "wos",
                })

                if len(results) >= k:
                    break

            first_record += page_size

        return results