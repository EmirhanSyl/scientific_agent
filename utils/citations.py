from typing import Dict, List, Any


class CitationFormatter:
    def __init__(self):
        self._styles = {
            "raw": self._raw,
            "bibtex": self._bibtex,
            "apa7": self._apa7,
        }

    # ===== Structured =====
    @staticmethod
    def to_structured(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in resources:
            doi = r.get("doi")
            out.append(
                {
                    "citekey": r.get("citekey"),
                    "title": r.get("title"),
                    "doi": doi,
                    "url": f"https://doi.org/{doi}" if doi else None,
                    "authors": r.get("authors") or [],
                    "year": r.get("year"),
                    "venue": r.get("venue"),
                    "publisher": r.get("publisher"),
                    "volume": r.get("volume"),
                    "issue": r.get("issue"),
                    "pages": r.get("pages"),
                    "source": r.get("source"),
                }
            )
        return out

    # ===== String formatting =====
    def format(self, resources: List[Dict], style: str = "raw") -> List[str]:
        if style not in self._styles:
            raise ValueError(f"Unsupported citation style: {style}")
        return [self._styles[style](r) for r in resources]

    @staticmethod
    def _raw(r: Dict) -> str:
        doi_part = f" – {r['doi']}" if r.get("doi") else ""
        return f"{r['citekey']}{doi_part}"

    @staticmethod
    def _bibtex(r: Dict) -> str:
        entry_type = "article" if r.get("source") in {"crossref", "wos", "scopus"} else "misc"
        doi_line = f"  doi    = {{{r['doi']}}}\n" if r.get("doi") else ""
        venue_line = f"  journal= {{{r.get('venue','')}}}\n" if r.get("venue") else ""
        volume_line = f"  volume = {{{r.get('volume','')}}}\n" if r.get("volume") else ""
        number_line = f"  number = {{{r.get('issue','')}}}\n" if r.get("issue") else ""
        pages_line = f"  pages  = {{{r.get('pages','')}}}\n" if r.get("pages") else ""
        return (
            f"@{entry_type}{{{r['citekey']}}},\n"
            f"  title  = {{{r.get('title','')}}},\n"
            f"  year   = {{{r.get('year','n.d.')}}},\n"
            f"{venue_line}{volume_line}{number_line}{pages_line}{doi_line}"
            f"}}"
        )

    @staticmethod
    def _apa7(r: Dict) -> str:
        year = r.get("year") or "n.d."
        title = r.get("title") or "[No title]"
        venue = r.get("venue")
        parts = [f"{r['citekey']}. ({year}). {title}."]
        if venue:
            parts.append(venue + ".")
        if r.get("doi"):
            parts.append(f"https://doi.org/{r['doi']}")
        return " ".join(parts)