from typing import Dict, List

class CitationFormatter:
    def __init__(self):
        self._styles = {
            "raw": self._raw,
            "bibtex": self._bibtex,
            "apa7": self._apa7,
        }

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
        entry_type = "article" if r.get("source") == "crossref" else "misc"
        doi_line = f"  doi    = {{{r['doi']}}}\n" if r.get("doi") else ""
        return (
            f"@{entry_type}{{{r['citekey']}}},\n"
            f"  title  = {{{r.get('title','')}}},\n"
            f"  year   = {{{r.get('year','n.d.')}}},\n"
            f"{doi_line}"
            f"}}"
        )

    @staticmethod
    def _apa7(r: Dict) -> str:
        # Plain-text APA-like, no italics to match your example
        year = r.get("year") or "n.d."
        title = r.get("title") or "[No title]"
        parts = [f"{r['citekey']}. ({year}). {title}."]
        if r.get("doi"):
            parts.append(f"https://doi.org/{r['doi']}")
        return " ".join(parts)
