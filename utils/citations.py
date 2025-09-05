from typing import Dict, List

class CitationFormatter:
    def __init__(self):
        self._styles = {
            "raw": self._raw,
            "bibtex": self._bibtex,
            "apa7": self._apa7,
        }

    # ── public ────────────────────────────────────────────────────────────
    def format(self, resources: List[Dict], style: str = "raw") -> List[str]:
        if style not in self._styles:
            raise ValueError(f"Unsupported citation style: {style}")
        return [self._styles[style](r) for r in resources]

    # ── style funcs ───────────────────────────────────────────────────────
    @staticmethod
    def _raw(r: Dict) -> str:
        doi_part = f" – {r['doi']}" if r.get("doi") else ""
        return f"{r['citekey']}{doi_part}"

    @staticmethod
    def _bibtex(r: Dict) -> str:
        entry_type = "article" if r.get("venue") else "misc"
        doi_line = f"  doi    = {{{r['doi']}}}\n" if r.get("doi") else ""
        venue_line = f"  journal= {{{r['venue']}}},\n" if r.get("venue") else ""
        return (
            f"@{entry_type}{{{r['citekey']}}},\n"
            f"  title  = {{{r.get('title','')}}},\n"
            f"{venue_line}"
            f"  year   = {{{r.get('year','n.d.')}}},\n"
            f"{doi_line}"
            f"}}"
        )

    @staticmethod
    def _apa7(r: Dict) -> str:
        year = r.get("year") or "n.d."
        title = r.get("title") or "[No title]"
        venue = r.get("venue")
        doi = r.get("doi")
        parts = [f"{r['citekey']}. ({year}). *{title}*"]
        if venue:
            parts.append(venue)
        if doi:
            parts.append(f"https://doi.org/{doi}")
        return " ".join(parts)
