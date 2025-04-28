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
        return f"{r['citekey']} – {r['doi']}"

    @staticmethod
    def _bibtex(r: Dict) -> str:
        entry_type = "article" if r["source"] == "crossref" else "misc"
        return (
            f"@{entry_type}{{{r['citekey']}}},\n"
            f"  title  = {{{r['title']}}},\n"
            f"  year   = {{{r['year']}}},\n"
            f"  doi    = {{{r['doi']}}}\n"
            f"}}"
        )

    @staticmethod
    def _apa7(r: Dict) -> str:
        return f"{r['citekey']}. ({r['year']}). *{r['title']}*. https://doi.org/{r['doi']}"
