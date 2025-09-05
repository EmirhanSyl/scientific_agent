import os
from rag_agent.retrievers.crossref import CrossrefRetriever

def test_crossref_basic():
    # Ensure polite usage to avoid throttling
    os.environ.setdefault("CROSSREF_MAILTO", "you@example.com")
    r = CrossrefRetriever()
    data = r.fetch_metadata("transformer models", k=3)
    assert isinstance(data, list) and len(data) > 0
    for item in data:
        assert "title" in item
        assert "abstract" in item
