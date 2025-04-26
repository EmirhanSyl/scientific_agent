import os
from rag_agent.retrievers.crossref import CrossrefRetriever

def test_crossref_basic():
    r = CrossrefRetriever()
    data = r.fetch_metadata("transformer models", k=3)
    assert len(data) > 0
    for item in data:
        assert 'title' in item
