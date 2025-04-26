from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def fetch_metadata(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Query the remote index and return list of metadata dictionaries."""
        raise NotImplementedError

    def _clean_text(self, text: str) -> str:
        """Utility to normalise whitespace."""
        return " ".join(text.split())
