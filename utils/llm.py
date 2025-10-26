from __future__ import annotations

from typing import Type, Any
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


LOCAL_CFG = dict(
    model="gpt-oss",
    base_url="http://localhost:11434",
    streaming=False,
    temperature=0.1,
    keep_alive=-1,
    # extra params accepted by langchain-ollama
    repeat_penalty=3,
    num_ctx=32768,
)


def build_llm() -> BaseChatModel:
    """Prefer local Ollama; fall back to OpenAI `gpt-5` if local is unavailable."""
    try:
        llm = ChatOllama(**LOCAL_CFG)
        # smoke test – will raise if server/model not reachable
        _ = llm.invoke("ping")
        return llm
    except Exception:
        return ChatOpenAI(model="gpt-5", temperature=0.2)


def with_structured_output(llm: BaseChatModel, schema: Type[Any]) -> BaseChatModel:
    """Get a structured-output wrapper, retrying with OpenAI if needed."""
    try:
        return llm.with_structured_output(schema)
    except Exception:
        # Fallback to OpenAI which has first-class structured output
        llm2 = ChatOpenAI(model="gpt-5", temperature=0.2)
        return llm2.with_structured_output(schema)