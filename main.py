import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

from rag_agent.agent import generate_review  # sync, returns str

load_dotenv()
app = FastAPI(title="RAG Literature Review API")

class ReviewRequest(BaseModel):
    topic: str
    citation_format: str | None = Query(
        default="raw",
        pattern="^(raw|bibtex|apa7)$",
    )
    language: str | None = Query(
        default="English",
        description="Output language e.g. English, Turkish, German …",
    )

@app.post("/literature-review")
async def literature_review(req: ReviewRequest):
    if not req.topic.strip():
        raise HTTPException(400, "Topic must not be empty")

    # generate_review is sync → run it in a worker thread
    out = await asyncio.to_thread(
        generate_review,
        topic=req.topic,
        citation_format=(req.citation_format or "raw").lower(),
        language=req.language or "English",
    )
    print(out)
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)
