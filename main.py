import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

from rag_agent.agent import generate_review

load_dotenv()
app = FastAPI(title="RAG Literature Review API")

class ReviewRequest(BaseModel):
    topic: str
    citation_format: str | None = Query(
        default="raw",
        regex="^(raw|bibtex|apa7)$",
        description="Citation output style",
    )
@app.post("/literature-review")
async def literature_review(req: ReviewRequest):
    if not req.topic.strip():
        raise HTTPException(400, "Topic must not be empty")
    out = await generate_review(
        topic=req.topic,
        citation_format=req.citation_format.lower(),
    )
    print(out)
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)


# TODO: KULLANILAN KAYNAKÇALARI ÇIKAR VE REQUEST SONUCUNDA DÖNDÜR
# TODO: DİL SEÇENEĞİ EKLE
# TODO: REFERANS FORMATI SEÇENEĞİ EKLE (APA7 vs.)
