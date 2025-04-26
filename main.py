import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

from rag_agent.agent import generate_review

load_dotenv()
app = FastAPI(title="RAG Literature Review API")

class Query(BaseModel):
    topic: str

@app.post("/literature-review")
async def literature_review(q: Query):
    if not q.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty")
    review = await generate_review(q.topic)
    return {"topic": q.topic, "review": review}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)


# TODO: METADATA EMPTY
# TODO: PROMPTU İYİLEŞTİR
# TODO: DİL SEÇENEĞİ EKLE
# TODO: REFERANS FORMATI SEÇENEĞİ EKLE (APA7 vs.)
# TODO: KULLANILAN KAYNAKÇALARI ÇIKAR VE REQUEST SONUCUNDA DÖNDÜR