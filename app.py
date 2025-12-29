from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import trafilatura
import requests

app = FastAPI()

class ArticleRequest(BaseModel):
    url: str

@app.post("/extract")
def extract_article(data: ArticleRequest):
    try:
        response = requests.get(
            data.url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (ArticleExtractorBot/1.0)"
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch URL")
        
        downloaded = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False
        )
        
        if not downloaded:
            raise HTTPException(status_code=422, detail="Could not extract article")
        
        return {
            "url": data.url,
            "content": downloaded.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "article-extractor"}