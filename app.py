import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


class ArticleRequest(BaseModel):
    url: str


# Define request models
class NewsArticle(BaseModel):
    id: str
    summaries: str


class DeduplicateRequest(BaseModel):
    news_link_set_1: List[NewsArticle]
    news_link_set_2: List[NewsArticle]
    threshold: Optional[float] = 0.8


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


@app.post("/deduplicate")
def deduplicate_news(request: DeduplicateRequest):
    try:
        # Convert JSON input to DataFrames
        news_link_set_1 = pd.DataFrame([article.dict() for article in request.news_link_set_1])
        news_link_set_2 = pd.DataFrame([article.dict() for article in request.news_link_set_2])
        
        # Validate that summaries exist
        if 'summaries' not in news_link_set_1.columns or 'summaries' not in news_link_set_2.columns:
            raise HTTPException(status_code=400, detail="Missing 'summaries' field in input data")
        
        # Get embeddings
        news_link_set_1["embeddings"] = model.encode(
            news_link_set_1['summaries'].tolist(),
            normalize_embeddings=True
        )
        news_link_set_2["embeddings"] = model.encode(
            news_link_set_2['summaries'].tolist(),
            normalize_embeddings=True
        )

        # Compute similarity matrix
        sim_matrix = cosine_similarity(
            news_link_set_1["embeddings"].tolist(),
            news_link_set_2["embeddings"].tolist()
        )
        
        # For each article in news_link_set_1, find similar articles in news_link_set_2
        similar_ids_list = []
        similarities_list = []
        
        for i in range(len(news_link_set_1)):
            similarities = sim_matrix[i]
            
            # Information for where similarity >= threshold
            high_sim_indices = [j for j, sim in enumerate(similarities) if sim >= request.threshold]
            similar_ids = [news_link_set_2.iloc[j]['id'] for j in high_sim_indices]  # Changed from 'url' to 'id'
            similar_scores = [float(similarities[j]) for j in high_sim_indices]  # Convert to float for JSON serialization
            
            similar_ids_list.append(similar_ids)
            similarities_list.append(similar_scores)
        
        # Add new columns to news_link_set_1
        news_link_set_1['similar_ids'] = similar_ids_list
        news_link_set_1['similarities'] = similarities_list
        
        # Drop embeddings (not JSON serializable)
        news_link_set_1 = news_link_set_1.drop(columns=['embeddings'])
        news_link_set_2 = news_link_set_2.drop(columns=['embeddings'])
        
        # Convert to dict for JSON response
        result = news_link_set_1.to_dict(orient='records')
        
        return {
            "success": True,
            "results": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "article-extractor"}