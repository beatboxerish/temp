import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from urllib.parse import urlparse, parse_qs, urlunparse
import tldextract
from crawl4ai import AsyncWebCrawler

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


class ArticleRequest(BaseModel):
    url: str


# Define request models
class NewsArticle(BaseModel):
    id: str
    summaries: str


class EmbedRequest(BaseModel):
    id: str
    summaries: str


class EmbedResponse(BaseModel):
    id: str
    embeddings: List[float]

class NewsArticleEmbedding(BaseModel):
    id: str
    embeddings: List[float]


class DeduplicateEmbeddingRequest(BaseModel):
    news_link_set_1: List[NewsArticleEmbedding]
    news_link_set_2: List[NewsArticleEmbedding]
    threshold: Optional[float] = 0.8

class Crawl4aiLinksResponse(BaseModel):
    total_links: int
    links: List[str]



def is_google_news_url(url: str) -> bool:
    return (
        "news.google.com" in url
        or url.startswith("https://news.google.")
    )


def resolve_google_news_url(url: str, timeout=500000) -> str | None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(1500)  # allow redirect window

            final_url = page.url

        except PlaywrightTimeout:
            browser.close()
            return None

        browser.close()

        # skip if Google didn't redirect
        if "news.google.com" in final_url:
            return None

        return final_url
    
def clean_readability_output(html_content):
    """Clean HTML from readability output and extract plain text"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get text with proper spacing
    text = soup.get_text(separator='\n', strip=True)
    
    # Clean up multiple newlines
    import re
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def ensure_scheme(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def is_html_extension(url: str):
    blocked_ext = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif",
        ".webp", ".svg", ".doc", ".docx", ".xls",
        ".xlsx", ".zip", ".rar", ".mp4", ".mp3",
        ".avi", ".mov"
    )
    return not url.lower().endswith(blocked_ext)


def normalize_url(url: str):
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    query = parse_qs(parsed.query)

    if "page" in query:
        try:
            if int(query["page"][0]) >= 2:
                return None
        except:
            pass

    return urlunparse(parsed).rstrip("/")

@app.post("/extract")
def extract_article(data: ArticleRequest):
    try:
        url = data.url

        # Resolve Google News redirects (ONLY if needed)
        if is_google_news_url(url):
            url = resolve_google_news_url(url, timeout=100000)
            
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch content from URL")
        
        content_type = response.headers.get('Content-Type', '').lower()
        if not any(t in content_type for t in ['text/html', 'text/plain', 'application/xhtml']):
            raise HTTPException(status_code=422, detail=f"URL returned non-HTML content: {content_type}")
        
        
        try:
            downloaded = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False
            )
        
            if not downloaded:
                doc = Document(response.text)
                downloaded = clean_readability_output(doc.summary())
        
            if not downloaded:
                return {"url": data.url,
                        "content": response.text,
                        "error":"Could not extract main content from URL content"}

            return {
                "url": data.url,
                "content": downloaded.strip()
            }
        except:
            return {"url": data.url,
                    "content": response.text,
                    "error":"Could not extract main content from URL content"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=List[EmbedResponse])
def create_embeddings(request: List[EmbedRequest]):
    try:
        summaries = [x.summaries for x in request]

        embeddings = model.encode(
            summaries,
            normalize_embeddings=True
        )

        # Convert numpy arrays -> JSON safe list[float]
        result = []
        for i, item in enumerate(request):
            result.append({
                "id": item.id,
                "embeddings": embeddings[i].astype(float).tolist()
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/deduplicate")
def deduplicate_news(request: DeduplicateEmbeddingRequest):
    try:
        # Convert JSON input to DataFrames
        news_link_set_1 = pd.DataFrame([article.dict() for article in request.news_link_set_1])
        news_link_set_2 = pd.DataFrame([article.dict() for article in request.news_link_set_2])

        # Validate embeddings exist
        if "embeddings" not in news_link_set_1.columns or "embeddings" not in news_link_set_2.columns:
            raise HTTPException(status_code=400, detail="Missing 'embeddings' field in input data")

        # Convert embeddings list -> numpy array
        embeddings_1 = np.array(news_link_set_1["embeddings"].tolist(), dtype=np.float32)
        embeddings_2 = np.array(news_link_set_2["embeddings"].tolist(), dtype=np.float32)

        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings_1, embeddings_2)

        similar_ids_list = []
        similarities_list = []

        for i in range(len(news_link_set_1)):
            similarities = sim_matrix[i]

            high_sim_indices = [j for j, sim in enumerate(similarities) if sim >= request.threshold]
            similar_ids = [news_link_set_2.iloc[j]["id"] for j in high_sim_indices]
            similar_scores = [float(similarities[j]) for j in high_sim_indices]

            similar_ids_list.append(similar_ids)
            similarities_list.append(similar_scores)

        # Add columns to response
        news_link_set_1["similar_ids"] = similar_ids_list
        news_link_set_1["similarities"] = similarities_list

        result = news_link_set_1.to_dict(orient="records")

        return {
            "success": True,
            "results": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape-links-crawl4ai", response_model=Crawl4aiLinksResponse)
async def scrape_links_crawl4ai(data: ArticleRequest):

    try:
        url = ensure_scheme(data.url)

        async with AsyncWebCrawler(
            browser_type="chromium",
            headless=True,
            verbose=False
        ) as crawler:

            result = await crawler.arun(url=url)

            discovered_links = set()

            if result.links:
                internal_links = result.links.get("internal", [])

                for link_obj in internal_links:
                    href = link_obj.get("href")
                    if not href:
                        continue

                    normalized = normalize_url(href)
                    if not normalized:
                        continue

                    if is_html_extension(normalized):
                        discovered_links.add(normalized)

        links = sorted(discovered_links)

        return {
            "total_links": len(links),
            "links": links
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/extract-crawl4ai")
async def extract_article_crawl4ai(data: ArticleRequest):

    try:
        url = ensure_scheme(data.url)

        async with AsyncWebCrawler(
            browser_type="chromium",
            headless=True,
            verbose=False
        ) as crawler:

            result = await crawler.arun(url=url)
            html_content = result.html or ""

        extracted_content = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )

        if not extracted_content:
            return {
                "url": data.url,
                "content": "",
                "error": "Could not extract main content"
            }

        return {
            "url": data.url,
            "content": extracted_content.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "article-extractor"}
