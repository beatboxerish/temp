import requests
import pandas as pd
import numpy as np
import re
import html as html_lib
import networkx as nx

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
from urllib.parse import urlparse, parse_qs, urlunparse, urljoin
import tldextract
from crawl4ai import AsyncWebCrawler

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

app = FastAPI()

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


SUMMARY_SENTENCES = 6

_tokenizer = Tokenizer("english")

SUMMARIZERS = {
    "lsa": LsaSummarizer,
    "lex_rank": LexRankSummarizer,
    "luhn": LuhnSummarizer,
    "reduction": ReductionSummarizer,
}


#  Pydantic Models

class ArticleRequest(BaseModel):
    url: str


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


class ArticleV2(BaseModel):
    id: str
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    full_text: Optional[str] = None
    created_at: Optional[str] = None
    pubDate: Optional[str] = None
    embeddings: Optional[List[float]] = None
    set_id: Optional[int] = None
    generated_summary: Optional[str] = None


class DeduplicateV2Request(BaseModel):
    articles: List[ArticleV2]
    threshold: Optional[float] = 0.8
    summarizer: Optional[str] = "lsa"


#  URL Utilities 

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
            page.wait_for_timeout(1500)
            final_url = page.url
        except PlaywrightTimeout:
            browser.close()
            return None

        browser.close()

        if "news.google.com" in final_url:
            return None

        return final_url


def clean_readability_output(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
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
        except Exception:
            pass

    return urlunparse(parsed).rstrip("/")


# Preprocessing
def clean_title(raw) -> str | float:
    if pd.isna(raw) or str(raw).strip().lower() == "null":
        return float("nan")
    t = html_lib.unescape(str(raw))
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else float("nan")


def clean_summary(raw) -> str | float:
    if pd.isna(raw) or str(raw).strip().lower() == "null":
        return float("nan")
    t = html_lib.unescape(str(raw))
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else float("nan")


def clean_full_text(raw) -> str | float:
    if pd.isna(raw) or str(raw).strip().lower() == "null":
        return float("nan")
    t = html_lib.unescape(str(raw))
    t = BeautifulSoup(t, "html.parser").get_text(separator="\n")
    t = re.sub(r"\{\{[^}]*\}\}", "", t)
    t = re.sub(r"[\-\u2013]\s*[A-Z]{2,6}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\(With inputs from[^)]+\)\.?", "", t)
    t = re.sub(r"Sign up:.*?(?=\n|$)", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t if t else float("nan")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["title_clean"] = df["title"].apply(clean_title)
    df["summary_clean"] = df["summary"].apply(clean_summary)
    df["full_text_clean"] = df["full_text"].apply(clean_full_text)
    df["is_truncated"] = (
        df["full_text"].astype(str).str.rstrip().str.endswith("..")
    )
    return df


# Extractive Summarization
def _build_full_input(row: pd.Series) -> str:
    parts = []
    for col in ("title_clean", "summary_clean", "full_text_clean"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return "\n".join(parts)


def _extractive_summarize(text: str, summarizer, n: int = SUMMARY_SENTENCES) -> str:
    parser = PlaintextParser.from_string(text, _tokenizer)
    sentences = summarizer(parser.document, n)
    return " ".join(str(s) for s in sentences)


def build_new_summaries(df: pd.DataFrame, summarizer_name: str):
    col = f"new_summary_{summarizer_name}"
    summarizer = SUMMARIZERS[summarizer_name]()

    summaries = []
    for _, row in df.iterrows():
        combined = _build_full_input(row)
        if combined:
            summaries.append(_extractive_summarize(combined, summarizer))
        else:
            summaries.append(float("nan"))

    df = df.copy()
    df[col] = summaries
    return df, col


# Embedding Generation 
def generate_embeddings(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# Similarity Matrix 
def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim = np.dot(embeddings, embeddings.T)
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


# Graph-Based Clustering 
def cluster_at_threshold(sim_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n = sim_matrix.shape[0]
    rows, cols = np.where(np.triu(sim_matrix >= threshold, k=1))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))

    cluster_ids = np.empty(n, dtype=int)
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            cluster_ids[node] = cid
    return cluster_ids


# Set-ID Assignment
def assign_set_ids(
    df: pd.DataFrame,
    old_indices: list[int],
    new_indices: list[int],
    sim_matrix: np.ndarray,
    threshold: float,
) -> tuple[dict[int, int], int]:
    """
    Returns (new_set_id_assignments, new_sets_created).
    new_set_id_assignments maps DataFrame index -> set_id for every new article.
    """
    old_setid_map: dict[int, int] = {
        i: int(df.at[i, "set_id"])
        for i in old_indices
        if df.at[i, "set_id"] is not None
    }

    existing_setids = list(old_setid_map.values())
    max_existing = max(existing_setids) if existing_setids else 0
    next_id = max_existing + 1
    new_sets_created = 0

    if not new_indices:
        return {}, 0

    # Step 1: for each new article, find which old set_ids it exceeds threshold with
    new_to_old_setids: dict[int, dict[int, float]] = {}
    for new_i in new_indices:
        connected: dict[int, float] = {}
        for old_i, sid in old_setid_map.items():
            sim = float(sim_matrix[new_i, old_i])
            if sim >= threshold:
                if sid not in connected or sim > connected[sid]:
                    connected[sid] = sim
        new_to_old_setids[new_i] = connected

    # Step 2: assign articles that connect to old clusters
    assigned: dict[int, int] = {}
    for new_i in new_indices:
        connected = new_to_old_setids[new_i]
        if len(connected) == 1:
            assigned[new_i] = next(iter(connected))
        elif len(connected) > 1:
            # Rule 2: highest similarity wins
            assigned[new_i] = max(connected, key=lambda sid: connected[sid])

    # Step 3: cluster unassigned new articles among themselves
    unassigned = [i for i in new_indices if i not in assigned]
    if unassigned:
        G = nx.Graph()
        G.add_nodes_from(unassigned)
        for a in range(len(unassigned)):
            for b in range(a + 1, len(unassigned)):
                i, j = unassigned[a], unassigned[b]
                if float(sim_matrix[i, j]) >= threshold:
                    G.add_edge(i, j)

        for component in nx.connected_components(G):
            group_set_id = next_id
            next_id += 1
            new_sets_created += 1
            for node in sorted(component):
                assigned[node] = group_set_id

    return assigned, new_sets_created


# Existing Endpoints 
@app.post("/extract")
def extract_article(data: ArticleRequest):
    try:
        url = data.url

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
                        "error": "Could not extract main content from URL content"}

            return {
                "url": data.url,
                "content": downloaded.strip()
            }
        except Exception:
            return {"url": data.url,
                    "content": response.text,
                    "error": "Could not extract main content from URL content"}

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
        news_link_set_1 = pd.DataFrame([article.dict() for article in request.news_link_set_1])
        news_link_set_2 = pd.DataFrame([article.dict() for article in request.news_link_set_2])

        if "embeddings" not in news_link_set_1.columns or "embeddings" not in news_link_set_2.columns:
            raise HTTPException(status_code=400, detail="Missing 'embeddings' field in input data")

        embeddings_1 = np.array(news_link_set_1["embeddings"].tolist(), dtype=np.float32)
        embeddings_2 = np.array(news_link_set_2["embeddings"].tolist(), dtype=np.float32)

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
            verbose=False,
            browser_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu"
            ]
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
            verbose=False,
            browser_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu"
            ]
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



@app.post("/extract-links")
def extract_links(data: ArticleRequest):
    try:
        url = data.url

        if is_google_news_url(url):
            url = resolve_google_news_url(url, timeout=100000)

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch content from URL")

        content_type = response.headers.get('Content-Type', '').lower()
        if not any(t in content_type for t in ['text/html', 'text/plain', 'application/xhtml']):
            raise HTTPException(status_code=422, detail=f"URL returned non-HTML content: {content_type}")

        soup = BeautifulSoup(response.text, "lxml")
        base_extract = tldextract.extract(url)
        base_domain = f"{base_extract.domain}.{base_extract.suffix}"
        links = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()

            if href.startswith(("javascript:", "mailto:", "tel:")):
                continue

            absolute_url = urljoin(url, href)

            extracted = tldextract.extract(absolute_url)
            link_domain = f"{extracted.domain}.{extracted.suffix}"
            if link_domain == base_domain:
                links.add(absolute_url)

        return {
            "url": data.url,
            "total_links": len(links),
            "links": list(links)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#/deduplicate_v2 
@app.post("/deduplicate_v2")
def deduplicate_v2(request: DeduplicateV2Request):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="No articles provided")

        if request.summarizer not in SUMMARIZERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown summarizer '{request.summarizer}'. Choose from: {list(SUMMARIZERS.keys())}"
            )

        threshold = request.threshold
        summarizer_name = request.summarizer

        # Build DataFrame
        df = pd.DataFrame([a.dict() for a in request.articles])

        # Ensure all expected columns exist
        for col in ["id", "url", "title", "summary", "full_text", "created_at",
                    "pubDate", "embeddings", "set_id", "generated_summary"]:
            if col not in df.columns:
                df[col] = None

        # Sort by created_at (ascending); articles without date go last
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df = df.sort_values("created_at", na_position="last").reset_index(drop=True)

        # Classify old vs new
        # OLD = has all three: embeddings, set_id, generated_summary
        def _is_old(row) -> bool:
            return row["set_id"] is not None and not pd.isna(row["set_id"])

        df["_is_old"] = df.apply(_is_old, axis=1)
        old_indices: list[int] = df.index[df["_is_old"]].tolist()
        new_indices: list[int] = df.index[~df["_is_old"]].tolist()
        new_count = len(new_indices)
        # Preprocess all articles (adds title_clean, summary_clean, full_text_clean)
        df = preprocess_dataframe(df)

        #  Summarize new articles 
        if new_indices:
            new_slice = df.loc[new_indices].copy()
            new_slice, summary_col = build_new_summaries(new_slice, summarizer_name)
            df.loc[new_indices, "generated_summary"] = new_slice[summary_col].values

        #  Embed new articles 
        if new_indices:
            embed_inputs = [
                str(v).strip() if pd.notna(v) and str(v).strip() else ""
                for v in df.loc[new_indices, "generated_summary"]
            ]
            new_embs = generate_embeddings(embed_inputs)
            for i, idx in enumerate(new_indices):
                df.at[idx, "embeddings"] = new_embs[i].tolist()

        #  Collect all embeddings 
        emb_dim = 384  # all-MiniLM-L6-v2 dimension
        all_embs: list[np.ndarray] = []
        for idx in df.index:
            raw = df.at[idx, "embeddings"]
            if raw is None:
                all_embs.append(np.zeros(emb_dim, dtype=np.float32))
            else:
                all_embs.append(np.array(raw, dtype=np.float32))

        all_embs_arr = np.array(all_embs, dtype=np.float32)

        # Similarity matrix 
        sim_matrix = compute_similarity_matrix(all_embs_arr)

        # Set-ID assignment 
        new_assignments, new_sets_created = assign_set_ids(
            df, old_indices, new_indices, sim_matrix, threshold
        )

        for idx, sid in new_assignments.items():
            df.at[idx, "set_id"] = sid

        #  Build response: only new articles with their processed fields 
        processed = []
        for idx in new_indices:
            emb = df.at[idx, "embeddings"]
            processed.append({
                "id": df.at[idx, "id"],
                "generated_summary": df.at[idx, "generated_summary"] if pd.notna(df.at[idx, "generated_summary"]) else None,
                "set_id": int(df.at[idx, "set_id"]) if pd.notna(df.at[idx, "set_id"]) else None,
                "embeddings": emb if emb is not None else [],
            })

        return processed

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def health_check():
    return {"status": "healthy", "service": "article-extractor"}
