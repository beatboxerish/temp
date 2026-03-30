import requests
import pandas as pd
import numpy as np
import re
import html as html_lib
import networkx as nx
from dataclasses import dataclass, field

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
import asyncio
from urllib.parse import urlparse, parse_qs, urlunparse, urljoin
import tldextract
from crawl4ai import AsyncWebCrawler

import joblib
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

from langdetect import detect, LangDetectException

MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARY_SENTENCES = 6
MINHASH_NUM_PERM = 128
MINHASH_SHINGLE_K = 3

SUMMARIZERS = {
    "lsa": LsaSummarizer,
    "lex_rank": LexRankSummarizer,
    "luhn": LuhnSummarizer,
    "reduction": ReductionSummarizer,
}

# Globals populated at startup
model = None
_tokenizer = None
classifier = None
svm_model = None
market_tagger_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, classifier, _tokenizer, svm_model, market_tagger_model
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # embedding model
    model = SentenceTransformer(MODEL_NAME)

    # # zero-shot classifier
    classifier = hf_pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
        local_files_only=True
    )   
    _tokenizer = Tokenizer("english")
    svm_model = joblib.load("svm_model.pkl")
    market_tagger_model = joblib.load("LinearSVM_tfidf_model.pkl")
    yield

app = FastAPI(lifespan=lifespan)


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


class SummarizeEmbedRequest(BaseModel):
    articles: List[ArticleV2]
    summarizer: Optional[str] = "lsa"


class ArticleForDedup(BaseModel):
    id: str
    embeddings: List[float]
    set_id: Optional[int] = None


class DeduplicateV2Request(BaseModel):
    articles: List[ArticleForDedup]
    threshold: Optional[float] = 0.8


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
        return {},{}, 0

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
    similarities: dict[int, float] = {}
    for new_i in new_indices:
        connected = new_to_old_setids[new_i]
        if len(connected) == 1:
            sid = next(iter(connected))
            assigned[new_i] = sid
            similarities[new_i] = connected[sid]
        elif len(connected) > 1:
            # Rule 2: highest similarity wins
            sid = max(connected, key=lambda sid: connected[sid])
            assigned[new_i] = sid
            similarities[new_i] = connected[sid]

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
            component = sorted(component)
            for node in component:
                assigned[node] = group_set_id
                peers = [p for p in component if p != node]
                similarities[node] = float(max(sim_matrix[node, p] for p in peers)) if peers else 0.0

    return assigned, similarities, new_sets_created


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
        links = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()

            if href.startswith(("javascript:", "mailto:", "tel:")):
                continue

            absolute_url = urljoin(url, href)

            extracted = tldextract.extract(absolute_url)
            link_domain = f"{extracted.domain}.{extracted.suffix}"
            if link_domain not in BLOCKED_DOMAINS and  is_html_extension(absolute_url):
                links.add(absolute_url)

        return {
            "url": data.url,
            "total_links": len(links),
            "links": list(links)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 1: Summarize + Embed
@app.post("/summarize_and_embed")
def summarize_and_embed(request: SummarizeEmbedRequest):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="No articles provided")

        if request.summarizer not in SUMMARIZERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown summarizer '{request.summarizer}'. Choose from: {list(SUMMARIZERS.keys())}"
            )

        df = pd.DataFrame([a.dict() for a in request.articles])

        for col in ["id", "url", "title", "summary", "full_text", "created_at", "pubDate"]:
            if col not in df.columns:
                df[col] = None

        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        df = preprocess_dataframe(df)

        df, summary_col = build_new_summaries(df, request.summarizer)

        embed_inputs = [
            str(v).strip() if pd.notna(v) and str(v).strip() else ""
            for v in df[summary_col]
        ]
        embs = generate_embeddings(embed_inputs)

        ids, generated_summaries, embeddings = [], [], []
        for i, row in df.iterrows():
            ids.append(row["id"])
            generated_summaries.append(row[summary_col] if pd.notna(row[summary_col]) else None)
            embeddings.append(embs[i].tolist())

        return {"ids": ids, "generated_summaries": generated_summaries, "embeddings": embeddings}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 2: Deduplication (set_id assignment)
@app.post("/deduplicate_v2")
def deduplicate_v2(request: DeduplicateV2Request):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="No articles provided")

        threshold = request.threshold

        df = pd.DataFrame([a.dict() for a in request.articles])
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # OLD = has set_id, NEW = no set_id
        def _is_old(row) -> bool:
            return row["set_id"] is not None and not pd.isna(row["set_id"])

        df["_is_old"] = df.apply(_is_old, axis=1)
        old_indices: list[int] = df.index[df["_is_old"]].tolist()
        new_indices: list[int] = df.index[~df["_is_old"]].tolist()

        # Collect all embeddings
        emb_dim = 384
        all_embs: list[np.ndarray] = []
        for idx in df.index:
            raw = df.at[idx, "embeddings"]
            all_embs.append(np.array(raw, dtype=np.float32) if raw else np.zeros(emb_dim, dtype=np.float32))

        all_embs_arr = np.array(all_embs, dtype=np.float32)

        sim_matrix = compute_similarity_matrix(all_embs_arr)

        new_assignments, sim_scores, _ = assign_set_ids(
            df, old_indices, new_indices, sim_matrix, threshold
        )

        for idx, sid in new_assignments.items():
            df.at[idx, "set_id"] = sid

        ids, set_ids, similarities = [], [], []
        for idx in new_indices:
            ids.append(df.at[idx, "id"])
            set_ids.append(int(df.at[idx, "set_id"]) if pd.notna(df.at[idx, "set_id"]) else None)
            similarities.append(sim_scores.get(idx, 0.0))

        return {"ids": ids, "set_ids": set_ids, "similarities": similarities}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ─── Prefilter constants ───────────────────────────────────────────────────────

BLOCKED_DOMAINS = {
    'drishtiias.com',
    'n.tradingview.com',
    'in.tradingview.com',
    'instagram.com',
    'linkedin.com',
    'x.com',
    'youtube.com',
    'nature.com',
    'liberalcurrents.com',
    'insightsonindia.com',
}

MARKET_RESEARCH_SIGNALS: list = [
    # STRONG signals
    ("cagr_mention",        re.compile(r'\bcagr\b'),                                                    3.0),
    ("market_reach_year",   re.compile(r'market.*(?:reach|hit|surpass|exceed|cross).*\b20\d{2}\b'),     3.0),
    ("market_by_year",      re.compile(r'market.*\bby\b.*\b20\d{2}\b'),                                 2.5),
    ("billion_projection",  re.compile(r'(?:usd|us\$?|billion|bn|trillion).*\b20\d{2}\b'),              2.5),
    ("market_size_share",   re.compile(r'market[\w\s-]*(?:size|share)'),                                 2.5),
    ("forecast_keyword",    re.compile(r'\b(?:forecast|projection|projected)\b'),                        2.5),
    # MODERATE signals
    ("market_growth",       re.compile(r'market.*(?:growth|growing|grow)'),                              1.5),
    ("market_outlook",      re.compile(r'\boutlook\b.*(?:market|industry|sector|\b20\d{2}\b)'),          1.5),
    ("market_set_to",       re.compile(r'market[\w\s-]*set[\w\s-]*(?:to|for)'),                         1.5),
    ("market_expected",     re.compile(r'market[\w\s-]*(?:expected|poised|slated)'),                     1.5),
    ("market_trend",        re.compile(r'market[\w\s-]*trend'),                                         1.5),
    ("industry_analysis",   re.compile(r'(?:industry|sector)[\w\s-]*(?:analysis|report|insights)'),     1.5),
    ("market_report",       re.compile(r'market[\w\s-]*(?:report|research|study|survey|analysis)'),     1.5),
    ("market_insights",     re.compile(r'market[\w\s-]*insight'),                                       1.5),
    ("value_projection",    re.compile(r'(?:valued?|worth|estimated)[\w\s-]*(?:at|usd|us\$?)'),         1.5),
    # WEAK signals
    ("generic_market",      re.compile(r'\bmarket\b'),                                                  0.5),
    ("billion_mention",     re.compile(r'\b(?:billion|bn|trillion)\b'),                                 0.5),
    ("percentage_growth",   re.compile(r'\d+[\.-]?\d*\s*%'),                                            0.5),
    ("year_target",         re.compile(r'\b20(?:2[6-9]|3\d|4\d)\b'),                                   0.5),
    ("says_research_firm",  re.compile(r'(?:says|according)[\w-]*(?:research|intelligence|tmr|datam|allied|maximize|persistence|tbrc)'), 1.0),
]

COUNTER_SIGNALS: list = [
    ("corporate_results",   re.compile(r'\b(?:reports?|announces?|publishes?)\b.*\b(?:results?|earnings?|revenue)\b'), -3.0),
    ("product_launch",      re.compile(r'\b(?:launches?|unveils?|introduces?|secures?|partners?)\b'),                  -1.5),
    ("guide_playbook",      re.compile(r'\b(?:guide|playbook|handbook|toolkit|manual)\b'),                             -2.0),
    ("how_to",              re.compile(r'\bhow[\w-]'),                                                                  -1.5),
    ("specific_event",      re.compile(r'\b(?:acquir|merger|deal|invest|fund|award|appoint|hire|join)\b'),             -2.0),
    ("policy_regulation",   re.compile(r'\b(?:regulat|legislation|policy|law|ban|mandate|sanction)\b'),                -1.5),
]

MKT_HIGH_THRESHOLD   = 5.0
MKT_MEDIUM_THRESHOLD = 3.0
ZERO_CLF_THRESH = 0.8

KEYWORDS = [
    'carbon markets',
    'Refuel YYZ Facility',
    'DAKOTA PRAIRIE REFINING',
    'Cargill',
    'Aramco',
    'RedRock Biofuels',
    'Repsol Puertollano',
    'Infinium',
    'West Coast 100',
    'Praj',
    'Rotterdam renewable products refinery',
    'Nova Pangaea',
    'carbon tax',
    'United Airlines Ventures',
    'ifrs',
    'Negishi Refinery',
    'The Heide Region Development Agency',
    'nationally determined contributions',
    'Department of energy security and net zero',
    'refining capacity',
    'Hyundai Oilbank',
    'ENI Taranto Refinery',
    'HyNovera Plant',
    'Delta Air',
    'Diamond Green Diesel Holdings LLC',
    'Dansuk Industrial',
    'financed emissions',
    'Hainan Qilu Development (investor and subsidiary of Qilu Transportation Development Group)',
    'UK wind',
    'Cosmo Oil/Mitsui',
    'REG Geismar',
    'perform achieve trade',
    'Safran Corporate Ventures',
    'generation capacity',
    'RFS / Renewable Fuel standard',
    'Repsol',
    'ESG',
    'Exxonmobil',
    'Covenant Energy',
    'TotalEnergies SE',
    'Alléo Energy',
    'Ltd.',
    'Jet Zero Australia Pty Ltd',
    'Stuttgart Airport',
    'Södra',
    'World Energy',
    'Oregon CFP',
    'climate policy India',
    'fixed assets)',
    'MoEFCC',
    'climate fund',
    'AltAir Paramount',
    'DSL-01 project',
    'Indian corporate partnership',
    'climate disclosure',
    'biomass',
    'GFANZ',
    'Panama Government',
    'ZEV',
    'emissions reduction',
    'Oceania Biofuels',
    'Government of Newfoundland and Labrador',
    'Cielo Waste Solutions',
    'California Air Resources Board',
    'Ostrand Biorefinery',
    'fuel cell',
    'Plock Plant',
    'climate legislation',
    'climate target',
    'bp',
    'Oriental Energy',
    'Sasol ecoFT',
    'liquid natural gas',
    'Royal Schiphol Group',
    'Jaxon Energy',
    'LG Chem',
    'Kern Energy',
    'cap and trade',
    'BC LCFS',
    'Washington CFS',
    'P2X-Europe',
    'TotalEnergies BioTFuel Plant',
    'The HyKero plant',
    'national hydrogen mission',
    'biomethane',
    'KLM Royal Dutch Airlines',
    'CDR ',
    'Quantafuel',
    'BP Co-processing Refinery',
    'SCA',
    'JGC Holdings Corp.',
    'electric vehicle',
    'University of Southampton',
    'carbon footprint',
    'Velocys',
    'Shell Energy and Chemicals Park',
    'Grandpuits Refinery',
    'hydrogen',
    'Neste',
    'Vattenfall',
    'Gevo/ New-Zero 1 plant',
    'Shell',
    'WasteFuel Project',
    'Carbon offsets',
    'twh',
    'Neste Renewable Fuels Oy',
    'capacity addition',
    'AtmosFUEL',
    'Marubeni',
    'Dimensional Energy/Heliogen Plant',
    'climate risk',
    'CCUS ',
    'ESG India',
    'biofuel',
    'TCFD',
    'Petronor Plant',
    'HCS Group',
    'Freedom Pines Fuels',
    'electrolyzer',
    'Ryze Renewables',
    'Europe wind',
    'Lighthouse Green Fuel',
    'BEV',
    'Lawter',
    'NOM',
    'American Carbon Registry ',
    'green hydrogen mission',
    'TotalEnergies',
    'Port of Amsterdam',
    'megawatt',
    'Suria Capital',
    'SBTi ',
    'CEMEX',
    'Refuel Energy Inc.',
    'LanzaJet',
    'DG Fuels',
    'Plaju Refinery',
    'carbon credits price',
    'Alder Fuels',
    'Waste to Sustainable Aviation Fuel (WtF) plant',
    'bcm',
    'carbon credit ',
    'Etihad Airways',
    'Växjö Energi',
    'World Energy Houston biodiesel plant',
    'Cosmo Oil Co.',
    'Tidewater Renewables',
    'Kiko Ventures',
    'the city of Amsterdam',
    'East Kansas Agri-Energy',
    'carbon pricing',
    'diesel',
    'Revo International Inc.',
    'pollution',
    'Brasil BioFuels',
    'OMV Petrom',
    'Beijing Sanju (primary owner',
    'advanced clean truck',
    'Sanju Biofuels',
    'biodiversity',
    'PHILLIPS 66 COMPANY',
    'Braya Renewable Fuels',
    'Heliogen',
    'green bonds',
    'Technip Energies',
    'electricity generation',
    'JGC Holdings Corporation',
    'PT Pertamina',
    'CBAM',
    'carbon',
    'carbon neutrality',
    'Marathon Petroleum Corporation',
    'Diamond Green Diesel',
    'ECO Environmental',
    'SEC climate disclosure',
    'ecology',
    'Honeywell',
    'power grid',
    'St. Joseph Renewable Fuels LLC',
    'waste-to-energy',
    'ENOVA',
    'greenhouse gas',
    'Aemetis',
    'Oregon DEQ',
    'blended finance',
    'Synhelion AG',
    'green H2',
    'gas',
    'Thanachok Oil Light Co.',
    'Inc',
    'advanced clean car',
    'RPO',
    'ESG score',
    'BioRefinery Cilacap',
    'sustainability action',
    'Greenergy biofuels plant',
    'Livorno Biorefinery',
    'Paris Agreement',
    'CCUS',
    'Euromovement Industriepark GmbH',
    'ESG disclosure',
    'OMV AG',
    'Eni',
    'PHEV',
    'SAS',
    'Lanzajet and Marquis integrated renewable plant',
    'Par Pacific Holdings',
    'EssarOil UK',
    'Cosmo Oil',
    'Climeworks',
    'DAWN Facility',
    'Synkero',
    'European Netwrok of Transmission System operators for electricity',
    'climate investment',
    'bioCNG',
    'Tupras',
    'sustainable aviation fuel',
    'Frontier Impact Group',
    'EDL Anlagenbau Gesellschaft',
    'Dansuk',
    'IAG',
    'climate finance',
    'transition finance',
    'Vandelay Ventures',
    'Strategic Biofuels',
    'geothermal',
    'Galp Energia',
    'hydrogen fuel',
    'Sabah Maju Jaya Renewable Energy Plant',
    'Emirates National Oil Company',
    'CCTS',
    'net-zero',
    'sustainable equity',
    'SkyNRG',
    'Finnish Government',
    'emissions trading',
    'COP30',
    'Cosmo Oil’s Sakai Refinery',
    'Enerkem Rotterdam Facility',
    'co2',
    'Canada CFR',
    'installed capacity',
    'CORSIA DAC',
    'LCFS',
    'net zero',
    'integrated reporting',
    'LG Chem/Dansuk',
    'OGE',
    'Airbus',
    'Scope 1',
    'Altalto Velocys',
    'BSGF',
    'net zero pledge',
    'renewable natural gas',
    'Atmosfair',
    'EV adoption',
    'Dubai Municipality',
    'voluntary carbon ',
    'Petronor',
    'Renewable Energy Group',
    'Fulcrum Bioenergy',
    'solar',
    'Sierra Biofuels Plant',
    'legislative offsets',
    'UPM Biofuels',
    'Kern Oil & Refining Co.',
    'Washington Ecology',
    'alternative jet fuel',
    'mtpa',
    'ECB Group',
    'green ammonia',
    'St1',
    'Neste Singapore Refinery',
    'financing intervention',
    'Cepsa',
    'e-fuels',
    'carbon border tax',
    'USA Bioenergy',
    'Fulcrum BioEnergy',
    'Emerald Biofuels',
    'GHG emissions',
    'SAFuels X Facility',
    'Chevron Corporation',
    'blending mandate',
    'FCEV',
    'sustainable',
    'Saudi Aramco',
    'BRSR',
    'Gunvor Group Ltd',
    'Siemens Energy',
    'carbon credit trading scheme',
    'hydrocarbon',
    'marquis saf',
    'emissions allowances',
    'Seaboard Energy',
    'Acelen Renewables',
    'Houston biodiesel plant',
    'natural gas',
    'ENERTRAG',
    'PBF Energy',
    'ethanol',
    'Juxian County Government (minority owner',
    'Project Speedbird',
    'marquis sustainable aviation fuel',
    'Ørsted Deutschland',
    'mwh',
    'california air resources board',
    'Indaba Renewable Fuels',
    'Zenid',
    'The Port of Rotterdam',
    'carbon market',
    'aviation',
    'Byogy',
    'Galp',
    'Emission Trading System',
    'Fidelis New Energy',
    'carbon disclosure',
    'ultra-low sulfur',
    'Phillips 66 Renewable Plant',
    'Heide’s municipal utility',
    'Repsol Tarragona',
    'Thüga Aktiengesellschaft',
    'Swedish Biofuels/COWI',
    'sustainability report',
    'Washington Clean Fuel',
    'Indian corporate climate',
    'power-to-X',
    'Depart for Transport',
    'Aemetis biofuel production plant',
    'Alfanar',
    'Trafigura',
    'BP Kwinana Refinery',
    'SAF + Consortium',
    'SkyNRG/Stuttgart Airport/Schwenk Zement',
    'LanzaJet Inc',
    'corporate climate commitment',
    'climate disclosure goals',
    'St1 Gothenburg Biorefinery',
    'renewable',
    'carbon fund',
    'National Energy System Operator',
    'Rotterdam The Hague Airport',
    'carbon intensity',
    'electrolysis',
    'carbon capture',
    'climate bonds',
    'Hynamics Deutschland GmbH',
    'decarbonization',
    'Clean Energy Ventures',
    'Hy2gen AG',
    'Hy2gen',
    'BP Plc',
    'CVR Energy',
    'Thyssenkrupp Industrial Solutions',
    'East Kansas Agri-Energy LLC',
    'Darling Ingredients Inc',
    'carbon futures',
    'Silverpeak',
    'fleet electrification',
    'stranded assets',
    'fossil fuel',
    'Enerkem',
    'MNRE',
    'Lanza Jet',
    'renewable energy India',
    'renewable diesel',
    'Preem',
    'Engie New Ventures',
    'voluntary carbon market',
    'ETS',
    'Global Clean Energy Holdings Inc.',
    'Oregon clean fuel',
    'global carbon market',
    'The Bayou Fuels biorefinery',
    'Walter Pincus',
    'Bureau of Energy Efficiency',
    'climate policy',
    'bfsi',
    'Sumitomo Corporation of Americas',
    'ConocoPhillips',
    'Climate Action Reserve ',
    'green hydrogen',
    'Sveaskog',
    'Sinopec',
    'Stonebriar Commercial Finance',
    'Shell Energy And Chemicals Park',
    'BP Castellon',
    'Omega Green Plant',
    'carbon sequestration ',
    'Alder Greencrude (AGC) production facility',
    'SBTi',
    'low carbon fuel',
    'sustainable finance',
    'JANGADA Project',
    'BESIX',
    'compliance market',
    'Gela Biorefinery',
    'SHV Energy',
    'West Coast University of Applied Sciences',
    'EPA regulations',
    'oil',
    'fuel pathway',
    'energy',
    'renewable energy',
    'CEPCO Power Plant',
    'BP Products North America',
    'Dimensional Energy',
    'Bolivian government',
    'Raffinerie Heide GmbH',
    'PM Surya Ghar',
    'Hainan Qilu Development',
    'Repsol Petronor Plant',
    'Scope 2',
    'Braavos Capital',
    'decarbonize',
    'Fulcrum NorthPoint',
    'PKN ORLEN',
    'energy storage',
    'Scope 3',
    'green finance',
    'science-based targets',
    'feedstock',
    'LanzaTech',
    'Vertex Energy Inc.',
    'AIC Energy',
    'FCL',
    'battery',
    'carbon registry',
    'Jet 2',
    'KIRAM',
    'Gevo Silsbee Refinery',
    'Synhelion',
    'OMV Schwechat Refinery',
    'VCM ',
    'mmbtu',
    'E-fuel1',
    'LLC',
    'World Energy Paramount Facility',
    'charging infrastructure',
    'greenhouse',
    'Cresta Fund Management',
    'battery storage',
    'Alternate fuel',
    'coal',
    'Valero Energy Corp.',
    'ESG funds',
    'Washington cap and Invest',
    'Europe energy',
    'SGP BioEnergy',
    'clean energy',
    'barrel',
    'ESG rating',
    'Gron Fuels',
    'Ltd. (29%)',
    'carbon credit',
    'Mitsui & Co.',
    'production capacity',
    'LanzaTech UK',
    'Project DRAGON',
    'Wyoming Renewable Diesel Company LLC',
    'transition',
    'wind power',
    'Petrixo Oil & Gas',
    'impact investing',
    'Gold Standard',
    'Carnarvon Energy',
    'Bangchak Corporation Plc.',
    'EV mandate',
    'Convent Refinery',
    'Centrepoint Biorefinery',
    'Marathon Petroleum Corp.',
    'Private Investors',
    'KLM',
    'power plant',
    'green taxonomy',
    'wind farmwind energy',
    'Porvoo Refinery',
    'voluntary offset ',
    'Gevo',
    'blue carbon',
    'clean fuel',
    'ESG news',
    'British Columbia low carbon fuel',
    'fossil fuel divestment',
    'British Airways',
    'Södra skogsägarna',
    'Eni Sustainable Mobility',
    'ISSB',
    'Norsk E-Fuel',
    'Engie',
    'jet fuel',
    'Repsol Cartagena',
    'Oaktree Capital Management L.P.',
    'NEXT Renewable Fuels',
    'Holcim Germany Group',
    'Uniper',
    'li-ion',
    'carbon India',
    'investor and tech provider)',
    'SCHWENK Zement',
    'Alberta TIER',
    'La Mede Biorefinery',
    'Green Fuels Hamburg',
    'Indian carbon market',
    'idemitsu chiba complex',
    'carbon neutral',
    'tonne',
    'paris agreement',
    'Engie/ Safran Plant',
    'gigawatt',
    'The Love’s Family of Companies',
    'The Navigator Company',
    'CastleRock Green Energy',
    'Rotterdam The Hague Innovation Airport',
    'sustainability-linked bonds',
    'United Airline Ventures',
    'Darling Ingredients Inc.',
    'renewable purchase obligation',
    'HF Sinclair Corporation',
    'Phillips 66',
    'Euglena',
    'EV charging',
    'RFS LCFS',
    'carbon removal ',
    'transportation fuel',
    'REDD+ ',
    'RISE Research Institutes of Sweden',
    'IC-VCM ',
    'Parkland Fuel Corp.',
    'Beijing Haixin Energy Technology',
    'Engie/Infinium',
    'terawatt',
    'carbon budget',
    'Verra',
    'Solent Partners',
    'EPA (Environmental Protection agency)',
    'zero emission vehicle',
    'IHS Markit',
    'GRI',
    'bpd',
    'Beijing Sanju',
    'AVAPCO LLC',
    'European environment agency',
    'double materiality',
    'Valero Energy Corporation',
    'SAF RD RNG',
    'Ethanol',
    'Tadweer',
    'Enilive',
    'reforestation',
    'Cosmo Oil Co. Ltd.',
    'solar PV',
    'oil and gas',
    'German Federal Ministry for Economic Affairs and Climate Action (BMWK)',
    'gwh',
    'carbon offset',
    'saf',
    'BBGI Plc.',
    'Fulcrum Sierra BioFuels',
    'Neste Singapore Pte Ltd',
    'lithium ion',
    'Renewable identification number',
    'Doral Energy-Tech Ventures',
    'Grupo BBF',
    'Kansai Airports',
    'Synthetic SAF Plant',
    'Sinclair Wyoming Refining Company',
    'Vibra Energia',
    'ReadiFuels',
    'offshore wind',
    'TechEnergy Ventures',
    'Shell Rhineland refinery',
    'Verra ',
    '2030-sekretariatet',
    'emissions',
    'electrification',
    'carbon price',
    'PAT scheme',
    'Petrobrazi Refinery',
    'Ensyn Technologies Inc',
    'decarbonise',
    'sustainable investment',
    'clean fuels',
    'carbon capture ',
    'Rocky Mountain Clean Fuels Inc.',
    'Solarbelt FairFuel gGmbH',
    'ESG regulations',
    'CEMEX/SASOL',
    'European Netwrok of Transmission System operators for gas',
    'Schiphol',
    'environment',
    'Eco Biochemical Technology'
    # ── Capacity / unit keywords ──────────────────────────────────────────────
    'terawatt', 'gigawatt', 'megawatt',
    'mwh', 'gwh', 'twh',
    'mmbtu',
    'barrel', 'bpd',
    'bcm', 'mtpa',
    'installed capacity', 'generation capacity', 'production capacity',
    'refining capacity', 'capacity addition',
]

KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]

# ── Financial/earnings keywords to filter out ────────────────────────────────
FINANCIAL_KEYWORDS = ['earnings release', 'financial results', 'stocks', 'stock', 'buy back', 'stocks to watch']

# ── Content filter keywords  ────────────────────────────────────────
CONTENT_FILTER_PHRASES = ['publish research', 'award event', 'award function', 'research institute']
CONTENT_FILTER_EXACT   = ['global survey']
URL_FILTER_PHRASES     = ['research', 'masterclass', 'online learn']

# ── Zero shot classifier classes  ────────────────────────────────────────
candidate_labels = ["Awards / Rankings / Recognition", "Corporate Earnings / Stock Market / IPO", 
                    "Research Reports / Historic Analysis / Third party research", 
                    "Renewables / Energy Projects / Carbon market/ clean fuel market/ offset market"]



# ─── Prefilter helpers ─────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str | None:
    try:
        from urllib.parse import urlparse, unquote
        host = urlparse(unquote(str(url))).netloc.lower()
        return re.sub(r"^www\.", "", host) or None
    except Exception:
        return None
    

def detect_lang(text):
    """Return detected language code, or 'unknown' on failure."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def _extract_slug(url: str) -> str:
    """Extract and normalise the last path segment from an already-decoded URL."""
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path
        segments = [s for s in path.strip('/').split('/') if s]
        if not segments:
            return ''
        slug = segments[-1]
        slug = re.sub(r'\.\w{2,4}$', '', slug)
        slug = slug.replace('-', ' ').replace('_', ' ').lower()
        return slug
    except Exception:
        return ''


def _extract_slug_for_lsh(url: str) -> str:
    from urllib.parse import urlparse, unquote
    try:
        url = unquote(url)
        path = urlparse(url).path.rstrip('/')
        segments = [s for s in path.split('/') if s]
        if not segments:
            return ''
        slug = segments[-1]
        slug = re.sub(r'\.\w{2,4}$', '', slug)
        slug = slug.replace('-', ' ').replace('_', ' ').lower()
        return slug
    except Exception:
        return ''


def _make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM, k: int = MINHASH_SHINGLE_K) -> MinHash | None:
    if not text or len(text) < k:
        return None
    m = MinHash(num_perm=num_perm)
    text = text.lower()
    for i in range(len(text) - k + 1):
        m.update(text[i:i + k].encode('utf-8'))
    return m


def _score_text(text: str) -> tuple[float, list, list]:
    score = 0.0
    matched_signals, matched_counters = [], []
    for name, pattern, weight in MARKET_RESEARCH_SIGNALS:
        if pattern.search(text):
            score += weight
            matched_signals.append(f"{name}(+{weight})")
    for name, pattern, weight in COUNTER_SIGNALS:
        if pattern.search(text):
            score += weight
            matched_counters.append(f"{name}({weight})")
    return round(score, 1), matched_signals, matched_counters


def _classify_text(
    text: str,
    high_thresh: float = MKT_HIGH_THRESHOLD,
    medium_thresh: float = MKT_MEDIUM_THRESHOLD,
) -> tuple[str, float]:
    """Returns (confidence, score): confidence is HIGH / MEDIUM / LOW."""
    score, _, _ = _score_text(text.lower().strip() if text else '')
    if score >= high_thresh:
        return 'HIGH', score
    elif score >= medium_thresh:
        return 'MEDIUM', score
    return 'LOW', score


def keyword_score(title_clean: str, summary: str, full_text_clean: str) -> tuple[float, list[str], dict]:
    """
    Score an article by distinct ESG/climate keyword matches + frequency bonus.

    Returns:
        (final_score, matched_keywords, keyword_counts)
    """
    text = " ".join(
        v for v in [title_clean, summary, full_text_clean]
        if v and str(v).strip().lower() not in ("", "null", "nan")
    ).lower()

    if not text.strip():
        return 0.0, [], {}

    keyword_counts: dict[str, int] = {}
    for kw in KEYWORDS_LOWER:
        count = text.count(kw)
        if count > 0:
            keyword_counts[kw] = count

    if not keyword_counts:
        return 0.0, [], {}

    matched = list(keyword_counts.keys())
    distinct_score      = len(matched)
    frequency_score     = sum(min(c, 10) for c in keyword_counts.values())
    diversity_multiplier = 1 + 0.1 * (distinct_score - 1)
    final_score = round((distinct_score + frequency_score) * diversity_multiplier, 2)

    return final_score, matched, keyword_counts


# ─── Stemmed keyword scoring (v2) ────────────────────────────────────────────
from nltk.stem.snowball import SnowballStemmer

_stemmer = SnowballStemmer("english")


def _stem_phrase(phrase: str) -> str:
    """Stem each word in a phrase and rejoin."""
    return " ".join(_stemmer.stem(w) for w in phrase.split())


def _stem_text(text: str) -> str:
    """Stem every word in a text string."""
    return " ".join(_stemmer.stem(w) for w in text.split())


# Pre-compute stemmed keywords once at import time
_STEMMED_KW_MAP: dict[str, str] = {}    # {stemmed_phrase: original_keyword}
_STEMMED_KW_LIST: list[str] = []

for _kw in KEYWORDS_LOWER:
    _st = _stem_phrase(_kw)
    if _st not in _STEMMED_KW_MAP:
        _STEMMED_KW_MAP[_st] = _kw
        _STEMMED_KW_LIST.append(_st)

_STEMMED_KW_PATTERNS: dict[str, re.Pattern] = {
    sk: re.compile(r'\b' + re.escape(sk) + r'\b')
    for sk in _STEMMED_KW_LIST
}


def keyword_score_v2(title_clean: str, summary: str, full_text_clean: str) -> tuple[float, list[str], dict]:
    """
    Score an article using stemmed word-boundary keyword matching.

    Same scoring formula as keyword_score() but:
      - Both keywords and article text are stemmed (SnowballStemmer)
      - Uses word-boundary regex matching instead of substring .count()

    This catches morphological variants: plurals, tenses, -tion/-ment suffixes.
    e.g. 'decarbonize', 'decarbonization', 'decarbonizing' all match.

    Returns:
        (final_score, matched_keywords, keyword_counts)
        matched_keywords uses the original (un-stemmed) keyword forms for readability.
    """
    text = " ".join(
        v for v in [title_clean, summary, full_text_clean]
        if v and str(v).strip().lower() not in ("", "null", "nan")
    ).lower()

    if not text.strip():
        return 0.0, [], {}

    stemmed_text = _stem_text(text)

    keyword_counts: dict[str, int] = {}
    for stemmed_kw, pattern in _STEMMED_KW_PATTERNS.items():
        count = len(pattern.findall(stemmed_text))
        if count > 0:
            keyword_counts[_STEMMED_KW_MAP[stemmed_kw]] = count

    if not keyword_counts:
        return 0.0, [], {}

    matched = list(keyword_counts.keys())
    distinct_score      = len(matched)
    frequency_score     = sum(min(c, 10) for c in keyword_counts.values())
    diversity_multiplier = 1 + 0.1 * (distinct_score - 1)
    final_score = round((distinct_score + frequency_score) * diversity_multiplier, 2)

    return final_score, matched, keyword_counts

def filter_keywords(text, phrase, split=True):
    """Check if a phrase matches in text. split=True means all words must appear."""
    text = text.lower()
    if split:
        return all(kw in text for kw in phrase.split())
    else:
        return phrase in text

# ─── Prefilter Pydantic models ─────────────────────────────────────────────────

class PrefilterArticle(BaseModel):
    id: str
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    full_text: Optional[str] = None
    type: str  # 'g' = government, 'w' = wire, anything else = other


class PrefilterRequest(BaseModel):
    articles: List[PrefilterArticle]
    mkt_high_threshold: float = MKT_HIGH_THRESHOLD      # market-research classifier HIGH cutoff
    mkt_medium_threshold: float = MKT_MEDIUM_THRESHOLD  # market-research classifier MEDIUM cutoff
    zero_clf_threshold: float = ZERO_CLF_THRESH


class PrefilterFilteredItem(BaseModel):
    id: str
    filter_reason: str


class PrefilterResponse(BaseModel):
    passed: List[str]
    filtered: List[PrefilterFilteredItem]


# ─── Prefilter and Postfilter endpoint ────────────────────────────────────────────────────────

@app.post("/prefilter", response_model=PrefilterResponse)
def prefilter_articles(request: PrefilterRequest):
    """
    Pre-filter articles before keyword scoring.

    Preprocessing applied to every article before filtering:
      - URL percent-decoding (unquote)
      - Title cleaning  (HTML unescape, strip tags/whitespace)
      - Full-text cleaning (HTML → plain text, boilerplate removal)
      - Summary cleaning  (HTML unescape, strip tags/whitespace)
      - Domain extraction from decoded URL
      - Truncation detection (full_text ends with "..")

    Filters applied in order:
      1. Blocked domains
      2. Market-research URL classifier  (HIGH or MEDIUM confidence → removed)
      3. Market-research title classifier (HIGH or MEDIUM confidence → removed)
      4. Government-only noise: UPSC/IAS titles and about-us pages

    Input type field:
      - 'g'  → government source  (applies govt-specific noise filter)
      - 'w'  → wire/newswire source
      - anything else → other source

    Returns:
      - passed:   list of article ids that survived all filters
      - filtered: list of {id, filter_reason} for removed articles
    """
    from urllib.parse import unquote as _unquote

    passed: list[str] = []
    filtered: list[PrefilterFilteredItem] = []

    for article in request.articles:
        reason: str | None = None

        # ── Preprocessing ──────────────────────────────────────────────────────
        url_decoded     = _unquote(article.url) if article.url else ''
        title_clean_val = clean_title(article.title)               # HTML unescape + strip tags
        full_text_clean_val = clean_full_text(article.full_text)   # HTML → plain text + boilerplate removal
        summary_clean_val   = clean_summary(article.summary)       # HTML unescape + strip tags
        domain          = _extract_domain(url_decoded)             # extracted from decoded URL
        is_truncated    = str(article.full_text or '').rstrip().endswith('..')
        _ = (full_text_clean_val, summary_clean_val, is_truncated)  # preprocessed; available for downstream

        # Normalised lowercase versions used in filter checks
        title_lower = str(title_clean_val).lower() if title_clean_val and not pd.isna(title_clean_val) else ''
        url_lower   = url_decoded.lower()

        # ── Filter 1: Blocked domain ───────────────────────────────────────────
        if domain and domain in BLOCKED_DOMAINS:
            reason = f"blocked_domain({domain})"

        # 3b. Language filter — keep only English articles 
        if reason is None:
            lang = detect_lang(full_text_clean_val)
            if lang != 'en' and lang != 'unknown':
                reason = 'non_english'

        # ── Filter 2: Market-research URL classifier ───────────────────────────
        if reason is None:
            slug = _extract_slug(url_decoded)   # slug from already-decoded URL
            confidence, score = _classify_text(slug, request.mkt_high_threshold, request.mkt_medium_threshold)
            if confidence in ('HIGH', 'MEDIUM'):
                reason = f"mkt_research_url({confidence}) score={score} slug={slug[:60]}"

        # ── Filter 3: Market-research title classifier ─────────────────────────
        if reason is None:
            confidence, score = _classify_text(title_lower, request.mkt_high_threshold, request.mkt_medium_threshold)
            if confidence in ('HIGH', 'MEDIUM'):
                reason = f"mkt_research_title({confidence}) score={score}"

        # ── Filter 4: Government-specific noise (type == 'g' only) ────────────
        if reason is None and article.type == 'g':
            if (
                re.search(r'\b(upsc|ias)\b', title_lower)
                or 'about us' in title_lower
                or re.search(r'about[\-_]?us', url_lower)
            ):
                reason = "govt_noise(upsc/ias or about-us)"

        # # 5. Check financial/earnings noise in title and generative summary
        # if reason is None:
        #     combined_text = title_lower + ' ' + url_lower
        #     matched_fin_kw = [kw for kw in FINANCIAL_KEYWORDS if kw in combined_text]
        #     if matched_fin_kw:
        #         reason = "financial noise"

        # # 6. Check zero shot classifier output
        # if reason is None:
        #     combined_text = title_lower + ' ' + url_lower
        #     _ = classifier(combined_text, candidate_labels)
        #     zero_clf_label, zero_clf_score = _["labels"][0], _["scores"][0]
        #     if (zero_clf_label != "Renewables / Energy Projects / Carbon market/ clean fuel market/ offset market") and (zero_clf_score >= request.zero_clf_threshold):
        #         reason = f'zero shot classifier reject: {zero_clf_label}, {zero_clf_score}'
        
        # # 7. Keyword combination check
        # if reason is None:
        #     matched_content = None
        #     combined_text = title_lower + ' ' + url_lower
        #     for phrase in CONTENT_FILTER_PHRASES:
        #         if filter_keywords(combined_text, phrase, split=True):
        #             matched_content = phrase
        #             break
        #     if not matched_content:
        #         for phrase in CONTENT_FILTER_EXACT:
        #             if filter_keywords(combined_text, phrase, split=False):
        #                 matched_content = phrase
        #                 break
        #     if not matched_content:
        #         for phrase in URL_FILTER_PHRASES:
        #             if filter_keywords(url_lower, phrase, split=True):
        #                 matched_content = f'url:{phrase}'
        #                 break
        #     if matched_content:
        #         reason = f'phrase filter with matched content: {matched_content}'

        if reason:
            filtered.append(PrefilterFilteredItem(id=article.id, filter_reason=reason))
        else:
            passed.append(article.id)

    return PrefilterResponse(passed=passed, filtered=filtered)

@app.post("/postfilter", response_model=PrefilterResponse)
def postfilter_articles(request: PrefilterRequest):
    """
    Post filteration applied to articles after selection of potentially relevant ones
    Same format as prefiltered articles
    """
    from urllib.parse import unquote as _unquote

    passed: list[str] = []
    filtered: list[PrefilterFilteredItem] = []

    for article in request.articles:
        reason: str | None = None

        # ── Preprocessing ──────────────────────────────────────────────────────
        url_decoded     = _unquote(article.url) if article.url else ''
        title_clean_val = clean_title(article.title)               # HTML unescape + strip tags
        full_text_clean_val = clean_full_text(article.full_text)   # HTML → plain text + boilerplate removal
        summary_clean_val   = clean_summary(article.summary)       # HTML unescape + strip tags
        domain          = _extract_domain(url_decoded)             # extracted from decoded URL
        is_truncated    = str(article.full_text or '').rstrip().endswith('..')
        _ = (full_text_clean_val, summary_clean_val, is_truncated)  # preprocessed; available for downstream

        # Normalised lowercase versions used in filter checks
        title_lower = str(title_clean_val).lower() if title_clean_val and not pd.isna(title_clean_val) else ''
        url_lower   = url_decoded.lower()

        # 5. Check financial/earnings noise in title and generative summary
        if reason is None:
            combined_text = title_lower + ' ' + url_lower
            matched_fin_kw = [kw for kw in FINANCIAL_KEYWORDS if kw in combined_text]
            if matched_fin_kw:
                reason = "financial noise"

        # 6. Check zero shot classifier output
        if reason is None and article.type != 'g':
            combined_text = title_lower + ' ' + url_lower
            _ = classifier(combined_text, candidate_labels)
            zero_clf_label, zero_clf_score = _["labels"][0], _["scores"][0]
            if (zero_clf_label != "Renewables / Energy Projects / Carbon market/ clean fuel market/ offset market") and (zero_clf_score >= request.zero_clf_threshold):
                reason = f'zero shot classifier reject: {zero_clf_label}, {zero_clf_score}'
        
        # 7. Keyword combination check
        if reason is None:
            matched_content = None
            combined_text = title_lower + ' ' + url_lower
            for phrase in CONTENT_FILTER_PHRASES:
                if filter_keywords(combined_text, phrase, split=True):
                    matched_content = phrase
                    break
            if not matched_content:
                for phrase in CONTENT_FILTER_EXACT:
                    if filter_keywords(combined_text, phrase, split=False):
                        matched_content = phrase
                        break
            if not matched_content:
                for phrase in URL_FILTER_PHRASES:
                    if filter_keywords(url_lower, phrase, split=True):
                        matched_content = f'url:{phrase}'
                        break
            if matched_content:
                reason = f'phrase filter with matched content: {matched_content}'

        if reason:
            filtered.append(PrefilterFilteredItem(id=article.id, filter_reason=reason))
        else:
            passed.append(article.id)

    return PrefilterResponse(passed=passed, filtered=filtered)

# ─── Score endpoint models ─────────────────────────────────────────────────────

class ScoreArticle(BaseModel):
    id: str
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    full_text: Optional[str] = None
    type: str  # 'g' = government, 'w' = wire, anything else = other


class ScoreRequest(BaseModel):
    articles: List[ScoreArticle]
    govt_threshold: float = 5.0    # minimum kw_score for government sources
    wire_threshold: float = 50.0   # minimum kw_score for wire/newswire sources
    other_threshold: float = 5.0   # minimum kw_score for all other sources


class ScoredArticle(BaseModel):
    id: str
    kw_score: float
    kw_hits: List[str]
    source_type: str   # government / wire / other


class BelowThresholdArticle(BaseModel):
    id: str
    kw_score: float
    threshold_used: float
    source_type: str


class ScoreResponse(BaseModel):
    passed: List[ScoredArticle]
    filtered: List[BelowThresholdArticle]


# ─── Score endpoint ────────────────────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse)
def score_articles(request: ScoreRequest):
    """
    Keyword-score articles and return those exceeding their source-type threshold.

    Preprocessing applied before scoring (same as /prefilter):
      - URL percent-decoding
      - Title cleaning  (HTML unescape, strip tags/whitespace)
      - Full-text cleaning (HTML → plain text, boilerplate removal)
      - Summary cleaning

    Scoring formula (from notebook):
      distinct_score       = number of unique keyword matches
      frequency_score      = sum of per-keyword counts (capped at 10 each)
      diversity_multiplier = 1 + 0.1 * (distinct_score - 1)
      final_score          = (distinct_score + frequency_score) * diversity_multiplier

    Thresholds (all configurable):
      - govt_threshold  (default 5)  — applied to type == 'g'
      - wire_threshold  (default 50) — applied to type == 'w'
      - other_threshold (default 5)  — applied to everything else

    Returns:
      - passed:   articles with kw_score > threshold, with score and matched keywords
      - filtered: articles that fell below threshold, with their score and threshold used
    """
    from urllib.parse import unquote as _unquote

    passed: list[ScoredArticle] = []
    filtered: list[BelowThresholdArticle] = []

    for article in request.articles:
        # ── Preprocessing ──────────────────────────────────────────────────────
        url_decoded         = _unquote(article.url) if article.url else ''
        title_clean_val     = clean_title(article.title)
        full_text_clean_val = clean_full_text(article.full_text)
        summary_clean_val   = clean_summary(article.summary)

        title_str     = str(title_clean_val)     if title_clean_val     and not pd.isna(title_clean_val)     else ''
        full_text_str = str(full_text_clean_val) if full_text_clean_val and not pd.isna(full_text_clean_val) else ''
        summary_str   = str(summary_clean_val)   if summary_clean_val   and not pd.isna(summary_clean_val)   else ''

        _ = url_decoded  # decoded URL available for future use

        # ── Keyword scoring ────────────────────────────────────────────────────
        kw_score, kw_hits, _ = keyword_score(title_str, summary_str, full_text_str)

        # ── Source type + threshold selection ──────────────────────────────────
        if article.type == 'g':
            source_type = 'government'
            threshold = request.govt_threshold
        elif article.type == 'w':
            source_type = 'wire'
            threshold = request.wire_threshold
        else:
            source_type = 'other'
            threshold = request.other_threshold

        # ── Apply threshold ────────────────────────────────────────────────────
        if kw_score > threshold:
            passed.append(ScoredArticle(
                id=article.id,
                kw_score=kw_score,
                kw_hits=kw_hits,
                source_type=source_type,
            ))
        else:
            filtered.append(BelowThresholdArticle(
                id=article.id,
                kw_score=kw_score,
                threshold_used=threshold,
                source_type=source_type,
            ))

    return ScoreResponse(passed=passed, filtered=filtered)


# ─── Score V2 endpoint (stemmed matching) ─────────────────────────────────────

@app.post("/score-v2", response_model=ScoreResponse)
def score_articles_v2(request: ScoreRequest):
    """
    Keyword-score articles using stemmed word-boundary matching (v2).

    Same interface and thresholds as /score, but uses SnowballStemmer on both
    keywords and article text before matching. This catches morphological
    variants (plurals, tenses, -tion/-ment suffixes) that /score misses.

    Examples of additional matches vs /score:
      - 'decarbonize' matches 'decarbonization', 'decarbonizing'
      - 'renewable' matches 'renewables'
      - 'barrel' matches 'barrels'
      - 'emission' matches 'emissions'

    Also uses word-boundary matching (\\b) to avoid false substring hits
    like 'oil' in 'soil' or 'gas' in 'gasoline'.

    Includes capacity/unit keywords: kilowatt, terawatt, kwh, mwh, gwh, twh,
    mmbtu, barrel, bpd, bcm, mtpa, installed capacity, generation capacity,
    production capacity, refining capacity, capacity addition.
    """
    from urllib.parse import unquote as _unquote

    passed: list[ScoredArticle] = []
    filtered: list[BelowThresholdArticle] = []

    for article in request.articles:
        # ── Preprocessing ──────────────────────────────────────────────────────
        url_decoded         = _unquote(article.url) if article.url else ''
        title_clean_val     = clean_title(article.title)
        full_text_clean_val = clean_full_text(article.full_text)
        summary_clean_val   = clean_summary(article.summary)

        title_str     = str(title_clean_val)     if title_clean_val     and not pd.isna(title_clean_val)     else ''
        full_text_str = str(full_text_clean_val) if full_text_clean_val and not pd.isna(full_text_clean_val) else ''
        summary_str   = str(summary_clean_val)   if summary_clean_val   and not pd.isna(summary_clean_val)   else ''

        _ = url_decoded  # decoded URL available for future use

        # ── Stemmed keyword scoring ───────────────────────────────────────────
        kw_score, kw_hits, _ = keyword_score_v2(title_str, summary_str, full_text_str)

        # ── Source type + threshold selection ──────────────────────────────────
        if article.type == 'g':
            source_type = 'government'
            threshold = request.govt_threshold
        elif article.type == 'w':
            source_type = 'wire'
            threshold = request.wire_threshold
        else:
            source_type = 'other'
            threshold = request.other_threshold

        # ── Apply threshold ────────────────────────────────────────────────────
        if kw_score > threshold:
            passed.append(ScoredArticle(
                id=article.id,
                kw_score=kw_score,
                kw_hits=kw_hits,
                source_type=source_type,
            ))
        else:
            filtered.append(BelowThresholdArticle(
                id=article.id,
                kw_score=kw_score,
                threshold_used=threshold,
                source_type=source_type,
            ))

    return ScoreResponse(passed=passed, filtered=filtered)


# ─── URL MinHash LSH Deduplication ─────────────────────────────────────────────

class URLDeduplicateArticle(BaseModel):
    id: Union[str, int]
    url: str


class URLDeduplicateRequest(BaseModel):
    articles: List[URLDeduplicateArticle]
    threshold: Optional[float] = 0.6


class DuplicateGroup(BaseModel):
    ids: List[str]
    urls: List[str]
    jaccard: float


class URLDeduplicateResponse(BaseModel):
    unique_ids: List[str]
    duplicate_groups: List[DuplicateGroup]
    total_input: int
    total_unique: int


@app.post("/deduplicate_url", response_model=URLDeduplicateResponse)
def deduplicate_url(request: URLDeduplicateRequest):
    """
    Deduplicate articles based on URL slug similarity using MinHash LSH.

    Extracts the last path segment (slug) from each URL, builds MinHash
    signatures from character 3-grams, and finds near-duplicate pairs
    using Locality-Sensitive Hashing.

    Articles with short/empty slugs (< 10 chars) are skipped and always
    included in unique_ids.

    Args:
        articles: list of {id, url}
        threshold: MinHash Jaccard threshold (default 0.6)

    Returns:
        unique_ids: deduplicated list of IDs (one kept per duplicate group)
        duplicate_groups: groups of duplicate articles with their Jaccard scores
        total_input: number of articles received
        total_unique: number of unique articles after deduplication
    """
    if not request.articles:
        raise HTTPException(status_code=400, detail="No articles provided")

    threshold = request.threshold
    articles = request.articles

    # Extract slugs and build minhashes
    slugs: list[str] = []
    minhashes: dict[int, MinHash] = {}
    for i, art in enumerate(articles):
        slug = _extract_slug_for_lsh(art.url)
        slugs.append(slug)
        if len(slug) >= 10:
            mh = _make_minhash(slug)
            if mh is not None:
                minhashes[i] = mh

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    for idx, mh in minhashes.items():
        try:
            lsh.insert(str(idx), mh)
        except ValueError:
            pass  # duplicate minhash key

    # Find candidate pairs and compute exact Jaccard
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    pair_scores: dict[tuple[int, int], float] = {}

    for i, mh in minhashes.items():
        for candidate in lsh.query(mh):
            j = int(candidate)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                jaccard = minhashes[key[0]].jaccard(minhashes[key[1]])
                if jaccard >= threshold:
                    edges.append(key)
                    pair_scores[key] = jaccard

    # Build connected components for duplicate groups
    G = nx.Graph()
    G.add_edges_from(edges)

    duplicate_groups: list[DuplicateGroup] = []
    duplicated_indices: set[int] = set()

    for component in nx.connected_components(G):
        group = sorted(component)  # keep input order (lowest index first)
        # Max Jaccard among all pairs in the group
        max_jac = 0.0
        for a in group:
            for b in group:
                if a < b:
                    max_jac = max(max_jac, pair_scores.get((a, b), 0.0))

        duplicate_groups.append(DuplicateGroup(
            ids=[str(articles[i].id) for i in group],
            urls=[articles[i].url for i in group],
            jaccard=round(max_jac, 4),
        ))
        # Keep first in group, mark rest as duplicates
        for idx in group[1:]:
            duplicated_indices.add(idx)

    unique_ids = [str(art.id) for i, art in enumerate(articles) if i not in duplicated_indices]

    return URLDeduplicateResponse(
        unique_ids=unique_ids,
        duplicate_groups=duplicate_groups,
        total_input=len(articles),
        total_unique=len(unique_ids),
    )

class SVMArticle(BaseModel):
    id: str
    embeddings: List[float]


class SVMPredictRequest(BaseModel):
    articles: List[SVMArticle]
    svm_threshold: Optional[float] = 0.2

@app.post("/predict_svm")
def predict_svm(request: SVMPredictRequest):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="No articles provided")

        ids = []
        embeddings = []

        for art in request.articles:
            ids.append(art.id)
            embeddings.append(art.embeddings)

        X_new = np.array(embeddings, dtype=np.float32)

        probs = svm_model.predict_proba(X_new)[:, 1]
        preds = (probs >= request.svm_threshold).astype(int)

        positive_ids = []
        negative_ids = []

        for i, pred in enumerate(preds):
            if pred == 1:
                positive_ids.append(ids[i])
            else:
                negative_ids.append(ids[i])

        return {
            "positive_ids": positive_ids,
            "negative_ids": negative_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MarketTaggerItem(BaseModel):
    id: str
    text: str


class MarketTaggerRequest(BaseModel):
    articles: List[MarketTaggerItem]


class MarketTaggerResponseItem(BaseModel):
    id: str
    class_label: int


class MarketTaggerResponse(BaseModel):
    results: List[MarketTaggerResponseItem]



# Market Tagger Endpoint
@app.post("/market_tagger", response_model=MarketTaggerResponse)
def market_tagger(request: MarketTaggerRequest):

    ids = [item.id for item in request.articles]
    texts = [item.text for item in request.articles]

    preds = market_tagger_model.predict(texts)

    results = [
        MarketTaggerResponseItem(
            id=i,
            class_label=int(p)
        )
        for i, p in zip(ids, preds)
    ]

    return MarketTaggerResponse(results=results)


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "article-extractor"}
