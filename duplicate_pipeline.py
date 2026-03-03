import sys
import re
import html as html_lib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


# ── Preprocessing ────────────────────────────────────────────────────────────

def clean_title(raw) -> str | float:
    """
    - Literal "null" string      → NaN
    - HTML entities (&amp; etc.) → unescaped
    - HTML tags (<b>, <i> etc.)  → stripped
    - Whitespace                 → normalised
    """
    if pd.isna(raw) or str(raw).strip().lower() == "null":
        return float("nan")
    t = html_lib.unescape(str(raw))
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else float("nan")


def clean_full_text(raw) -> str | float:
    """
    - Literal "null" string           → NaN
    - HTML entities                   → unescaped
    - HTML / full <html> documents    → parsed by BeautifulSoup, plain text extracted
    - Handlebars templates {{...}}    → removed
    - Agency bylines (–IANS, - PTI)   → removed
    - "(With inputs from agencies.)"  → removed
    - Newsletter CTAs (Sign up: ...)  → removed
    - Bare URLs                       → removed
    - Excessive whitespace            → normalised
    """
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
    """
    Applies notebook-identical cleaning to title and full_text.
    Adds columns: title_clean, full_text_clean, is_truncated.
    """
    print("Cleaning titles...")
    df["title_clean"] = df["title"].apply(clean_title)

    print("Cleaning full_text...")
    df["full_text_clean"] = df["full_text"].apply(clean_full_text)

    df["is_truncated"] = (
        df["full_text"]
        .astype(str)
        .str.rstrip()
        .str.endswith("..")
    )

    n = len(df)
    print(f"  title_clean     non-null : {df['title_clean'].notna().sum():,} / {n:,}")
    print(f"  full_text_clean non-null : {df['full_text_clean'].notna().sum():,} / {n:,}")
    print(f"  Truncated articles       : {df['is_truncated'].sum():,}")
    return df


# ── Embedding Input Construction ──────────────────────────────────────────────

def build_embedding_inputs(df: pd.DataFrame) -> list[str]:
    """
    Uses cleaned columns (title_clean, full_text_clean) produced by preprocess_dataframe.

    - summary exists            → title_clean + " " + summary
    - only full_text exists     → full_text_clean
    - neither exists (fallback) → title_clean
    """
    inputs = []
    for _, row in df.iterrows():
        summary   = row.get("summary")
        full_text = row.get("full_text_clean")
        title     = row.get("title_clean", "")

        has_summary   = pd.notna(summary)   and str(summary).strip()   != ""
        has_full_text = pd.notna(full_text) and str(full_text).strip() != ""
        title_str     = str(title).strip() if pd.notna(title) else ""

        if has_summary:
            inputs.append(f"{title_str} {str(summary).strip()}".strip())
        elif has_full_text:
            inputs.append(str(full_text).strip())
        else:
            inputs.append(title_str)

    return inputs


# ── Embeddings ────────────────────────────────────────────────────────────────

def generate_embeddings(texts: list[str]) -> np.ndarray:
    print(f"Encoding {len(texts):,} texts with '{MODEL_NAME}'...")
    # Qwen3-Embedding requires left-padding for correct last-token pooling
    model = SentenceTransformer(                                                                                                                                                                          
        MODEL_NAME,                                                                                                                                                                                         
        tokenizer_kwargs={"padding_side": "left"},                                                                                                                                                          
        device="mps",   # uses Apple GPU instead of CPU                                                                                                                                                   
    )                                
    embeddings = model.encode(
        texts,
        batch_size=16,               # 0.6B model is larger; smaller batch avoids OOM
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm → cosine sim = dot product
    )
    return embeddings.astype(np.float32)


# ── Similarity Matrix ─────────────────────────────────────────────────────────

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Vectorised cosine similarity via dot product (embeddings are L2-normalised)."""
    sim = np.dot(embeddings, embeddings.T)
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


# ── Graph-Based Clustering ────────────────────────────────────────────────────

def cluster_at_threshold(sim_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build undirected graph: edge (i,j) if sim >= threshold.
    Assign cluster_id = connected-component index.
    """
    n = sim_matrix.shape[0]
    rows, cols = np.where(
        np.triu(sim_matrix >= threshold, k=1)
    )

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))

    cluster_ids = np.empty(n, dtype=int)
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            cluster_ids[node] = cid

    return cluster_ids


# ── Ground Truth Pairs ────────────────────────────────────────────────────────

def build_pair_ground_truth(new_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (i_idx, j_idx, y_true) for all upper-triangle pairs.
    y_true = 1  iff  new_id[i] == new_id[j]  AND  new_id[i] != 0
    """
    i_idx, j_idx = np.triu_indices(len(new_ids), k=1)
    ni, nj = new_ids[i_idx], new_ids[j_idx]
    y_true = ((ni == nj) & (ni != 0)).astype(np.int8)
    return i_idx, j_idx, y_true


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    cluster_ids: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    y_pred = (cluster_ids[i_idx] == cluster_ids[j_idx]).astype(np.int8)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy":    acc,
        "precision_0": prec[0],
        "recall_0":    rec[0],
        "precision_1": prec[1],
        "recall_1":    rec[1],
        "f1_1":        f1[1],
        "confusion_matrix": cm,
    }


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, threshold: float, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    tick_labels = ["Not Duplicate (0)", "Duplicate (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    mid = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > mid else "black",
                fontsize=12,
            )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — threshold = {threshold}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    col_w = 12
    headers = ["Threshold", "Accuracy", "Prec(0)", "Rec(0)", "Prec(1)", "Rec(1)", "F1(1)"]
    sep = "=" * (col_w * len(headers))
    row_fmt = "".join(f"{h:>{col_w}}" for h in headers)

    print(f"\n{sep}")
    print("SEMANTIC DUPLICATE CLUSTERING — EVALUATION SUMMARY")
    print(sep)
    print(row_fmt)
    print("-" * (col_w * len(headers)))
    for r in results:
        print(
            f"{r['threshold']:>{col_w}.1f}"
            f"{r['accuracy']:>{col_w}.4f}"
            f"{r['precision_0']:>{col_w}.4f}"
            f"{r['recall_0']:>{col_w}.4f}"
            f"{r['precision_1']:>{col_w}.4f}"
            f"{r['recall_1']:>{col_w}.4f}"
            f"{r['f1_1']:>{col_w}.4f}"
        )
    print(sep)


def save_metrics_csv(results: list[dict], output_path: str) -> None:
    rows = [
        {
            "threshold":   r["threshold"],
            "accuracy":    r["accuracy"],
            "precision_0": r["precision_0"],
            "recall_0":    r["recall_0"],
            "precision_1": r["precision_1"],
            "recall_1":    r["recall_1"],
            "f1_1":        r["f1_1"],
        }
        for r in results
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nMetrics saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python duplicate_pipeline.py input.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # 1. Load
    df = load_data(csv_path)

    # 2. Preprocess title and full_text (mirrors notebook cleaning)
    print("Preprocessing data...")
    df = preprocess_dataframe(df)

    # 3. Build embedding inputs from cleaned columns
    print("Building embedding inputs...")
    texts = build_embedding_inputs(df)

    # 4. Embed
    embeddings = generate_embeddings(texts)

    # 5. Similarity matrix (computed once, reused for all thresholds)
    print("Computing cosine similarity matrix...")
    sim_matrix = compute_similarity_matrix(embeddings)

    # 6. Ground truth
    print("Building pair-level ground truth...")
    new_ids = df["new_id"].fillna(0).astype(int).values
    i_idx, j_idx, y_true = build_pair_ground_truth(new_ids)
    n_pairs    = len(y_true)
    n_positive = int(y_true.sum())
    print(f"  Total pairs : {n_pairs:,}")
    print(f"  Positive (duplicate) pairs : {n_positive:,}")
    print(f"  Negative pairs : {n_pairs - n_positive:,}")

    # 7. Cluster + evaluate at each threshold
    results = []
    for threshold in THRESHOLDS:
        print(f"\n[Threshold = {threshold}]")
        cluster_ids   = cluster_at_threshold(sim_matrix, threshold)
        n_clusters    = len(np.unique(cluster_ids))
        print(f"  Connected components (clusters): {n_clusters:,}")

        metrics = evaluate(cluster_ids, i_idx, j_idx, y_true)
        metrics["threshold"] = threshold
        results.append(metrics)

        cm_path = f"confusion_matrix_{threshold}.png"
        plot_confusion_matrix(metrics["confusion_matrix"], threshold, cm_path)

    # 8. Report
    print_summary_table(results)
    save_metrics_csv(results, "threshold_metrics.csv")


if __name__ == "__main__":
    main()
