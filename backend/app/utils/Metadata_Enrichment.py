"""
Stage 3 — Metadata Enrichment (Redis-style)

- Adds metadata to each chunk
- Detects type: code | prose | table
- Attaches embedding + IDs + structure info
- Ready for vector DB (Redis, FAISS, etc.)
"""

import uuid
import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Type Detection
# -----------------------------
PROTOTYPES = {
    "code": [
        "def function(): return value",
        "for i in range(10): print(i)",
        "public static void main(String[] args)",
        "if (x > 0) { return x; }"
    ],
    "table": [
        "Name | Age | City",
        "Column1 | Column2 | Column3",
        "A | B | C",
        "Row1: value1, value2, value3"
    ],
    "prose": [
        "This section explains the methodology used in the experiment.",
        "The results indicate a strong correlation between variables.",
        "In conclusion, the approach improves performance."
    ]
}
# Precompute prototype embeddings
PROTO_EMB = {
    k: model.encode(v) for k, v in PROTOTYPES.items()
}


# -----------------------------
# Structural Score
# -----------------------------
def structural_score(text: str) -> Dict[str, float]:
    scores = {"code": 0.0, "table": 0.0, "prose": 0.0}

    # Code signals
    if re.search(r'\b(def |class |#include|public |function |\{|\}|=>)', text):
        scores["code"] += 0.6
    if re.search(r'[;{}()]', text):
        scores["code"] += 0.3

    # Table signals
    if "|" in text:
        scores["table"] += 0.6
    if re.search(r'\n.*\|.*\n.*\|', text):
        scores["table"] += 0.4

    # Prose signals (natural language density)
    word_count = len(text.split())
    if word_count > 20:
        scores["prose"] += 0.5

    return scores


# -----------------------------
# Embedding Score
# -----------------------------
def embedding_score(text: str) -> Dict[str, float]:
    emb = model.encode(text)
    scores = {}

    for k, proto_embs in PROTO_EMB.items():
        emb_norm = np.linalg.norm(emb)
        proto_norms = np.linalg.norm(proto_embs, axis=1)
        # Cosine similarity between chunk embedding and each prototype embedding.
        sims = (proto_embs @ emb) / (proto_norms * emb_norm + 1e-12)
        scores[k] = float(np.max(sims))  # best match

    return scores


# -----------------------------
# Final Classifier
# -----------------------------
def detect_type_advanced(text: str) -> Dict:
    """
    Returns:
    {
        "type": "code/table/prose",
        "confidence": float,
        "scores": {...}
    }
    """

    s_score = structural_score(text)
    e_score = embedding_score(text)

    final_scores = {}

    for k in ["code", "table", "prose"]:
        # weighted fusion
        final_scores[k] = 0.5 * s_score[k] + 0.5 * e_score[k]

    # pick best
    best_type = max(final_scores, key=final_scores.get)
    confidence = final_scores[best_type]

    return {
        "type": best_type,
        "confidence": round(confidence, 3),
        "scores": final_scores
    }

# def detect_type(text: str) -> str:
#     """
#     Detects if chunk is code, table, or prose
#     """
#     # Code detection (simple heuristic)
#     if re.search(r'\b(def |class |#include|public static|function |\{|\})', text):
#         return "code"

#     # Table detection (markdown or structured)
#     if "|" in text and "-" in text:
#         return "table"

#     return "prose"


# -----------------------------
# Section Extraction (from headings)
# -----------------------------
def extract_section(text: str) -> str:
    """
    Extracts section title if present
    """
    match = re.search(r'#{1,6}\s+(.*)', text)
    if match:
        return match.group(1).strip()
    return "unknown"


# -----------------------------
# Metadata Enrichment
# -----------------------------
def enrich_chunks(
    chunks: List[str],
    source: str = "pdf_1"
) -> List[Dict]:
    """
    Adds metadata + embeddings
    """

    enriched = []

    # Generate embeddings in batch (faster)
    embeddings = model.encode(chunks)

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        metadata = {
            "chunk_id": chunk_id,
            "source": source,
            "section": extract_section(chunk),
            "type": detect_type_advanced(chunk),
            "embedding": embeddings[i].tolist(),
            "text": chunk
        }

        enriched.append(metadata)

    return enriched


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    refined_chunks = [
        "# Methodology\nWe use machine learning models to analyze data.",
        "def add(a, b): return a + b",
        "| Name | Age |\n|------|-----|\n| John | 25 |"
    ]

    enriched = enrich_chunks(refined_chunks, source="pdf_1")

    for item in enriched:
        print("\n--- CHUNK ---")
        for k, v in item.items():
            if k == "embedding":
                print(f"{k}: [vector of length {len(v)}]")
            else:
                print(f"{k}: {v}")