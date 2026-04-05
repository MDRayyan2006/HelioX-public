"""approx taking 5-10s per 1000 chunks on CPU, can be reduced to 1-2s with GPU and batch processing"""
"""need to be optimized for large documents, currently it can be slow due to the semantic similarity calculations and embedding generation, we can optimize it by using batch processing for embeddings and parallel processing for similarity calculations"""

# -----------------------------
# IMPORT STAGES
# -----------------------------
from recursive_chunking import build_chunks
from semantic_refinement import semantic_refine_chunks
from Metadata_Enrichment import enrich_chunks
from text_processing import load_folder
import os

# -----------------------------
# PIPELINE FUNCTION
# -----------------------------
def run_pipeline(
    text: str,
    source: str = "pdf_1",
    max_tokens: int = 500,
    overlap_ratio: float = 0.12,
    similarity_threshold: float = 0.65
):
    """
    Full RAG Chunking Pipeline

    Steps:
    1. Structural Chunking
    2. Semantic Refinement
    3. Metadata Enrichment
    """

    # Stage 1 — Structural
    # stage1_chunks = build_chunks(
    #     text,
    #     max_tokens=max_tokens,
    #     overlap_ratio=overlap_ratio
    # )/
    stage1_chunks = build_chunks(text)

    # Stage 2 — Semantic
    stage2_chunks = semantic_refine_chunks(
        stage1_chunks,
        similarity_threshold=similarity_threshold,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio
    )

    # Stage 3 — Metadata
    final_chunks = enrich_chunks(
        stage2_chunks,
        source=source
    )

    return final_chunks


# -----------------------------
# OPTIONAL DEBUG PIPELINE
# -----------------------------
def debug_pipeline(text: str):
    """
    Returns intermediate outputs (very useful for tuning)
    """

    stage1 = build_chunks(text)
    stage2 = semantic_refine_chunks(stage1)
    stage3 = enrich_chunks(stage2)

    return {
        "stage1_chunks": stage1,
        "stage2_chunks": stage2,
        "final_chunks": stage3
    }



# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
#     sample_text = """
# # Methodology
# We use machine learning models to analyze data. The approach improves performance.

# def add(a, b): return a + b

# | Name | Age |
# |------|-----|
# | John | 25 |
# """

#     results = run_pipeline(sample_text)

#     for r in results:
#         print("\n--- FINAL CHUNK ---")
#         for k, v in r.items():
#             if k == "embedding":
#                 print(f"{k}: [vector length {len(v)}]")
#             else:
#                 print(f"{k}: {v}")
    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
    docs = load_folder(folder_path)
    for doc in docs:
        loadr_result=run_pipeline(doc['text'], source=doc['metadata']['file_name'])
        print(f"\n--- Processed {doc['metadata']['file_name']} ---")
        print(f"Generated {len(loadr_result)} chunks with metadata enrichment.")
        for i in range(min(3, len(loadr_result))):  # Print first 3 chunks as a sample
            print(f"Sample Chunk:\n{loadr_result[i]['text']}")  # Print first 200 chars of each chunk

    # for doc in docs:
    #     print(f"\n--- Processing {doc['metadata']['file_name']} ---")
    #     loader_results = debug_pipeline(doc['text'])

    #     print(f"Stage 1 Chunks: {loader_results['stage1_chunks']}")
    #     print(f"Stage 2 Chunks: {loader_results['stage2_chunks']}")
    #     print(f"Final Chunks: {loader_results['final_chunks']}")


    

    # loader_results = debug_pipeline(sample_text)