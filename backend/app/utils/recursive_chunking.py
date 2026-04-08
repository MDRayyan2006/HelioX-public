# # def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
# #     from langchain_text_splitters import RecursiveCharacterTextSplitter
    
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=chunk_size,
# #         chunk_overlap=chunk_overlap,
# #         length_function=len,
# #         separators=["\n\n", "\n", " ", ""]
# #     )
# #     chunks = splitter.split_documents(documents)
# #     print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
# #     return chunks


# # if __name__ == "__main__":
# #     from text_processing import load_folder
# #     from langchain_core.documents import Document
# #     import os
    
# #     folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
# #     if os.path.exists(folder_path):
# #         docs = load_folder(folder_path)
# #         if docs:
# #             print(f"\n\n=== Testing with PDF Files ({len(docs)} documents) ===")
            
# #             # Convert to LangChain Documents
# #             langchain_docs = []
# #             for doc in docs:
# #                 langchain_doc = Document(
# #                     page_content=doc['text'],
# #                     metadata=doc['metadata']
# #                 )
# #                 langchain_docs.append(langchain_doc)
            
# #             # Chunk the documents
# #             chunks = chunk_documents(langchain_docs, chunk_size=500, chunk_overlap=100)
            
# #             print(f"\nGenerated {len(chunks)} total chunks")
            
# #             # Display sample chunks
# #             if chunks:
# #                 num_samples = min(3, len(chunks))  # Show first 3 chunks
# #                 for i in range(num_samples):
# #                     print(f"\n--- Sample Chunk {i + 1} ---")
# #                     print(f"Content: {chunks[i].page_content[:200]}...")
# #                     print(f"Metadata: {chunks[i].metadata}")
# #     else:
# #         print(f"Uploads folder not found at: {folder_path}")

# import os
# import re
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from concurrent.futures import ProcessPoolExecutor


# def clean_text(text: str) -> str:
#     if not text:
#         return ""
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     text = re.sub(r'[ \t]+', ' ', text)
#     text = re.sub(r' \n', '\n', text)
#     return text.strip()


# def _chunk_worker(args):
#     page_content, metadata, doc_id, chunk_size, chunk_overlap, max_chars = args

#     text = clean_text(page_content)
#     if len(text) > max_chars:
#         text = text[:max_chars]

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     result = []
#     for i, content in enumerate(splitter.split_text(text)):
#         content = content.strip()
#         if len(content) < 50:
#             continue
#         result.append({
#             "page_content": content,
#             "metadata": {
#                 **metadata,
#                 "doc_id": doc_id,
#                 "chunk_id": i,
#                 "chunk_global_id": f"{doc_id}_{i}",
#             }
#         })
#     return result


# def _normalize(doc, idx, chunk_size, chunk_overlap, max_chars):
#     if isinstance(doc, dict):
#         meta = doc.get("metadata", {})
#         return (
#             doc.get("text", ""),
#             {
#                 "file_name": meta.get("file_name", "unknown"),
#                 "file_path": meta.get("file_path", ""),
#                 "num_pages": meta.get("num_pages", 1),
#                 "file_size": meta.get("file_size", 0),
#                 "source":    meta.get("source", "unknown"),
#             },
#             idx, chunk_size, chunk_overlap, max_chars
#         )
#     else:
#         return (doc.page_content, doc.metadata, idx, chunk_size, chunk_overlap, max_chars)


# def build_chunks(
#     documents,
#     chunk_size=500,
#     chunk_overlap=100,
#     max_chars=100_000,
#     max_workers=None,
#     parallel_threshold=20   # ← only use multiprocessing above this count
# ):
#     args_list = [
#         _normalize(doc, idx, chunk_size, chunk_overlap, max_chars)
#         for idx, doc in enumerate(documents)
#     ]

#     all_chunks = []

#     if len(documents) >= parallel_threshold:
#         # ── Many docs: true CPU parallelism worth the spawn overhead
#         print(f"[INFO] Parallel mode ({len(documents)} docs)")
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             for batch in executor.map(_chunk_worker, args_list, chunksize=8):
#                 for item in batch:
#                     all_chunks.append(Document(
#                         page_content=item["page_content"],
#                         metadata=item["metadata"]
#                     ))
#     else:
#         # ── Few docs: plain loop is faster (no process spawn cost)
#         print(f"[INFO] Single-thread mode ({len(documents)} docs)")
#         for args in args_list:
#             for item in _chunk_worker(args):
#                 all_chunks.append(Document(
#                     page_content=item["page_content"],
#                     metadata=item["metadata"]
#                 ))

#     print(f"[INFO] {len(args_list)} docs → {len(all_chunks)} chunks")
#     return all_chunks


# if __name__ == "__main__":
#     import time
#     from text_processing import load_folder

#     folder_path = os.path.join(os.path.dirname(__file__), "../uploads")

#     if not os.path.exists(folder_path):
#         print(f"Folder not found: {folder_path}")
#     else:
#         start = time.perf_counter()
#         docs = load_folder(folder_path)
#         end = time.perf_counter()
#         print(f"Loading documents completed in {end - start:.2f} seconds")
#         if not docs:
#             print("No documents found.")
#         else:
#             print(f"\n=== Processing {len(docs)} document(s) ===")
#             start = time.perf_counter()

#             chunks = build_chunks(
#                 docs,
#                 chunk_size=500,
#                 chunk_overlap=100,
#                 max_workers=os.cpu_count()
#             )

#             print(f"Chunking completed in {time.perf_counter() - start:.2f} seconds")
#             print(f"Generated {len(chunks)} chunks\n")

#             for i in range(len(chunks)):
#                 print(f"--- Chunk {i + 1} ---")
#                 print(f"Content : {chunks[i].page_content[:200]}...")
#                 print(f"Metadata: {chunks[i].metadata}\n")

import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ProcessPoolExecutor


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)
    return text.strip()


# ── Document type detection ──────────────────────────────────────
def detect_doc_type(text: str) -> str:
    """Detect document structure to choose best chunking strategy."""
    lines = text[:2000].split('\n')
    
    # Markdown / structured headers
    header_lines = [l for l in lines if re.match(r'^#{1,4}\s+\w+', l)]
    if len(header_lines) >= 2:
        return "markdown"
    
    # Research paper signals
    paper_signals = ['abstract', 'introduction', 'methodology',
                     'conclusion', 'references', 'related work']
    upper_lines = [l.strip().lower() for l in lines if l.strip().isupper() or
                   re.match(r'^\d+\.\s+[A-Z]', l.strip())]
    if sum(1 for s in paper_signals if any(s in l for l in upper_lines)) >= 2:
        return "research_paper"
    
    # Legal document signals
    legal_signals = ['whereas', 'hereby', 'pursuant', 'notwithstanding',
                     'article', 'section', 'clause']
    text_lower = text[:3000].lower()
    if sum(1 for s in legal_signals if s in text_lower) >= 3:
        return "legal"
    
    # Code file
    code_signals = ['def ', 'class ', 'function ', 'import ', '#!/']
    if sum(1 for s in code_signals if s in text[:1000]) >= 2:
        return "code"
    
    return "general"  # fallback


# ── Strategy: header-based (markdown / structured docs) ──────────
def chunk_by_headers(text: str, metadata: dict, doc_id: int) -> list:
    """Split on section headers — preserves author-defined units."""
    pattern = r'\n(?=#{1,4}\s|\n[A-Z][A-Z\s]{3,}\n)'
    sections = re.split(pattern, text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=50, length_function=len,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    results = []
    for sec_idx, section in enumerate(sections):
        section = section.strip()
        if len(section) < 50:
            continue
        
        # Extract header if present
        header_match = re.match(r'^(#{1,4}\s+.+|[A-Z][A-Z\s]{3,})\n', section)
        section_title = header_match.group(1).strip() if header_match else f"section_{sec_idx}"
        
        sub_chunks = splitter.split_text(section)
        for i, content in enumerate(sub_chunks):
            content = content.strip()
            if len(content) < 50:
                continue
            results.append({
                "page_content": content,
                "metadata": {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "section": section_title,
                    "chunk_global_id": f"{doc_id}_{sec_idx}_{i}",
                    "chunk_strategy": "header"
                }
            })
    return results


# ── Strategy: recursive char (general prose) ─────────────────────
def chunk_recursive(text: str, metadata: dict, doc_id: int,
                    chunk_size=500, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, separators=["\n\n", "\n", ". ", " ", ""]
    )
    results = []
    for i, content in enumerate(splitter.split_text(text)):
        content = content.strip()
        if len(content) < 50:
            continue
        results.append({
            "page_content": content,
            "metadata": {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": i,
                "chunk_global_id": f"{doc_id}_{i}",
                "chunk_strategy": "recursive"
            }
        })
    return results


# ── Strategy: sentence-aware (legal / dense technical) ───────────
def chunk_by_sentences(text: str, metadata: dict, doc_id: int,
                       max_sentences=5) -> list:
    """Group sentences into chunks — better for clause-heavy text."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    results = []
    buffer = []
    buf_len = 0
    chunk_idx = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        buffer.append(sent)
        buf_len += len(sent)
        
        if len(buffer) >= max_sentences or buf_len >= 600:
            content = ' '.join(buffer)
            if len(content) >= 50:
                results.append({
                    "page_content": content,
                    "metadata": {
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_id": chunk_idx,
                        "chunk_global_id": f"{doc_id}_{chunk_idx}",
                        "chunk_strategy": "sentence"
                    }
                })
                chunk_idx += 1
            # keep last sentence as overlap
            buffer = [buffer[-1]]
            buf_len = len(buffer[0])
    
    # flush remainder
    if buffer and buf_len >= 50:
        results.append({
            "page_content": ' '.join(buffer),
            "metadata": {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": chunk_idx,
                "chunk_global_id": f"{doc_id}_{chunk_idx}",
                "chunk_strategy": "sentence"
            }
        })
    return results


# ── Worker ────────────────────────────────────────────────────────
def _chunk_worker(args):
    page_content, metadata, doc_id, chunk_size, chunk_overlap, max_chars = args

    text = clean_text(page_content)
    if len(text) > max_chars:
        text = text[:max_chars]

    doc_type = detect_doc_type(text)
    metadata = {**metadata, "doc_type": doc_type}

    if doc_type in ("markdown", "research_paper"):
        return chunk_by_headers(text, metadata, doc_id)
    elif doc_type == "legal":
        return chunk_by_sentences(text, metadata, doc_id)
    else:
        return chunk_recursive(text, metadata, doc_id, chunk_size, chunk_overlap)


# ── Normalize ─────────────────────────────────────────────────────
def _normalize(doc, idx, chunk_size, chunk_overlap, max_chars):
    if isinstance(doc, dict):
        meta = doc.get("metadata", {})
        return (
            doc.get("text", ""),
            {
                "file_name": meta.get("file_name", "unknown"),
                "file_path": meta.get("file_path", ""),
                "num_pages": meta.get("num_pages", 1),
                "file_size": meta.get("file_size", 0),
                "source":    meta.get("source", "unknown"),
            },
            idx, chunk_size, chunk_overlap, max_chars
        )
    return (doc.page_content, doc.metadata, idx, chunk_size, chunk_overlap, max_chars)


# ── Main ──────────────────────────────────────────────────────────
def chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
    max_chars=100_000,
    max_workers=None,
    parallel_threshold=20
):
    args_list = [
        _normalize(doc, idx, chunk_size, chunk_overlap, max_chars)
        for idx, doc in enumerate(documents)
    ]

    all_chunks = []

    if len(documents) >= parallel_threshold:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in executor.map(_chunk_worker, args_list, chunksize=8):
                for item in batch:
                    all_chunks.append(Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"]
                    ))
    else:
        for args in args_list:
            for item in _chunk_worker(args):
                all_chunks.append(Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"]
                ))

    # Log strategy distribution
    strategies = {}
    for c in all_chunks:
        s = c.metadata.get("chunk_strategy", "unknown")
        strategies[s] = strategies.get(s, 0) + 1
    print(f"[INFO] {len(documents)} docs → {len(all_chunks)} chunks | strategies: {strategies}")
    return all_chunks


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from text_processing import load_folder

    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
    else:
        docs = load_folder(folder_path)
        if not docs:
            print("No documents found.")
        else:
            start = time.perf_counter()
            chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
            print(f"Done in {time.perf_counter() - start:.2f}s")

            for i in range(min(3, len(chunks))):
                print(f"\n--- Chunk {i+1} [{chunks[i].metadata.get('chunk_strategy')}] ---")
                print(f"Section : {chunks[i].metadata.get('section', 'n/a')}")
                print(f"Content : {chunks[i].page_content[:200]}...")
