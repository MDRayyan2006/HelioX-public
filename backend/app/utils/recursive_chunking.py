# def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
#     return chunks


# if __name__ == "__main__":
#     from text_processing import load_folder
#     from langchain_core.documents import Document
#     import os
    
#     folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
#     if os.path.exists(folder_path):
#         docs = load_folder(folder_path)
#         if docs:
#             print(f"\n\n=== Testing with PDF Files ({len(docs)} documents) ===")
            
#             # Convert to LangChain Documents
#             langchain_docs = []
#             for doc in docs:
#                 langchain_doc = Document(
#                     page_content=doc['text'],
#                     metadata=doc['metadata']
#                 )
#                 langchain_docs.append(langchain_doc)
            
#             # Chunk the documents
#             chunks = chunk_documents(langchain_docs, chunk_size=500, chunk_overlap=100)
            
#             print(f"\nGenerated {len(chunks)} total chunks")
            
#             # Display sample chunks
#             if chunks:
#                 num_samples = min(3, len(chunks))  # Show first 3 chunks
#                 for i in range(num_samples):
#                     print(f"\n--- Sample Chunk {i + 1} ---")
#                     print(f"Content: {chunks[i].page_content[:200]}...")
#                     print(f"Metadata: {chunks[i].metadata}")
#     else:
#         print(f"Uploads folder not found at: {folder_path}")

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


def _chunk_worker(args):
    page_content, metadata, doc_id, chunk_size, chunk_overlap, max_chars = args

    text = clean_text(page_content)
    if len(text) > max_chars:
        text = text[:max_chars]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    result = []
    for i, content in enumerate(splitter.split_text(text)):
        content = content.strip()
        if len(content) < 50:
            continue
        result.append({
            "page_content": content,
            "metadata": {
                **metadata,
                "doc_id": doc_id,
                "chunk_id": i,
                "chunk_global_id": f"{doc_id}_{i}",
            }
        })
    return result


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
    else:
        return (doc.page_content, doc.metadata, idx, chunk_size, chunk_overlap, max_chars)


def build_chunks(
    documents,
    chunk_size=500,
    chunk_overlap=100,
    max_chars=100_000,
    max_workers=None,
    parallel_threshold=20   # ← only use multiprocessing above this count
):
    args_list = [
        _normalize(doc, idx, chunk_size, chunk_overlap, max_chars)
        for idx, doc in enumerate(documents)
    ]

    all_chunks = []

    if len(documents) >= parallel_threshold:
        # ── Many docs: true CPU parallelism worth the spawn overhead
        print(f"[INFO] Parallel mode ({len(documents)} docs)")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch in executor.map(_chunk_worker, args_list, chunksize=8):
                for item in batch:
                    all_chunks.append(Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"]
                    ))
    else:
        # ── Few docs: plain loop is faster (no process spawn cost)
        print(f"[INFO] Single-thread mode ({len(documents)} docs)")
        for args in args_list:
            for item in _chunk_worker(args):
                all_chunks.append(Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"]
                ))

    print(f"[INFO] {len(args_list)} docs → {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    import time
    from text_processing import load_folder

    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
    else:
        start = time.perf_counter()
        docs = load_folder(folder_path)
        end = time.perf_counter()
        print(f"Loading documents completed in {end - start:.2f} seconds")
        if not docs:
            print("No documents found.")
        else:
            print(f"\n=== Processing {len(docs)} document(s) ===")
            start = time.perf_counter()

            chunks = build_chunks(
                docs,
                chunk_size=500,
                chunk_overlap=100,
                max_workers=os.cpu_count()
            )

            print(f"Chunking completed in {time.perf_counter() - start:.2f} seconds")
            print(f"Generated {len(chunks)} chunks\n")

            for i in range(len(chunks)):
                print(f"--- Chunk {i + 1} ---")
                print(f"Content : {chunks[i].page_content[:200]}...")
                print(f"Metadata: {chunks[i].metadata}\n")


